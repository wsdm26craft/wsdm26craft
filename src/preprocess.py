import os.path
import warnings
import pandas as pd
import numpy as np
import torch
import multiprocessing
import argparse
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from sklearn.cluster import KMeans
from multiprocessing import cpu_count
from collections import defaultdict

from data import TokenSequenceDataset, DateLevelDataset, DateLevelDataLoader, CustomDataLoader, save_data
from torch.utils.data import DataLoader
from collections import Counter

warnings.filterwarnings(action='ignore')

def calculate_log_return(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for each stock in the dataset
    :param price_data: price data
    :return: price data with log returns
    """
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.sort_values(by=['symbol', 'date'], inplace=True)

    # Calculate log returns
    price_data['log_return'] = price_data.groupby('symbol')['adjusted_close'].apply(lambda x: np.log(x / x.shift(1)))
    price_data.dropna(subset=['log_return'], inplace=True)
    return price_data[['date', 'symbol', 'log_return']]

def calculate_embedding_per_date(price_data: pd.DataFrame, date_pair: tuple) -> pd.DataFrame:
    """
    Calculate embeddings for each symbol at a given date
    :param price_data:
    :param date_pair:
    :return:
    """
    start_date, current_date = date_pair
    # Filter data for the past 60 trading days
    window_data = price_data[
        (price_data['date'] >= start_date) & (price_data['date'] < current_date)
    ]

    # Ensure we have 60 unique dates
    if len(window_data['date'].unique()) < 60:
        return None

    # Pivot to create a matrix of log returns with dates as rows and symbols as columns
    pivot_data = window_data.pivot(index='date', columns='symbol', values='log_return')

    # Drop columns (symbols) with missing data
    pivot_data.dropna(axis=1, inplace=True)

    # Ensure we have more than one symbol
    if pivot_data.shape[1] < 2:
        return None

    # Compute the correlation matrix
    corr_matrix = pivot_data.corr()
    # Fill NaN values with zeros
    corr_matrix.fillna(0, inplace=True)

    # Perform SVD
    svd = TruncatedSVD(n_components=2)
    embedding = svd.fit_transform(corr_matrix)

    # Store the embeddings for each symbol
    symbols = corr_matrix.index.tolist()
    date_embeddings = pd.DataFrame({
        'date': current_date,
        'symbol': symbols,
        'embedding_x': embedding[:, 0],
        'embedding_y': embedding[:, 1]
    })
    return date_embeddings


def adjust_prices(price_data: pd.DataFrame, *, verbose=False) -> pd.DataFrame:
    """
    Adjust the price and volume, then calculate the adjusted_volume. 
    After that, vectorize the price and volume movements over the past 5 days to create the columns 
    c_1 to c_4 and v_1 to v_4. For the c_t features, make sure to shift the date to t+4 so that 
    future information is not used.
    """
    if verbose:
        print(price_data)

  
    price_data['date'] = pd.to_datetime(price_data['date'])

    price_data = price_data.dropna(subset=['close', 'adjusted_close', 'volume'])
    price_data = price_data[(price_data['close'] != 0) & (price_data['adjusted_close'] != 0) & (price_data['volume'] != 0)]

    
    price_ratio = price_data['adjusted_close'] / price_data['close']

 
    price_data['adjusted_open'] = price_data['open'] * price_ratio
    price_data['adjusted_high'] = price_data['high'] * price_ratio
    price_data['adjusted_low'] = price_data['low'] * price_ratio
    price_data['adjusted_volume'] = price_data['volume'] * (price_data['close'] / price_data['adjusted_close'])

 
    price_data.sort_values(by=['symbol', 'date'], inplace=True)

 
    def create_features(group):
       
        group = group.reset_index(drop=True)

     
        group['adj_close_t0'] = group['adjusted_close']
        group['adj_volume_t0'] = group['adjusted_volume']

        
        for i in range(1, 5):
            group[f'adj_close_t{i}'] = group['adjusted_close'].shift(-i)
            group[f'adj_volume_t{i}'] = group['adjusted_volume'].shift(-i)

        for i in range(1, 5):
            group[f'c_{i}'] = group[f'adj_close_t{i}'] / group['adj_close_t0'] - 1
            group[f'v_{i}'] = group[f'adj_volume_t{i}'] / group['adj_volume_t0'] - 1

            
            group[f'v_{i}'] = group[f'v_{i}'].clip(upper=5)

       
        group['date'] = group['date'].shift(-4)

        
        group = group[:-4]

        return group

    price_data = price_data.groupby('symbol', group_keys=False).apply(create_features)
    price_data.reset_index(drop=True, inplace=True)

    
    price_data = price_data[['symbol', 'date', 'c_1', 'c_2', 'c_3', 'c_4']]

  
    price_data.dropna(subset=['date', 'c_1', 'c_2', 'c_3', 'c_4'], inplace=True)


    price_data_cleaned = clip_outliers(price_data, z_threshold=4)

    return price_data_cleaned


def clip_outliers(df, z_threshold=4):
    numeric_cols = df.select_dtypes(include=[np.number]).columns 
    df_clipped = df.copy()
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std == 0:
            continue 

        z_scores = (df[col] - col_mean) / col_std
        upper_mask = z_scores > z_threshold
        lower_mask = z_scores < -z_threshold

        df_clipped.loc[upper_mask, col] = col_mean + z_threshold * col_std
        df_clipped.loc[lower_mask, col] = col_mean - z_threshold * col_std

    return df_clipped

def get_fixed_representations(data_tensor):
    with torch.no_grad():
        return data_tensor.cpu().numpy()


def process_date(date, embedding_df, token_embeddings_cpu, grouped_tokens, grouped_dates):
    """
    Process a single date, calculating auxiliary embeddings for each symbol on that date.
    Returns a dictionary of {(date, symbol): aux_embedding} for that date.
    """
    aux_embedding_subdict = {}

    date_embeddings = embedding_df[embedding_df['date'] == date]
    symbols_on_date = date_embeddings['symbol'].values
    embeddings_on_date = date_embeddings[['embedding_x', 'embedding_y']].values

    symbol_to_embedding = dict(zip(symbols_on_date, embeddings_on_date))

    for symbol in symbols_on_date:
        if symbol not in grouped_tokens or date not in grouped_dates[symbol]:
            continue

        try:
            token_index = grouped_dates[symbol].index(date)
            token_id = grouped_tokens[symbol][token_index]
            token_embedding = token_embeddings_cpu[token_id]
        except ValueError:
            continue  

        symbol_embedding = symbol_to_embedding[symbol]

        similarities = []
        for other_symbol in symbols_on_date:
            if other_symbol == symbol:
                continue
            other_embedding = symbol_to_embedding[other_symbol]
            
            distance = np.linalg.norm(symbol_embedding - other_embedding)
            similarity = np.exp(-distance)
            similarities.append((similarity, other_symbol))

        
        total_similarity = sum([sim[0] for sim in similarities])
        if total_similarity > 0:
            normalized_weights = [(sim[0] / total_similarity, sim[1]) for sim in similarities]
        else:
            normalized_weights = [(0, sim[1]) for sim in similarities]

        
        aux_embedding = np.zeros_like(token_embedding)

        for weight, other_symbol in normalized_weights:
            try:
                other_token_index = grouped_dates[other_symbol].index(date)
                other_token_id = grouped_tokens[other_symbol][other_token_index]
                other_token_embedding = token_embeddings_cpu[other_token_id]
                aux_embedding += weight * other_token_embedding
            except ValueError:
                continue  

        
        aux_embedding_subdict[(date, symbol)] = aux_embedding

    return aux_embedding_subdict


def calculate_aux_embeddings_optimized(embedding_df, token_embeddings, grouped_tokens, grouped_dates):
    """
    Calculate auxiliary embeddings using existing grouped tokens and dates, and embedding_df.
    """
    # Ensure embedding_df is a DataFrame with necessary columns
    required_columns = {'date', 'symbol', 'embedding_x', 'embedding_y'}
    if not required_columns.issubset(embedding_df.columns):
        raise ValueError(f"embedding_df must contain columns: {required_columns}")

    # Convert token_embeddings to CPU numpy array
    token_embeddings_cpu = token_embeddings.cpu().numpy()

    # Extract all unique dates from grouped_dates
    all_dates = set()
    for dates in grouped_dates.values():
        all_dates.update(dates)
    unique_dates = sorted(all_dates)

    # Set up parallel processing
    num_jobs = max(cpu_count() - 1, 1)  # Leave one CPU free
    print(f"Using {num_jobs} parallel jobs.")

    # Use joblib's Parallel and delayed to process dates in parallel
    results = Parallel(n_jobs=num_jobs, backend='loky')(
        delayed(process_date)(date, embedding_df, token_embeddings_cpu, grouped_tokens, grouped_dates)
        for date in tqdm(unique_dates, desc="Processing Dates")
    )

    # Merge all sub-dictionaries into one
    aux_embedding_dict = {}
    for subdict in results:
        aux_embedding_dict.update(subdict)

    return aux_embedding_dict

def group_tokens_by_symbol(symbol_dates, tokens):
    """
    Groups a flat array of tokens by symbol based on symbol_dates, ensuring the token count matches the date count.

    Args:
    - symbol_dates: A list of (symbol, date) pairs.
    - tokens: A flat list or array of token indices.

    Returns:
    - grouped_tokens: A dictionary {symbol: [tokens]} grouped by symbol.
    - grouped_dates: A dictionary {symbol: [dates]} grouped by symbol.
    """

    grouped_tokens = defaultdict(list)
    grouped_dates = defaultdict(list)

    # Ensure tokens are indices (integers)
    tokens = [int(token) for token in tokens]

    # Iterate over each symbol-date pair and group tokens accordingly
    for i, (symbol, date) in enumerate(symbol_dates):
        if i >= len(tokens):
            print(
                f"Warning: Not enough tokens for symbol {symbol}. Skipping. (Symbol Dates: {len(symbol_dates)}, Tokens: {len(tokens)})")
            continue  # If we run out of tokens, skip this symbol

        grouped_tokens[symbol].append(tokens[i])
        grouped_dates[symbol].append(date)

    return grouped_tokens, grouped_dates


def create_symbol_dates(df):
    """
    Create a dictionary mapping symbols to a list of dates from the given DataFrame.

    Args:
    - df: DataFrame with 'symbol' and 'date' columns

    Returns:
    - symbol_dates: Dictionary {symbol: [date1, date2, ...]}
    """
    symbol_dates = {}
    grouped = df.groupby('symbol')
    for symbol, group in grouped:
        symbol_dates[symbol] = group['date'].tolist()  # List of dates for this symbol
    return symbol_dates


def create_token_sequences(symbol_data, symbol_dates, seq_len):
    token_sequences = {}
    date_sequences = {}

    for symbol in symbol_data:
        data = symbol_data[symbol]
        dates = symbol_dates[symbol]

        if len(data) <= seq_len:
            continue  # Skip if not enough data

        sequences = []
        date_seqs = []

        for i in range(len(data) - seq_len):
            input_seq = data[i:i + seq_len]
            target_seq = data[i + 1:i + seq_len + 1]  # Targets are inputs shifted by one
            sequences.append((input_seq, target_seq))
            date_seq = dates[i:i + seq_len]
            date_seqs.append(date_seq)

        token_sequences[symbol] = sequences
        date_sequences[symbol] = date_seqs

    return token_sequences, date_sequences


def add_updown_label(df):
    """
    df: must have columns ['symbol','date','c_3','c_4'].
    1) 정렬 => groupby => shift(-1) => (next_c4 - next_c3)>0 => 1 else 0
    2) updown_label 컬럼
    """
    df = df.sort_values(by=['symbol','date']).reset_index(drop=True)
    next_c4 = df.groupby('symbol')['c_4'].shift(-1)
    next_c3 = df.groupby('symbol')['c_3'].shift(-1)

    df['c4_minus_c3_future'] = next_c4 - next_c3
    df['updown_label'] = (df['c4_minus_c3_future']>0).astype(int)
    # drop last row for each symbol if shift is NaN
    df = df.dropna(subset=['c4_minus_c3_future']).reset_index(drop=True)
    return df

def create_date_level_2d_data(
    token_sequences, date_sequences,
    aux_embedding_dict,
    token_embedding_matrix, # shape [vocab_size, emb_dim]
    updown_map,
    seq_len, aux_dim
):
    """
    Returns data_list: each item = { 'date':..., 'inputs':[n,l,d], 'targets':[n,l], 'updowns':[n] }
    ...
    """
    data_list = []
    date_dict = {}

    emb_dim = token_embedding_matrix.shape[1]

    for sym in token_sequences:
        seq_pairs = token_sequences[sym]
        dseqs = date_sequences[sym]
        for (inp_tokens, tgt_tokens), d_seq in zip(seq_pairs, dseqs):
            inp_emb = token_embedding_matrix[inp_tokens] # [l,emb_dim]
            aux_list = []
            for d_ in d_seq:
                if (d_, sym) in aux_embedding_dict:
                    arr_ = torch.tensor(aux_embedding_dict[(d_, sym)], dtype=torch.float32)
                else:
                    arr_ = torch.zeros(aux_dim, dtype=torch.float32)
                aux_list.append(arr_)
            aux_arr = torch.stack(aux_list, dim=0) # [l, aux_dim]
            aux_arr = aux_arr.to(inp_emb.device)    # ← 이 줄 추가!
            combined = torch.cat([inp_emb, aux_arr], dim=-1) # [l, emb_dim+aux_dim]

            # up/down label => last date
            last_date = d_seq[-1]
            up_label = updown_map.get((sym, last_date), 0)

            if last_date not in date_dict:
                date_dict[last_date] = []
            date_dict[last_date].append((sym, combined, tgt_tokens, up_label))

    for date_key in sorted(date_dict.keys()):
        items = date_dict[date_key]
        if len(items)==0:
            continue
        arr_2d=[]
        tgt_2d=[]
        up_1d=[]
        for (sym, combined_np, tgt_tokens, up_label) in items:
            arr_2d.append(combined_np.cpu().numpy())  # CPU numpy
            tgt_2d.append(tgt_tokens)
            up_1d.append(up_label)
        arr_2d = np.stack(arr_2d, axis=0) # [n,l,d]
        tgt_2d = np.stack(tgt_2d, axis=0) # [n,l]
        up_1d  = np.array(up_1d, dtype=np.int64) # [n]
        data_list.append({
            'date': date_key,
            'inputs': arr_2d,  # shape [n,l,d]
            'targets':tgt_2d,  # [n,l]
            'updowns':up_1d    # [n]
        })
    return data_list

def make_time_class_weights(train_tokens, n_clusters, device='cpu'):
    """
    - train_tokens: list or array of token indices (0 <= index < n_clusters)
    - n_clusters: total number of token classes
    - returns time_class_weights: shape=[n_clusters], where weight[i] = inverse frequency
    """
    from collections import Counter

    # 1) Count frequency
    class_counts = Counter(train_tokens)
    # 2) Initialize a tensor for class counts
    class_counts_tensor = torch.zeros(n_clusters)
    # 3) Fill
    for cls, count in class_counts.items():
        class_counts_tensor[cls] = count

    total_count = class_counts_tensor.sum().float()
    # 4) Compute class weights = inverse frequency
    #    weight[i] = total_count / (n_clusters * class_counts[i])
    time_class_weights = total_count / (n_clusters * (class_counts_tensor + 1e-10))
    time_class_weights = time_class_weights.to(device)

    return time_class_weights

def prepare_data_once(
    train_df, valid_df, test_df,
    train_token_sequences, train_date_sequences,
    valid_token_sequences, valid_date_sequences,
    test_token_sequences, test_date_sequences,
    train_aux_embedding_dict, valid_aux_embedding_dict, test_aux_embedding_dict,
    token_embeddings,
    seq_len=20,
    aux_dim=4
):
    # 1) label
    train_df_labeled = add_updown_label(train_df)
    valid_df_labeled = add_updown_label(valid_df)
    test_df_labeled  = add_updown_label(test_df)

    def build_updown_map(df_):
        m_={}
        for idx, row in df_.iterrows():
            sym= row['symbol']
            dt=  row['date']
            lbl= row['updown_label']
            m_[(sym, dt)] = lbl
        return m_

    up_map_train = build_updown_map(train_df_labeled)
    up_map_valid = build_updown_map(valid_df_labeled)
    up_map_test  = build_updown_map(test_df_labeled)

    # (A) up/down 분포 from train_df_labeled
    num_up   = (train_df_labeled['updown_label']==1).sum()
    num_down = (train_df_labeled['updown_label']==0).sum()
    if num_up==0 or num_down==0:
        print("Warning: one class (up/down) not found in train. set weights=1,1")
        cw_up, cw_down= 1.0,1.0
    else:
        total= num_up + num_down
        cw_up   = total/(2.0*num_up)
        cw_down = total/(2.0*num_down)
    class_weights_updown= (cw_down, cw_up)

    emb_dim= token_embeddings.shape[1]
    input_dim= emb_dim+ aux_dim

    # create 2d data
    train_list= create_date_level_2d_data(
        train_token_sequences, train_date_sequences,
        train_aux_embedding_dict,
        token_embeddings,
        up_map_train,
        seq_len, aux_dim
    )
    valid_list= create_date_level_2d_data(
        valid_token_sequences, valid_date_sequences,
        valid_aux_embedding_dict,
        token_embeddings,
        up_map_valid,
        seq_len, aux_dim
    )
    test_list= create_date_level_2d_data(
        test_token_sequences, test_date_sequences,
        test_aux_embedding_dict,
        token_embeddings,
        up_map_test,
        seq_len, aux_dim
    )

    train_ds= DateLevelDataset(train_list)
    valid_ds= DateLevelDataset(valid_list)
    test_ds=  DateLevelDataset(test_list)

    train_loader= DateLevelDataLoader(train_ds, batch_size=1, shuffle=True)
    valid_loader= DateLevelDataLoader(valid_ds, batch_size=1, shuffle=False)
    test_loader= DateLevelDataLoader(test_ds,  batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader, input_dim, class_weights_updown

def preprocess_price_data(price_data: pd.DataFrame, *, verbose=False) -> tuple:
    """
    Preprocess the data
    :param price_data: price data
    :return: preprocessed data
    """
    # Calculate log returns
    log_return = calculate_log_return(price_data)

    unique_dates = log_return['date'].drop_duplicates().sort_values().tolist()
    all_symbols = log_return['symbol'].unique()

    # Prepare date pairs (start_date, current_date) for 60 trading days window
    date_pairs = [
        (unique_dates[i - 60], unique_dates[i]) for i in range(60, len(unique_dates))
    ]

    # Parallel processing over date pairs
    num_cores = min(30, multiprocessing.cpu_count())

    embeddings_list = Parallel(n_jobs=num_cores)(
        delayed(calculate_embedding_per_date)(log_return, date_pair) for date_pair in tqdm(date_pairs, desc="Processing Dates")
    )

    # Filter out None results and concatenate
    embeddings_list = [df for df in embeddings_list if df is not None]
    if len(embeddings_list) == 0:
        ValueError("No embeddings were computed. Please check the data and the date ranges.")

    embedding_df = pd.concat(embeddings_list, ignore_index=True)
    if verbose:
        print(embedding_df)
    prices_df = adjust_prices(price_data, verbose=verbose).drop_duplicates()
    return embedding_df, prices_df

def preprocess_price_data_without_scaling(prices_df, batch_size=128, start_date="2020-01-01", valid_date="2024-01-01", test_date="2024-07-01", *, verbose=False):
    """
    Preprocess data with c_3, c_4 and add c_4_minus_c_3 feature.
    Remove symbols with excessive outliers but return features in unscaled state.
    """
    # 1. Convert date column to datetime and sort
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df.sort_values(by='date', inplace=True)

    # 2. Add the c_4_minus_c_3 feature
#     merged_df['c_4_minus_c_3'] = merged_df['c_4'] - merged_df['c_3']

    # 3. Select necessary columns
    features = prices_df.drop(columns=['symbol', 'date', 'c_3', 'c_4'])

    # 4. Split the data by date
    train_cutoff = valid_date
    valid_cutoff = test_date

    train_df = prices_df[(prices_df['date'] >= start_date) & (prices_df['date'] < train_cutoff)]
    valid_df = prices_df[(prices_df['date'] >= train_cutoff) & (prices_df['date'] < valid_cutoff)]
    test_df = prices_df[prices_df['date'] >= valid_cutoff]

    # 5. Identify outliers based on a threshold (e.g., absolute value greater than a threshold)
    outlier_threshold = 4  # Adjust the threshold as needed

    # 6. Count outliers by symbol for the entire dataset
    symbol_outlier_counts = prices_df.groupby('symbol').apply(
        lambda df: ((df.drop(columns=['symbol', 'date', 'c_3', 'c_4']) > outlier_threshold).sum().sum())
    ).sort_values(ascending=False)

    # 7. Identify symbols to remove (outlier count >= 1)
    symbols_to_remove = symbol_outlier_counts[symbol_outlier_counts >= 1].index
    print("Symbols with excessive outliers to be removed:", symbols_to_remove)

    # 8. Remove identified symbols from the dataset
    merged_df = prices_df[~prices_df['symbol'].isin(symbols_to_remove)]
    train_df = train_df[~train_df['symbol'].isin(symbols_to_remove)]
    valid_df = valid_df[~valid_df['symbol'].isin(symbols_to_remove)]
    test_df = test_df[~test_df['symbol'].isin(symbols_to_remove)]

    # 9. Extract features without scaling
    X_train = train_df.drop(columns=['symbol', 'date']).values
    X_valid = valid_df.drop(columns=['symbol', 'date']).values
    X_test = test_df.drop(columns=['symbol', 'date']).values

    # Since we are not scaling, we don't need to return scalers
    scalers = None

    return X_train, X_valid, X_test, scalers, train_df, valid_df, test_df, symbol_outlier_counts

def preprocess_data(
        data_path,
        output_path=None,
        batch_size=128,
        train_date="2020-01-01",
        valid_date="2024-01-01",
        test_date="2024-07-01",
        n_clusters=200,*,
        verbose=False):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    prices_df = pd.read_csv(data_path)

    embedding_data, prices_data = preprocess_price_data(prices_df, verbose=verbose)
    X_train, X_valid, X_test, scalers, train_df, valid_df, test_df, symbol_outlier_counts = preprocess_price_data_without_scaling(prices_data, batch_size, train_date, valid_date, test_date, verbose=verbose)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Assuming X_train, X_valid, and X_test are your preprocessed high-dimensional data arrays
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_reps = get_fixed_representations(X_train_tensor)
    valid_reps = get_fixed_representations(X_valid_tensor)
    test_reps = get_fixed_representations(X_test_tensor)

    # Combine train and valid representations for K-means training
    train_valid_reps = np.concatenate((train_reps, valid_reps), axis=0)

    # K-means clustering on combined train and valid data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(train_valid_reps)

    # Get token embeddings (cluster centers)
    token_embeddings = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)  # Fixed embeddings

    # Assign tokens (cluster labels) to each data point
    train_tokens = kmeans.predict(train_reps)
    valid_tokens = kmeans.predict(valid_reps)
    test_tokens = kmeans.predict(test_reps)

    # Print shapes for confirmation
    if verbose:
        print("Train tokens shape:", train_tokens.shape)
        print("Valid tokens shape:", valid_tokens.shape)
        print("Test tokens shape:", test_tokens.shape)

    # Group tokens based on symbol and date
    grouped_train_tokens, grouped_train_dates = group_tokens_by_symbol(list(zip(train_df['symbol'], train_df['date'])),
                                                                       train_tokens)
    grouped_valid_tokens, grouped_valid_dates = group_tokens_by_symbol(list(zip(valid_df['symbol'], valid_df['date'])),
                                                                       valid_tokens)
    grouped_test_tokens, grouped_test_dates = group_tokens_by_symbol(list(zip(test_df['symbol'], test_df['date'])),
                                                                     test_tokens)

    # Create aux_embedding_dict
    train_aux_embedding_dict = calculate_aux_embeddings_optimized(embedding_data, token_embeddings, grouped_train_tokens,
                                                                  grouped_train_dates)
    valid_aux_embedding_dict = calculate_aux_embeddings_optimized(embedding_data, token_embeddings, grouped_valid_tokens,
                                                                  grouped_valid_dates)
    test_aux_embedding_dict = calculate_aux_embeddings_optimized(embedding_data, token_embeddings, grouped_test_tokens,
                                                                 grouped_test_dates)

    # Define the sequence length
    seq_len = 20  # Adjust as needed
    BATCH_SIZE = 256
    aux_embedding_dim = 4

    # Create token sequences and date sequences for each symbol
    train_token_sequences, train_date_sequences = create_token_sequences(grouped_train_tokens, grouped_train_dates,
                                                                         seq_len)
    valid_token_sequences, valid_date_sequences = create_token_sequences(grouped_valid_tokens, grouped_valid_dates,
                                                                         seq_len)
    test_token_sequences, test_date_sequences = create_token_sequences(grouped_test_tokens, grouped_test_dates, seq_len)

    # Assume token_embeddings is your token embedding matrix as a NumPy array
    token_embedding_matrix = token_embeddings  # [vocab_size, embedding_dim]

    # Create the datasets
    train_token_dataset = TokenSequenceDataset(
        token_sequences=train_token_sequences,
        symbol_dates=train_date_sequences,
        aux_embedding_dict=train_aux_embedding_dict,
        seq_len=seq_len,
        aux_embedding_dim=aux_embedding_dim,
        token_embedding_matrix=token_embedding_matrix
    )

    valid_token_dataset = TokenSequenceDataset(
        token_sequences=valid_token_sequences,
        symbol_dates=valid_date_sequences,
        aux_embedding_dict=valid_aux_embedding_dict,
        seq_len=seq_len,
        aux_embedding_dim=aux_embedding_dim,
        token_embedding_matrix=token_embedding_matrix
    )

    test_token_dataset = TokenSequenceDataset(
        token_sequences=test_token_sequences,
        symbol_dates=test_date_sequences,
        aux_embedding_dict=test_aux_embedding_dict,
        seq_len=seq_len,
        aux_embedding_dim=aux_embedding_dim,
        token_embedding_matrix=token_embedding_matrix
    )

    # DataLoader definitions
    train_token_loader = CustomDataLoader(
        train_token_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_token_loader = CustomDataLoader(
        valid_token_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_token_loader = CustomDataLoader(
        test_token_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Dataset sizes
    if verbose:
        print(f"Train dataset size: {len(train_token_dataset)}")
        print(f"Valid dataset size: {len(valid_token_dataset)}")
        print(f"Test dataset size: {len(test_token_dataset)}")

    # Define the number of clusters (classes)
    vocab_size = n_clusters  # Assuming this is set to 200, same as n_clusters in your clustering

    # Count the frequency of each class in train_tokens
    class_counts = Counter(train_tokens)

    # Initialize a tensor for class counts
    class_counts_tensor = torch.zeros(vocab_size)

    # Fill in class counts
    for cls, count in class_counts.items():
        class_counts_tensor[cls] = count

    # Calculate the total count of all targets
    total_count = class_counts_tensor.sum().float()

    # Compute class weights as inverse frequency
    class_weights = total_count / (vocab_size * (class_counts_tensor + 1e-10))  # Add epsilon to avoid division by zero
    class_weights = class_weights.to(device)  # Move to device if necessary

    if verbose:
        print("Class weights calculated:", class_weights)

    # (A) token_embeddings_tensor
    token_embeddings_tensor = torch.tensor(token_embeddings, dtype=torch.float32).to(device)

    # (B) prepare data once
    train_loader, valid_loader, test_loader, input_dim, class_weights_updown = prepare_data_once(
        train_df, valid_df, test_df,
        train_token_sequences, train_date_sequences,
        valid_token_sequences, valid_date_sequences,
        test_token_sequences, test_date_sequences,
        train_aux_embedding_dict, valid_aux_embedding_dict, test_aux_embedding_dict,
        token_embeddings,
        seq_len=20,
        aux_dim=4
    )

    # (C) class weights
    time_class_weights = make_time_class_weights(train_tokens, n_clusters, device=device)
    stock_class_weights = torch.tensor(list(class_weights_updown), dtype=torch.float32, device=device)

    if not output_path:
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        file_name = os.path.basename(data_path).split('.')[0]
        output_path = os.path.join('dataset', f'{file_name}_{train_date}_{valid_date}_{test_date}.pkl'.replace('-', '_'))
    save_data(
        path=output_path,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        token_embeddings_tensor=token_embeddings_tensor,
        input_dim=input_dim,
        class_weights_updown=class_weights_updown,
        time_class_weights=time_class_weights,
        stock_class_weights=stock_class_weights
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for training')
    parser.add_argument('--data_path', type=str, default='data/csi300.csv', help='Path to the data file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the preprocessed data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--train_date', type=str, default='2020-01-01', help='Training data cutoff date')
    parser.add_argument('--valid_date', type=str, default='2024-01-01', help='Validation data cutoff date')
    parser.add_argument('--test_date', type=str, default='2024-07-01', help='Test data cutoff date')
    parser.add_argument('--n_clusters', type=int, default=200, help='Number of clusters for K-means')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    batch_size = args.batch_size
    train_date = args.train_date
    valid_date = args.valid_date
    test_date = args.test_date
    n_clusters = args.n_clusters
    verbose = args.verbose
    preprocess_data(
        data_path, output_path, batch_size, train_date, valid_date, test_date, n_clusters, verbose=verbose
    )

if __name__ == '__main__':
    main()
