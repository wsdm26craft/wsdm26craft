from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle

class TokenSequenceDataset(Dataset):
    def __init__(self, token_sequences, symbol_dates, aux_embedding_dict, seq_len, aux_embedding_dim, token_embedding_matrix):
        self.inputs = []
        self.targets = []
        self.symbols = []
        self.dates = []
        self.aux_embeddings = []
        self.seq_len = seq_len
        self.aux_embedding_dim = aux_embedding_dim
        self.embedding_dim = token_embedding_matrix.shape[1]
        self.token_embedding_matrix = token_embedding_matrix

        min_aux_date = min(date for date, _ in aux_embedding_dict.keys())

        for symbol in token_sequences:
            sequences = token_sequences[symbol]        # List of (input_seq_tokens, target_seq_tokens)
            date_seqs = symbol_dates[symbol]           # List of date sequences

            for idx, ((input_seq_tokens, target_seq_tokens), date_seq) in enumerate(zip(sequences, date_seqs)):
                if date_seq[0] < min_aux_date:
                    continue 

                
                input_seq = self.token_embedding_matrix[input_seq_tokens]    # [seq_len, embedding_dim]
                target_seq = self.token_embedding_matrix[target_seq_tokens]  # [seq_len, embedding_dim]

                aux_embeds = []
                for date in date_seq:
                    aux_embed = aux_embedding_dict.get((date, symbol), np.zeros(self.aux_embedding_dim))
                    if not isinstance(aux_embed, torch.Tensor):
                        aux_embed = torch.tensor(aux_embed, dtype=torch.float32)
                    aux_embeds.append(aux_embed)
                aux_embeds = torch.stack(aux_embeds)  # [seq_len, aux_embedding_dim]

                
                self.inputs.append(torch.tensor(input_seq, dtype=torch.float32))    # [seq_len, embedding_dim]
                self.targets.append(torch.tensor(target_seq, dtype=torch.float32))  # [seq_len, embedding_dim]
                self.symbols.append(symbol)
                self.dates.append(date_seq)
                self.aux_embeddings.append(aux_embeds)  # [seq_len, aux_embedding_dim]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],          # [seq_len, embedding_dim]
            self.targets[idx],         # [seq_len, embedding_dim]
            self.symbols[idx],
            self.dates[idx],
            self.aux_embeddings[idx]   # [seq_len, aux_embedding_dim]
        )

class DateLevelDataset(Dataset):
    """
    data_list: each item => { 'date':..., 'inputs':[n,l,d], 'targets':[n,l], 'updowns':[n] }
    returns => (inputs, targets, updowns, date)
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        inputs  = torch.tensor(item['inputs'],  dtype=torch.float32)  # [n,l,d]
        targets = torch.tensor(item['targets'], dtype=torch.long)     # [n,l]
        updowns = torch.tensor(item['updowns'], dtype=torch.long)     # [n]
        date_   = item['date']
        return inputs, targets, updowns, date_


class DateLevelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        """
        Default batch_size=1 => each item is one date
        If we combine => shape [bs,n,l,d]
        """
        in_list, tgt_list, up_list, date_list = [], [], [], []
        for (inp, tgt, up, dt) in batch:
            in_list.append(inp)
            tgt_list.append(tgt)
            up_list.append(up)
            date_list.append(dt)
        inputs_stacked = torch.stack(in_list, dim=0)  # [bs,n,l,d]
        targets_stacked = torch.stack(tgt_list, dim=0)  # [bs,n,l]
        updowns_stacked = torch.stack(up_list, dim=0)  # [bs,n]
        return inputs_stacked, targets_stacked, updowns_stacked, date_list

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        inputs = torch.stack([item[0] for item in batch])          # [batch_size, seq_len, embedding_dim]
        targets = torch.stack([item[1] for item in batch])         # [batch_size, seq_len, embedding_dim]
        symbols = [item[2] for item in batch]
        dates = [item[3] for item in batch]
        aux_embeddings = torch.stack([item[4] for item in batch])  # [batch_size, seq_len, aux_embedding_dim]

        return inputs, targets, symbols, dates, aux_embeddings

def save_data(path, train_loader, valid_loader, test_loader, input_dim, class_weights_updown, token_embeddings_tensor, time_class_weights, stock_class_weights):
    with open(path, 'wb') as f:
        pickle.dump([train_loader, valid_loader, test_loader, input_dim, class_weights_updown, token_embeddings_tensor, time_class_weights, stock_class_weights], f)

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

