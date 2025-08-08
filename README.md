# Centroid-based Randomness Smoothing Approach for Stock Forecasting with Transformer Architecture

This project is the implementation of the paper "Centroid-based Randomness Smoothing Approach for Stock Forecasting with Transformer Architecture" (submitted to WSDM 2026).

We publicly release a [demo page](https://embedding-method.com) where users can query any ticker and date in the SP500 (US) dataset to view its most similar stocks, thereby illustrating the quality of our stock embeddings.

## Prerequisites

Our implementation is based on Python 3.8. The required packages are listed in the `requirements.txt` file. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation

You can prepare the data by running the following command:

```bash
cd src
python preprocess.py --data_path <path to the data> --output_path <path to save the output>
```

input data should be in the following format:

```csv
symbol,date,open,high,low,close,volume,adjusted_close
2010-01-04,30.62,31.1,30.59,30.95,123432400
...
```

output data will be a pickle file that contains necessary data for training and testing.

## Training

You can train the model by running the following command:

```bash
cd src
python train.py --dataset <path to the dataset>
```

You can also configure hyperparameters by passing arguments to the script. For example:

```bash
python train.py --dataset <path to the dataset> --epochs 100 --learning_rate 0.0001
```


## Grid Search

You can perform grid search by running the following command:

```bash
cd src
python run.py --dataset <path to the dataset>
```

## Testing

You can test the model by running the following command:

```bash
cd src
python test.py --dataset <path to the dataset> --model_path <path to the model>
```

