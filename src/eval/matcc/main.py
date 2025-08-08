MARKET_INDEX = {
    'cn': 'csi300',
    'us': 'sp500',
    'jp': 'ni225',
    'eu': 'stoxx50'
}

MARKET_INDEX_TICKER = {
    'csi300': '000300.SS',
    'stoxx50': '^STOXX50E',
    'sp500': '^GSPC',
    'ni225': '^N225'
}

ONLY_BACKTEST = True

import os
import sys
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.contrib.workflow.record_temp import SignalAccMccRecord
from qlib.tests.data import GetData
import yaml
import pprint as pp
import numpy as np
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Missing Input: region, method, seed')

    if len(sys.argv) < 3:
        raise ValueError('Missing Input: method, seed')

    if len(sys.argv) < 4:
        raise ValueError('Missing Input: seed')

    region = sys.argv[1]
    method = sys.argv[2]
    market = MARKET_INDEX[region]
    market_index = MARKET_INDEX_TICKER[market]
    seed = int(sys.argv[3])

    current_dir = os.getcwd()
    qlib_data_dir = f'../../../data/qlib_data/{region}_{method}_data'

    print(f'{region} {method} seed: {seed} - ONLY BACKTEST: {ONLY_BACKTEST}')
    GetData().qlib_data(target_dir=qlib_data_dir, region=region, exists_skip=True)
    qlib.init(provider_uri=qlib_data_dir, region=region)
    with open(f"./master_{method}_{market}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = os.path.join(current_dir, f'{method}_{market}_handler.pkl'.replace('-', ''))
    if not os.path.exists(h_path):
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print('Save preprocessed data to', h_path)
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config['task']["dataset"])

    print("Train segment:", dataset.segments.get("train"))
    print("Valid segment:", dataset.segments.get("valid"))
    print("Test segment:", dataset.segments.get("test"))

    if not os.path.exists('./model'):
        os.mkdir("./model")

    config['task']["model"]['kwargs']["seed"] = seed
    model = init_instance_by_config(config['task']["model"])

    # start exp
    if not ONLY_BACKTEST:
        model.fit(dataset=dataset)
    else:
        model.load_model(f"./model/master_{method}_{market}_{seed}.pkl")

    with R.start(experiment_name=f"workflow_seed{seed}"):
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SignalAccMccRecord(recorder)
        sar.generate()

        metrics = recorder.list_metrics()

        if not os.path.exists(f'./result/{method}_{market}'):
            os.makedirs(f'./result/{method}_{market}')
        if not os.path.exists(f'./result/{method}_{market}/{seed}'):
            os.makedirs(f'./result/{method}_{market}/{seed}')
        with open(os.path.join(f'../result/matcc_{method}_{market}_{seed}.txt')) as file:
            file.write(f"{metrics.get('ACC')},{metrics.get('MCC')}")