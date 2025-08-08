from main import train_model
from data import load_data
import argparse
import numpy as np
import pandas as pd

def run_grid_search(
    alpha_list,
    lr_list,
    nhead_time_list,
    nhead_stock_list,
    d_model_list,
    ff_list,
    use_cw_token_list,   # e.g. [False, True]
    use_cw_stock_list,   # e.g. [False, True]
    epoch_list,
    seeds=range(10),
    train_loader=None,
    valid_loader=None,
    test_loader=None,
    input_dim=None,
    token_embeddings_tensor=None,
    time_class_weights=None,   # [vocab_size]
    stock_class_weights=None,  # [2]
    stock_mode_list=['last','flatten'],
    model_save_dir="model",  # save best model,
    device='cuda'
):
    import numpy as np
    import os

    results=[]
    for alpha in alpha_list:
        for lr in lr_list:
            for nt in nhead_time_list:
                for ns in nhead_stock_list:
                    for dm in d_model_list:
                        for ff_dim in ff_list:
                            for use_cw_token in use_cw_token_list:
                                for use_cw_stock in use_cw_stock_list:
                                    for epoch in epoch_list:
                                        for stock_mode in stock_mode_list:
                                            acc_list=[]
                                            mcc_list=[]
                                            for s in seeds:
                                                test_acc, test_mcc= train_model(
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    test_loader=test_loader,
                                                    input_dim=input_dim,
                                                    token_embeddings_tensor=token_embeddings_tensor,
                                                    alpha=alpha,
                                                    epochs=epoch,
                                                    model_save_dir= model_save_dir,
                                                    d_model=dm,
                                                    nhead_time=nt,
                                                    nhead_stock=ns,
                                                    dim_feedforward=ff_dim,
                                                    learning_rate=lr,
                                                    seed=s,
                                                    do_debug_print=True,
                                                    stock_mode=stock_mode,
                                                    use_cw_token= use_cw_token,
                                                    use_cw_stock= use_cw_stock,
                                                    time_class_weights= time_class_weights,
                                                    stock_class_weights= stock_class_weights,
                                                    device=device
                                                )
                                                acc_list.append(test_acc)
                                                mcc_list.append(test_mcc)

                                            mean_acc= np.mean(acc_list)
                                            mean_mcc= np.mean(mcc_list)

                                            results.append({
                                                "alpha": alpha,
                                                "lr": lr,
                                                "nhead_time": nt,
                                                "nhead_stock": ns,
                                                "d_model": dm,
                                                "dim_feedforward": ff_dim,
                                                "use_cw_token": use_cw_token,
                                                "use_cw_stock": use_cw_stock,
                                                "epoch": epoch,
                                                "stock_mode": stock_mode,
                                                "acc": mean_acc,
                                                "mcc": mean_mcc
                                            })
                                            print(f"\n[GRID] alpha={alpha}, lr={lr}, nhead_time={nt}, nhead_stock={ns}, d_model={dm}, ff={ff_dim}, cw_token={use_cw_token}, cw_stock={use_cw_stock}, epoch={epoch}, stock_mode={stock_mode} => ACC={mean_acc:.4f}, MCC={mean_mcc:.4f}")
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/dataset.pkl')
    parser.add_argument('--output_path', type=str, default='grid_search_results.csv')
    parser.add_argument('--model_save_dir', type=str, default='model')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()

def main():
    args = parse_args()
    (
        train_loader,
        valid_loader,
        test_loader,
        input_dim,
        class_weights_updown,
        token_embeddings_tensor,
        time_class_weights,
        stock_class_weights
    ) = load_data(args.dataset)
    seed = args.seed
    device = args.device
    output_path = args.output_path
    model_save_dir = args.model_save_dir

    # (C) define hyper param grid
    alpha_list = [0.9, 0.7, 0.5]
    lr_list = [0.001]
    nhead_time_list = [1]
    nhead_stock_list = [4]
    d_model_list = [32, 64, 128, 256]
    ff_list = [256]
    seeds = range(5)
    stock_mode_list = ['last']
    use_cw_token_list = [False]
    use_cw_stock_list = [True, False]
    epoch_list = [30]

    results = run_grid_search(
        alpha_list=alpha_list,
        lr_list=lr_list,
        nhead_time_list=nhead_time_list,
        nhead_stock_list=nhead_stock_list,
        d_model_list=d_model_list,
        ff_list=ff_list,
        use_cw_token_list=use_cw_token_list,
        use_cw_stock_list=use_cw_stock_list,
        epoch_list=epoch_list,
        seeds=seeds,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        token_embeddings_tensor=token_embeddings_tensor,
        time_class_weights=time_class_weights,
        stock_class_weights=stock_class_weights,
        stock_mode_list=stock_mode_list,
        device=device,
        model_save_dir=model_save_dir
    )

    output = pd.DataFrame(results)
    output.to_csv(output_path, index=False)
    print(f"Grid search results are saved at {output_path}")



if __name__ == "__main__":
    main()