from functorch.dim import use_c

from model import CraftModel, Trainer
from data import load_data
import torch
import argparse



def train_model(
    train_loader, valid_loader, test_loader,
    input_dim,
    token_embeddings_tensor,  # shape [vocab_size, emb_dim], for embedding mae
    alpha=0.5,
    epochs=30,
    d_model=128,
    nhead_time=1,
    nhead_stock=4,
    dim_feedforward=128,
    learning_rate=0.001,
    use_cw_token=False,
    use_cw_stock=False,
    time_class_weights=None,
    stock_class_weights=None,
    seed=0,
    do_debug_print=True,
    model_save_path="model/craft.pth",
    model_save_dir="model",
    stock_mode='all',
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(device) == str:
        device = torch.device(device)
    vocab_size = token_embeddings_tensor.shape[0]
    model = CraftModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead_time=nhead_time,
        nhead_stock=nhead_stock,
        dim_feedforward=dim_feedforward,
        vocab_size=vocab_size,
        stock_mode=stock_mode
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, device)
    best_acc, best_mcc = trainer.fit(
        train_loader,
        valid_loader,
        test_loader,
        token_embeddings_tensor,
        alpha=alpha,
        epochs=epochs,
        seed=seed,
        do_debug_print=do_debug_print,
        model_save_dir=model_save_dir,
        use_cw_token=use_cw_token,
        use_cw_stock=use_cw_stock,
        time_class_weights=time_class_weights,
        stock_class_weights=stock_class_weights
    )
    print('Train complete:', best_acc, best_mcc)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved at {model_save_path}')
    return best_acc, best_mcc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../dataset/dataset.pkl')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead_time', type=int, default=1)
    parser.add_argument('--nhead_stock', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--model_save_path', type=str, default='../model/craft.pth')
    parser.add_argument('--model_save_dir', type=str, default='model')
    parser.add_argument('--stock_mode', type=str, default='last')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_cw_token', type=bool, default=False)
    parser.add_argument('--use_cw_stock', type=bool, default=False)
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
    )  = load_data(args.dataset)
    alpha = args.alpha
    epochs = args.epochs
    d_model = args.d_model
    nhead_time = args.nhead_time
    nhead_stock = args.nhead_stock
    dim_feedforward = args.dim_feedforward
    learning_rate = args.learning_rate
    seed = args.seed
    debug = args.debug
    model_save_path = args.model_save_path
    model_save_dir = args.model_save_dir
    stock_mode = args.stock_mode
    device = args.device
    use_cw_token = args.use_cw_token
    use_cw_stock = args.use_cw_stock

    train_model(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        token_embeddings_tensor=token_embeddings_tensor,
        alpha=alpha,
        epochs=epochs,
        d_model=d_model,
        nhead_time=nhead_time,
        nhead_stock=nhead_stock,
        dim_feedforward=dim_feedforward,
        learning_rate=learning_rate,
        use_cw_token=use_cw_token,
        use_cw_stock=use_cw_stock,
        time_class_weights=time_class_weights,
        stock_class_weights=stock_class_weights,
        seed=seed,
        do_debug_print=debug,
        model_save_path=model_save_path,
        model_save_dir=model_save_dir,
        stock_mode=stock_mode,
        device=device
    )

if __name__ == '__main__':
    main()