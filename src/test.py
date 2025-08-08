import argparse
from model import CraftModel, Trainer
from data import load_data
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/dataset.pkl')
    parser.add_argument('--model_path', type=str, default='model/model.pth')
    parser.add_argument('--result_save_path', type=str, default='result.txt')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead_time', type=int, default=1)
    parser.add_argument('--nhead_stock', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--stock_mode', type=str, default='last')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    model_path = args.model_path
    d_model = args.d_model
    nhead_time = args.nhead_time
    nhead_stock = args.nhead_stock
    dim_feedforward = args.dim_feedforward
    stock_mode = args.stock_mode
    device = args.device
    result_save_path = args.result_save_path

    train_loader, valid_loader, test_loader, token_embeddings_tensor, input_dim, class_weights_updown = load_data(dataset)
    model = CraftModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead_time=nhead_time,
        nhead_stock=nhead_stock,
        dim_feedforward=dim_feedforward,
        stock_mode=stock_mode
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    trainer = Trainer(model, None, device)
    acc, mcc = trainer.evaluate(test_loader)
    print(f'Test ACC: {acc:.4f}, MCC: {mcc:.4f}')
    with open(result_save_path) as file:
        file.write(f"{acc},{mcc}")

if __name__ == '__main__':
    main()