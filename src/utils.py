import matplotlib.pyplot as plt
import torch
import random

def visualize_embeddings_for_date(embedding_df, target_date):
    date_embeddings = embedding_df[embedding_df['date'] == target_date]
    plt.figure(figsize=(10, 8))
    plt.scatter(date_embeddings['embedding_x'], date_embeddings['embedding_y'])

    for _, row in date_embeddings.iterrows():
        plt.text(row['embedding_x'], row['embedding_y'], row['symbol'], fontsize=9)

    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title(f"Stock Embeddings for {target_date.date()}")
    plt.show()

def generate_causal_mask(seq_len, device='cpu'):
    """
    causal (upper-triangular) mask: shape [seq_len, seq_len]
    True = masked
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


def print_debug_info_for_epoch(model, train_loader, device, epoch):
    """

    - random batch => random symbol => print entire sequence of tokens
    - plus up/down
    """
    model.eval()

    all_batches = list(train_loader)
    if len(all_batches) == 0:
        return
    random_batch = random.choice(all_batches)
    (batch_inputs, batch_targets, batch_updowns, date_list) = random_batch

    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    batch_updowns = batch_updowns.to(device)

    bs = batch_inputs.size(0)
    bidx = random.randint(0, bs - 1)
    x_i = batch_inputs[bidx]
    t_i = batch_targets[bidx]
    u_i = batch_updowns[bidx]

    with torch.no_grad():
        nxt_logits, up_logits = model(x_i)

    print(f"\n[Epoch {epoch + 1}] Debug => random batch={bidx}")
    if nxt_logits.dim() == 3:
        # shape => [n,l,vocab]
        n_, l_, v_ = nxt_logits.shape
        sym_idx = random.randint(0, n_ - 1)
        # print full sequence
        pred_list = []
        true_list = []
        for tidx in range(l_):
            pred_tok = nxt_logits[sym_idx, tidx].argmax(dim=-1).item()
            true_tok = t_i[sym_idx, tidx].item()
            pred_list.append(str(pred_tok))
            true_list.append(str(true_tok))
        print(f"(TimeAxis) symbol={sym_idx}")
        print(f"  Pred tokens: {' '.join(pred_list)}")
        print(f"  True tokens: {' '.join(true_list)}")

        # up/down => shape => [n,2]
        pred_ud = up_logits[sym_idx].argmax(dim=-1).item()
        true_ud = u_i[sym_idx].item()
        print(f"(StockAxis) symbol={sym_idx}, Pred UpDown={pred_ud}, True UpDown={true_ud}")
    else:
        # flatten or last => shape => [n,1,vocab] or [n,vocab]
        print("  [warn] not shape [n,l,vocab]. partial info.")
