from datetime import datetime
from utils import generate_causal_mask, print_debug_info_for_epoch
import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialAttentionBlock(nn.Module):
    """
    Time-axis: TransformerDecoderLayer(d_model, ...)
    Stock-axis:
      - if stock_mode='last': use out_time[:, -1, :]
      - if stock_mode='flatten': flatten => [n,l*d_model] => proj => [n,d_model]
    """
    def __init__(self, d_model=32, nhead_time=1, nhead_stock=4,
                 dim_feedforward=128,
                 stock_mode='last',
                 seq_len=20  
        ):
        super().__init__()
        self.d_model = d_model
        self.nhead_time = nhead_time
        self.nhead_stock= nhead_stock
        self.dim_feedforward= dim_feedforward
        self.stock_mode= stock_mode
        self.seq_len= seq_len  

        # (A) Time-axis (d_model=32 등)
        self.time_attn = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead_time,
            dim_feedforward=dim_feedforward,
            batch_first=False
        )

        # (B) Stock-axis (same d_model=32)
        self.stock_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_stock,
            dim_feedforward=dim_feedforward,
            batch_first=False
        )

        # (C) flatten_proj => "l*d_model -> d_model"
        #   only used if stock_mode='flatten'
        self.flatten_proj = nn.Linear(self.seq_len * d_model, d_model)

    def forward(self, x):
        """
        x: [n,l,d_model]
        return shape =>
          'last': => [n,1,d_model]
          'flatten': => [n,1,d_model]
        """
        device = x.device
        n, l, d = x.shape
        assert d == self.d_model, f"Input last dim={d}, but d_model={self.d_model} mismatch."

        # --- 1) Time Axis ---
        x_time = x.permute(1,0,2)  # => [l,n,d_model]
        mask_time = generate_causal_mask(l, device=device)
        out_time = self.time_attn(tgt=x_time, memory=x_time, tgt_mask=mask_time)
        # => [l,n,d_model]
        out_time = out_time.permute(1,0,2)  # => [n,l,d_model]

        # --- 2) Stock Axis ---
        if self.stock_mode=='last':
            # last => [n,d_model], unsqueeze => [n,1,d_model]
            x_last = out_time[:, -1, :]  # => [n,d_model]
            x_last = x_last.unsqueeze(1) # => [n,1,d_model]
            out_stock= self.stock_attn(x_last) # => [n,1,d_model]
            return out_stock

        elif self.stock_mode=='flatten':
            # flatten => [n,l*d_model]
            x_flat = out_time.reshape(n, l*d)
            # (B) projection => => [n,d_model]
            x_flat_proj = self.flatten_proj(x_flat)
            # => unsqueeze => [n,1,d_model]
            x_flat_proj = x_flat_proj.unsqueeze(1)
            out_stock= self.stock_attn(x_flat_proj) # => [n,1,d_model]
            return out_stock

        else:
            raise ValueError("stock_mode must be 'last' or 'flatten'")



class CraftModel(nn.Module):
    """
    Time Axis => d_model => TransformerDecoderLayer
    Stock Axis => same d_model
      stock_mode='last': out_time[:, -1, :]
      stock_mode='flatten': flatten => Linear => shape [n,d_model]
    """
    def __init__(self,
                 input_dim,
                 d_model=32,
                 nhead_time=1,
                 nhead_stock=4,
                 dim_feedforward=128,
                 vocab_size=100,
                 seq_len=20,
                 stock_mode='last'):
        super().__init__()
        self.input_proj= nn.Linear(input_dim, d_model)  # => [n,l,d_model]
        self.time_attn= nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead_time,
            dim_feedforward=dim_feedforward,
            batch_first=False
        )
        self.time_fc= nn.Linear(d_model, vocab_size)  # next-token

        self.stock_attn= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_stock,
            dim_feedforward=dim_feedforward,
            batch_first=False
        )
        self.updown_fc= nn.Linear(d_model, 2)

        self.stock_mode= stock_mode
        self.seq_len= seq_len
        self.d_model= d_model

        # flatten_proj => (l*d_model -> d_model)
        if stock_mode=='flatten':
            self.flatten_proj= nn.Linear(seq_len*d_model, d_model)

    def forward(self, x):
        """
        x: [n,l,input_dim]
        returns nxt_logits => [n,l,vocab], up_logits => [n,2]
        """
        device= x.device
        n,l,in_dim= x.shape

        # (A) Time Axis
        x_proj= self.input_proj(x)  # => [n,l,d_model]
        x_time= x_proj.permute(1,0,2) # => [l,n,d_model]
        mask_time= generate_causal_mask(l, device=device)
        out_time= self.time_attn(tgt=x_time, memory=x_time, tgt_mask=mask_time)
        # => [l,n,d_model]
        out_time= out_time.permute(1,0,2) # => [n,l,d_model]
        # next-token => [n,l,vocab]
        nxt_logits= self.time_fc(out_time)

        # (B) Stock Axis
        if self.stock_mode=='last':
            x_last= out_time[:, -1, :]  # => [n,d_model]
            x_last= x_last.unsqueeze(1) # => [n,1,d_model], seq=n
            out_stock= self.stock_attn(x_last)  # => [n,1,d_model]
            up_logits= self.updown_fc(out_stock[:,0,:])
            return nxt_logits, up_logits

        elif self.stock_mode=='flatten':
            # flatten => shape [n, l*d_model]
            x_flat= out_time.reshape(n, l*self.d_model)
            # project => [n,d_model]
            x_proj_flat= self.flatten_proj(x_flat)
            # => [n,1,d_model], seq=n
            x_proj_flat= x_proj_flat.unsqueeze(1)
            out_stock= self.stock_attn(x_proj_flat)  # => [n,1,d_model]
            up_logits= self.updown_fc(out_stock[:,0,:])
            return nxt_logits, up_logits

        else:
            raise ValueError("stock_mode must be 'last' or 'flatten'")

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model= model.to(device)
        self.optimizer= optimizer
        self.device= device

    def train(self,
              loader,
              token_embeddings_tensor,
              alpha=0.5,
              use_cw_token=False,
              use_cw_stock=False,
              time_class_weights=None,   # [vocab_size] for token
              stock_class_weights=None
              ):
        self.model.train()
        total_train_loss = 0.0

        for (inputs, targets, updowns, date_list) in loader:
            inputs = inputs.to(self.device)  # [bs, n, l, d]
            targets = targets.to(self.device)  # [bs, n, l]
            updowns = updowns.to(self.device)  # [bs, n]
            self.optimizer.zero_grad()
            batch_loss_val = 0.0
            bs = inputs.size(0)
            for i in range(bs):
                x_i = inputs[i]  # [n,l,d]
                t_i = targets[i]  # [n,l]
                u_i = updowns[i]  # [n]
                nxt_logits, up_logits = self.model(x_i)
                # Time Axis => CE + Soft Embedding MAE
                # (A) Flatten for CE
                if nxt_logits.dim() == 3:
                    # => shape [n,l,vocab]
                    n_, l_, v_ = nxt_logits.shape
                    nt_logits_2d = nxt_logits.view(n_ * l_, v_)
                    t_1d = t_i.view(-1)
                else:
                    # [n,vocab] or [n,1,vocab]
                    n_, _, v_ = nxt_logits.shape if nxt_logits.dim() == 3 else (*nxt_logits.shape, 1)
                    if nxt_logits.dim() == 2:
                        nt_logits_2d = nxt_logits
                        t_1d = t_i[:, -1]
                    else:
                        nt_logits_2d = nxt_logits.view(n_ * 1, v_)
                        t_1d = t_i[:, -1]
                # (A-i) CE
                if use_cw_token and (time_class_weights is not None):
                    loss_time_ce = F.cross_entropy(nt_logits_2d, t_1d, weight=time_class_weights)
                else:
                    loss_time_ce = F.cross_entropy(nt_logits_2d, t_1d)
                # (A-ii) Embedding Loss with soft embedding
                # softmax => shape [N, vocab]
                p = F.softmax(nt_logits_2d, dim=-1)  # [N, vocab]
                # => pred_emb => [N, emb_dim], by matrix multiply p( [N,vocab] ) * token_embeddings_tensor( [vocab, emb_dim] )
                pred_emb = p @ token_embeddings_tensor  # shape => [N, emb_dim]
                # true_emb => [N, emb_dim], with t_1d => shape [N], index
                true_emb = token_embeddings_tensor[t_1d]
                # mae or mse
                loss_time_mae = F.l1_loss(pred_emb, true_emb)
                loss_time = loss_time_ce
                # (B) Stock Axis => updown
                if use_cw_stock and (stock_class_weights is not None):
                    loss_stock = F.cross_entropy(up_logits, u_i, weight=stock_class_weights)
                else:
                    loss_stock = F.cross_entropy(up_logits, u_i)
                # total => alpha*(time) + (1-alpha)*(stock)
                loss_ = alpha * loss_time + (1 - alpha) * loss_stock
                batch_loss_val += loss_
            batch_loss_val.backward()
            self.optimizer.step()
            total_train_loss += batch_loss_val.item()
        avg_train_loss = total_train_loss / len(loader)
        return avg_train_loss

    def fit(self,
            train_loader,
            valid_loader,
            test_loader,
            token_embeddings_tensor,
            alpha=0.5,
            beta=0.5,
            epochs=30,
            use_cw_token=False,
            use_cw_stock=False,
            time_class_weights=None,  # [vocab_size] for token
            stock_class_weights=None,
            seed = 0,
            do_debug_print = True,
            model_save_dir = "model"
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        best_val_loss = float("inf")
        best_model_path = None

        for ep in range(epochs):
            avg_train_loss = self.train(
                loader=train_loader,
                token_embeddings_tensor=token_embeddings_tensor,
                alpha=alpha,
                use_cw_token=use_cw_token,
                use_cw_stock=use_cw_stock,
                time_class_weights=time_class_weights,
                stock_class_weights=stock_class_weights
            )
            # (Option) Debug
            if do_debug_print:
                print_debug_info_for_epoch(self.model, train_loader, self.device, ep)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for (val_inp, val_tgt, val_up, val_dt) in valid_loader:
                    val_inp = val_inp.to(self.device)
                    val_tgt = val_tgt.to(self.device)
                    val_up = val_up.to(self.device)
                    bs_v = val_inp.size(0)
                    batch_val_loss = 0.0
                    for b_i in range(bs_v):
                        x_i = val_inp[b_i]
                        t_i = val_tgt[b_i]
                        u_i = val_up[b_i]
                        nxt_logits, up_logits = self.model(x_i)
                        # Flatten => CE
                        if nxt_logits.dim() == 3:
                            n_, l_, v_ = nxt_logits.shape
                            nt_2d = nxt_logits.view(n_ * l_, v_)
                            t_1d = t_i.view(-1)
                        else:
                            n_, _, v_ = nxt_logits.shape if nxt_logits.dim() == 3 else (*nxt_logits.shape, 1)
                            if nxt_logits.dim() == 2:
                                nt_2d = nxt_logits
                                t_1d = t_i[:, -1]
                            else:
                                nt_2d = nxt_logits.view(n_ * 1, v_)
                                t_1d = t_i[:, -1]
                        # CE
                        if use_cw_token and (time_class_weights is not None):
                            l_time_ce = F.cross_entropy(nt_2d, t_1d, weight=time_class_weights)
                        else:
                            l_time_ce = F.cross_entropy(nt_2d, t_1d)
                        # soft embedding
                        p_val = F.softmax(nt_2d, dim=-1)  # [N,vocab]
                        pred_emb_val = p_val @ token_embeddings_tensor
                        true_emb_val = token_embeddings_tensor[t_1d]
                        l_time_mae = F.l1_loss(pred_emb_val, true_emb_val)
                        l_time = beta * l_time_ce
                        # stock
                        if use_cw_stock and (stock_class_weights is not None):
                            l_stock = F.cross_entropy(up_logits, u_i, weight=stock_class_weights)
                        else:
                            l_stock = F.cross_entropy(up_logits, u_i)
                        loss_val = alpha * l_time + (1 - alpha) * l_stock
                        batch_val_loss += loss_val
                    total_val_loss += batch_val_loss.item()
            avg_val_loss = total_val_loss / len(valid_loader)
            test_acc_each, test_mcc_each = self.evaluate(test_loader)
            print(f"[Epoch {ep + 1}/{epochs}] train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, test_acc={test_acc_each:.4f}, test_mcc={test_mcc_each:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                best_model_path = os.path.join(model_save_dir, f"two_stage_{seed}_best.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  best model saved (val_loss={best_val_loss:.4f})")

        print(f"Training done. best val loss={best_val_loss:.4f}, path={best_model_path}")

        best_acc, best_mcc = 0.0, 0.0
        if best_model_path:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            best_acc, best_mcc = self.evaluate(test_loader)
            print(f"Test ACC={best_acc:.4f}, MCC={best_mcc:.4f}")
        return best_acc, best_mcc

    def evaluate(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for (inputs, targets, updowns, date_list) in loader:
                inputs = inputs.to(self.device)
                updowns = updowns.to(self.device)
                bs = inputs.size(0)
                for i in range(bs):
                    x_i = inputs[i]
                    u_i = updowns[i]
                    _, up_logits = self.model(x_i)
                    pred_ = up_logits.argmax(dim=-1)
                    y_true.append(u_i.cpu().numpy())
                    y_pred.append(pred_.cpu().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        acc = (y_true == y_pred).mean()
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
        if denom == 0:
            mcc = 0.0
        else:
            mcc = (tp * tn - fp * fn) / math.sqrt(denom)
        return acc, mcc

    def save(self, path):
        torch.save(self.model.state_dict(), path)

