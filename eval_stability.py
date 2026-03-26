import argparse
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class DrugEncoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, hidden_dim: int = 128):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)


def canonicalize_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=1)


def pair_keys_from_pairs(pairs: np.ndarray, num_nodes: int) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0,), dtype=np.int64)
    return pairs[:, 0].astype(np.int64) * num_nodes + pairs[:, 1].astype(np.int64)


def pair_keys_from_df(df: pd.DataFrame, num_nodes: int) -> np.ndarray:
    pairs = canonicalize_pairs(df["drug1_id"].to_numpy(np.int64), df["drug2_id"].to_numpy(np.int64))
    return pair_keys_from_pairs(pairs, num_nodes=num_nodes)


def load_pairs_csv(path: str) -> np.ndarray:
    pair_df = pd.read_csv(path)
    for col in ["drug1_id", "drug2_id"]:
        if col not in pair_df.columns:
            raise ValueError(f"Missing column '{col}' in pair file: {path}")
    pairs = canonicalize_pairs(
        pair_df["drug1_id"].to_numpy(np.int64),
        pair_df["drug2_id"].to_numpy(np.int64),
    )
    return np.unique(pairs, axis=0)


def filter_df_by_pair_keys(df: pd.DataFrame, all_keys: np.ndarray, selected_keys: np.ndarray) -> pd.DataFrame:
    mask = np.isin(all_keys, selected_keys)
    return df.loc[mask].copy()


def maybe_auto_split_path(path_arg: Optional[str], default_name: str) -> Optional[str]:
    if path_arg:
        return path_arg
    auto_path = os.path.join("splits", default_name)
    if os.path.exists(auto_path):
        return auto_path
    return None


def build_base_edge_index(df: pd.DataFrame, num_nodes: int, device: torch.device) -> torch.Tensor:
    a = df["drug1_id"].to_numpy(np.int64)
    b = df["drug2_id"].to_numpy(np.int64)
    pairs = np.unique(canonicalize_pairs(a, b), axis=0)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]], axis=0)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]], axis=0)
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)


def build_pair_gt_mask(df: pd.DataFrame, num_nodes: int, num_relations: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(df) == 0:
        return np.empty((0,), dtype=np.int64), np.zeros((0, num_relations), dtype=np.bool_)

    a = df["drug1_id"].to_numpy(np.int64)
    b = df["drug2_id"].to_numpy(np.int64)
    r = df["se_id"].to_numpy(np.int64)

    pair_key = pair_keys_from_pairs(canonicalize_pairs(a, b), num_nodes=num_nodes)
    uniq_keys, inv = np.unique(pair_key, return_inverse=True)
    gt_mask = np.zeros((len(uniq_keys), num_relations), dtype=np.bool_)
    gt_mask[inv, r] = True
    return uniq_keys, gt_mask


def lookup_gt_rows(
    pair_key: np.ndarray,
    uniq_keys: np.ndarray,
    gt_mask: np.ndarray,
    num_relations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(pair_key), num_relations), dtype=np.bool_)
    matched = np.zeros((len(pair_key),), dtype=np.bool_)
    if len(pair_key) == 0 or len(uniq_keys) == 0:
        return out, matched

    idx = np.searchsorted(uniq_keys, pair_key)
    in_range = idx < len(uniq_keys)
    safe = np.zeros_like(idx)
    safe[in_range] = idx[in_range]
    matched[in_range] = uniq_keys[safe[in_range]] == pair_key[in_range]
    if matched.any():
        out[matched] = gt_mask[safe[matched]]
    return out, matched


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs_decagon_full/final_model.pt")
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--train-pairs", default=None, help="CSV train pairs for clean encoder graph")
    ap.add_argument("--pairs", default="splits/val_pairs.csv", help="Pair split to evaluate")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--batch-pairs", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Device:", device)

    df_full = pd.read_parquet(args.parquet)
    num_nodes_df = int(max(df_full["drug1_id"].max(), df_full["drug2_id"].max()) + 1)
    num_relations_df = int(df_full["se_id"].max() + 1)
    full_pair_keys = pair_keys_from_df(df_full, num_nodes=num_nodes_df)

    eval_pairs = load_pairs_csv(args.pairs)
    eval_pair_keys = pair_keys_from_pairs(eval_pairs, num_nodes=num_nodes_df)
    df_eval = filter_df_by_pair_keys(df_full, full_pair_keys, eval_pair_keys)
    if len(df_eval) == 0:
        raise ValueError(f"Eval subset is empty for pair file: {args.pairs}")

    train_pairs_path = maybe_auto_split_path(args.train_pairs, default_name="train_pairs.csv")
    if train_pairs_path is None:
        warnings.warn(
            "No --train-pairs provided and splits/train_pairs.csv not found. "
            "Falling back to FULL graph for encoding (may leak eval structure)."
        )
        df_train_graph = df_full
    else:
        train_pairs = load_pairs_csv(train_pairs_path)
        train_pair_keys = pair_keys_from_pairs(train_pairs, num_nodes=num_nodes_df)
        df_train_graph = filter_df_by_pair_keys(df_full, full_pair_keys, train_pair_keys)
        if len(df_train_graph) == 0:
            raise ValueError(f"Train graph subset is empty for train pair file: {train_pairs_path}")

    ckpt = torch.load(args.model, map_location=device)
    num_nodes = int(ckpt["num_nodes"])
    num_relations = int(ckpt["num_relations"])
    hidden_dim = int(ckpt["hidden_dim"])

    if num_nodes != num_nodes_df:
        raise ValueError(f"Model num_nodes={num_nodes} != parquet num_nodes={num_nodes_df}")
    if num_relations != num_relations_df:
        raise ValueError(f"Model num_relations={num_relations} != parquet num_relations={num_relations_df}")

    encoder = DrugEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
    decoder = DistMultDecoder(num_relations=num_relations, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()

    edge_index_base = build_base_edge_index(df_train_graph, num_nodes=num_nodes, device=device)

    print("⏳ Building GT mask for eval split...")
    uniq_keys_eval, gt_mask_eval = build_pair_gt_mask(df_eval, num_nodes=num_nodes, num_relations=num_relations)
    print(f"✅ Eval GT mask: pairs={len(uniq_keys_eval):,}, relations={num_relations}")
    print(
        f"Split stats | train graph pairs={len(np.unique(pair_keys_from_df(df_train_graph, num_nodes))):,} "
        f"| eval pairs={len(eval_pair_keys):,} | eval triples={len(df_eval):,}"
    )

    z = encoder(edge_index_base)  # [N,D]
    rel = decoder.rel_emb.weight  # [R,D]

    k = min(args.topk, num_relations)
    batch_pairs = args.batch_pairs
    n = len(eval_pairs)
    eval_a = eval_pairs[:, 0].astype(np.int64)
    eval_b = eval_pairs[:, 1].astype(np.int64)

    hit_list = []
    recall_list = []
    overlap_list = []
    found_list = []
    topk_span_list = []
    top1_logit_list = []
    missing_total = 0

    for i in range(0, n, batch_pairs):
        a_np = eval_a[i : i + batch_pairs]
        b_np = eval_b[i : i + batch_pairs]

        aa = torch.tensor(a_np, device=device, dtype=torch.long)
        bb = torch.tensor(b_np, device=device, dtype=torch.long)

        pair_vec = z[aa] * z[bb]  # [b,D]
        scores = pair_vec @ rel.t()  # [b,R] logits

        vals, idxs = torch.topk(scores, k=k, dim=1)
        idxs_np = idxs.cpu().numpy()

        pair_key = pair_keys_from_pairs(canonicalize_pairs(a_np, b_np), num_nodes=num_nodes)
        gt_rows, matched = lookup_gt_rows(pair_key, uniq_keys_eval, gt_mask_eval, num_relations=num_relations)
        missing_total += int((~matched).sum())

        overlap = gt_rows[np.arange(gt_rows.shape[0])[:, None], idxs_np].sum(axis=1)
        found = gt_rows.sum(axis=1)
        hit = (overlap > 0).astype(np.float32)
        denom = np.minimum(found, k)
        denom = np.where(denom == 0, 1, denom)
        recall = overlap / denom

        topk_span = (vals[:, 0] - vals[:, -1]).detach().cpu().numpy()
        top1_logit = vals[:, 0].detach().cpu().numpy()

        hit_list.append(hit)
        recall_list.append(recall.astype(np.float32))
        overlap_list.append(overlap.astype(np.int32))
        found_list.append(found.astype(np.int32))
        topk_span_list.append(topk_span.astype(np.float32))
        top1_logit_list.append(top1_logit.astype(np.float32))

    hit = np.concatenate(hit_list)
    recall = np.concatenate(recall_list)
    overlap = np.concatenate(overlap_list)
    found = np.concatenate(found_list)
    topk_span = np.concatenate(topk_span_list)
    top1_logit = np.concatenate(top1_logit_list)

    print("\n=== Evaluation metrics (clean split) ===")
    print(f"Pairs: {n:,} | K={k}")
    print(f"Missing GT pairs in eval subset: {missing_total:,}")
    print(f"Hit@K:        {hit.mean():.4f} ± {hit.std():.4f}")
    print(f"Recall@K:     {recall.mean():.4f} ± {recall.std():.4f}")
    print(f"Overlap@K:    {overlap.mean():.2f} ± {overlap.std():.2f}")
    print(f"Found (GT):   {found.mean():.2f} ± {found.std():.2f}")

    print("\n=== Stability-ish stats (logit space) ===")
    print(
        f"Top1 logit:   mean={top1_logit.mean():.4f} std={top1_logit.std():.4f} "
        f"min={top1_logit.min():.4f} max={top1_logit.max():.4f}"
    )
    print(
        f"TopK span:    mean={topk_span.mean():.6f} std={topk_span.std():.6f} "
        f"min={topk_span.min():.6f} max={topk_span.max():.6f}"
    )


if __name__ == "__main__":
    main()
