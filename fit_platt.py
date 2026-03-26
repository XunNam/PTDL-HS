import argparse
import json
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
    pairs = np.unique(
        canonicalize_pairs(df["drug1_id"].to_numpy(np.int64), df["drug2_id"].to_numpy(np.int64)),
        axis=0,
    )
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


def lookup_gt_row(pair_key: int, uniq_keys: np.ndarray, gt_mask: np.ndarray, num_relations: int) -> Optional[np.ndarray]:
    if len(uniq_keys) == 0:
        return None
    idx = int(np.searchsorted(uniq_keys, pair_key))
    if idx >= len(uniq_keys) or int(uniq_keys[idx]) != int(pair_key):
        return None
    return gt_mask[idx]


@torch.no_grad()
def score_triplets(z: torch.Tensor, relW: torch.Tensor, a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (z[a] * relW[r] * z[b]).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--train-pairs", default=None, help="CSV train pairs for clean encoder graph")
    ap.add_argument("--val-pairs", default="splits/val_pairs.csv", help="(Legacy) calibration pair split")
    ap.add_argument("--pairs", default=None, help="Calibration pair split (overrides --val-pairs)")
    ap.add_argument("--neg-per-pos", type=int, default=3, help="Negatives per positive")
    ap.add_argument("--max-pos-per-pair", type=int, default=50, help="Cap positives per pair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="calib/platt.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    df_full = pd.read_parquet(args.parquet)
    num_nodes = int(max(df_full["drug1_id"].max(), df_full["drug2_id"].max()) + 1)
    num_relations = int(df_full["se_id"].max() + 1)
    full_pair_keys = pair_keys_from_df(df_full, num_nodes=num_nodes)

    train_pairs_path = maybe_auto_split_path(args.train_pairs, default_name="train_pairs.csv")
    if train_pairs_path is None:
        warnings.warn(
            "No --train-pairs provided and splits/train_pairs.csv not found. "
            "Falling back to FULL graph for encoding (can leak calibration)."
        )
        df_train_graph = df_full
    else:
        train_pairs = load_pairs_csv(train_pairs_path)
        train_keys = pair_keys_from_pairs(train_pairs, num_nodes=num_nodes)
        df_train_graph = filter_df_by_pair_keys(df_full, full_pair_keys, train_keys)
        if len(df_train_graph) == 0:
            raise ValueError(f"Train graph subset is empty for train pair file: {train_pairs_path}")
        print(f"✅ Using train graph from: {train_pairs_path}")

    calib_pairs_path = args.pairs if args.pairs is not None else args.val_pairs
    if calib_pairs_path is None:
        raise ValueError("Calibration pair file is required (--pairs or --val-pairs).")
    if not os.path.exists(calib_pairs_path):
        raise FileNotFoundError(f"Calibration pair file not found: {calib_pairs_path}")
    calib_pairs = load_pairs_csv(calib_pairs_path)
    calib_keys = pair_keys_from_pairs(calib_pairs, num_nodes=num_nodes)
    df_calib = filter_df_by_pair_keys(df_full, full_pair_keys, calib_keys)
    if len(df_calib) == 0:
        raise ValueError(f"Calibration subset is empty for pair file: {calib_pairs_path}")

    print("⏳ Building calibration GT mask (split-only)...")
    uniq_keys_calib, gt_mask_calib = build_pair_gt_mask(df_calib, num_nodes=num_nodes, num_relations=num_relations)
    print(f"✅ Calibration GT mask: pairs={len(uniq_keys_calib):,}, relations={num_relations}")
    print(
        f"Split stats | train graph pairs={len(np.unique(pair_keys_from_df(df_train_graph, num_nodes))):,} "
        f"| calib pairs={len(calib_keys):,} | calib triples={len(df_calib):,}"
    )

    print("⏳ Build train graph...")
    edge_index = build_base_edge_index(df_train_graph, num_nodes=num_nodes, device=device)

    print("⏳ Load model...")
    ckpt = torch.load(args.model, map_location=device)
    hidden_dim = int(ckpt["hidden_dim"])
    if int(ckpt["num_nodes"]) != num_nodes or int(ckpt["num_relations"]) != num_relations:
        raise ValueError(
            f"Model/data shape mismatch: model=({ckpt['num_nodes']},{ckpt['num_relations']}) "
            f"vs data=({num_nodes},{num_relations})"
        )

    enc = DrugEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
    dec = DistMultDecoder(num_relations=num_relations, hidden_dim=hidden_dim).to(device)
    enc.load_state_dict(ckpt["encoder_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    enc.eval()
    dec.eval()

    print("⏳ Encode once...")
    with torch.no_grad():
        z = enc(edge_index)  # [N,D]
        relW = dec.rel_emb.weight  # [R,D]

    logits_list = []
    labels_list = []
    skipped_pairs = 0

    print("⏳ Collect calibration samples...")
    for (a, b) in calib_pairs:
        pair_key = int(min(a, b) * num_nodes + max(a, b))
        gt_row = lookup_gt_row(pair_key, uniq_keys_calib, gt_mask_calib, num_relations=num_relations)
        if gt_row is None:
            skipped_pairs += 1
            continue

        pos = np.where(gt_row)[0]
        if len(pos) == 0:
            continue
        if len(pos) > args.max_pos_per_pair:
            pos = rng.choice(pos, size=args.max_pos_per_pair, replace=False)

        neg_pool = np.where(~gt_row)[0]
        m = min(len(neg_pool), len(pos) * args.neg_per_pos)
        if m == 0:
            continue
        neg = rng.choice(neg_pool, size=m, replace=False)

        a_pos = torch.full((len(pos),), int(a), device=device, dtype=torch.long)
        b_pos = torch.full((len(pos),), int(b), device=device, dtype=torch.long)
        r_pos = torch.tensor(pos, device=device, dtype=torch.long)
        lp = score_triplets(z, relW, a_pos, b_pos, r_pos).detach().cpu()

        a_neg = torch.full((len(neg),), int(a), device=device, dtype=torch.long)
        b_neg = torch.full((len(neg),), int(b), device=device, dtype=torch.long)
        r_neg = torch.tensor(neg, device=device, dtype=torch.long)
        ln = score_triplets(z, relW, a_neg, b_neg, r_neg).detach().cpu()

        logits_list.append(lp)
        labels_list.append(torch.ones_like(lp))
        logits_list.append(ln)
        labels_list.append(torch.zeros_like(ln))

    if not logits_list:
        raise ValueError("No calibration samples collected. Check split files and dataset consistency.")

    logits = torch.cat(logits_list).to(device=device, dtype=torch.float32)
    labels = torch.cat(labels_list).to(device=device, dtype=torch.float32)
    print(f"✅ Calibration set: N={logits.numel():,} (pos={int(labels.sum().item()):,}) | skipped_pairs={skipped_pairs:,}")

    a_platt = torch.tensor(1.0, device=device, requires_grad=True)
    b_platt = torch.tensor(0.0, device=device, requires_grad=True)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([a_platt, b_platt], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = bce(a_platt * logits + b_platt, labels)
        loss.backward()
        return loss

    loss = float(opt.step(closure).item())
    a_val = float(a_platt.detach().cpu().item())
    b_val = float(b_platt.detach().cpu().item())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"a": a_val, "b": b_val, "loss": loss}, f, indent=2)

    print(f"✅ Fitted Platt: a={a_val:.6f}, b={b_val:.6f} | loss={loss:.6f}")
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
