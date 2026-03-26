import argparse
import datetime
import json
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# -------------------------
# Models
# -------------------------
class DrugEncoder(nn.Module):
    """GNN encoder over a base (drug-drug) graph."""

    def __init__(self, num_nodes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight  # [num_nodes, hidden_dim]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings


class DistMultDecoder(nn.Module):
    """Multi-relation decoder: score(drug1, se_id, drug2)."""

    def __init__(self, num_relations: int, hidden_dim: int = 128):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rh = self.rel_emb(r)
        return (z[h] * rh * z[t]).sum(dim=-1)  # [batch]


# -------------------------
# Pair utils
# -------------------------
def canonicalize_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=1)


def pair_keys_from_pair_array(pairs: np.ndarray, num_nodes: int) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0,), dtype=np.int64)
    return pairs[:, 0].astype(np.int64) * num_nodes + pairs[:, 1].astype(np.int64)


def pair_keys_from_df(df: pd.DataFrame, num_nodes: int) -> np.ndarray:
    pairs = canonicalize_pairs(
        df["drug1_id"].to_numpy(np.int64),
        df["drug2_id"].to_numpy(np.int64),
    )
    return pair_keys_from_pair_array(pairs, num_nodes=num_nodes)


def unique_pairs_from_df(df: pd.DataFrame) -> np.ndarray:
    pairs = canonicalize_pairs(
        df["drug1_id"].to_numpy(np.int64),
        df["drug2_id"].to_numpy(np.int64),
    )
    return np.unique(pairs, axis=0)


def load_pairs_csv(path: str) -> np.ndarray:
    pair_df = pd.read_csv(path)
    for col in ["drug1_id", "drug2_id"]:
        if col not in pair_df.columns:
            raise ValueError(f"Missing column '{col}' in pair file: {path}")
    pairs = canonicalize_pairs(
        pair_df["drug1_id"].to_numpy(np.int64),
        pair_df["drug2_id"].to_numpy(np.int64),
    )
    pairs = np.unique(pairs, axis=0)
    return pairs


def filter_df_by_pair_keys(df: pd.DataFrame, all_pair_keys: np.ndarray, selected_pair_keys: np.ndarray) -> pd.DataFrame:
    mask = np.isin(all_pair_keys, selected_pair_keys)
    return df.loc[mask].copy()


def sample_pairs_fixed(pairs: np.ndarray, sample_size: int, seed: int = 42) -> np.ndarray:
    """
    Sample a fixed subset of undirected pairs for train-side eval metrics.
    Using a fixed sample keeps train_eval curves stable across epochs.
    """
    if pairs.size == 0 or sample_size <= 0:
        return np.empty((0, 2), dtype=np.int64)
    if len(pairs) <= sample_size:
        return pairs.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=sample_size, replace=False)
    return pairs[idx]


def maybe_auto_split_path(path_arg: Optional[str], default_name: str) -> Optional[str]:
    if path_arg:
        return path_arg
    auto_path = os.path.join("splits", default_name)
    if os.path.exists(auto_path):
        return auto_path
    return None


def validate_pair_splits(
    train_keys: np.ndarray,
    val_keys: np.ndarray,
    test_keys: Optional[np.ndarray],
    all_keys: np.ndarray,
    strict_union: bool = False,
) -> None:
    if np.intersect1d(train_keys, val_keys).size > 0:
        raise ValueError("Detected overlap between train_pairs and val_pairs.")
    if test_keys is not None and np.intersect1d(train_keys, test_keys).size > 0:
        raise ValueError("Detected overlap between train_pairs and test_pairs.")
    if test_keys is not None and np.intersect1d(val_keys, test_keys).size > 0:
        raise ValueError("Detected overlap between val_pairs and test_pairs.")

    if strict_union and test_keys is not None:
        union = np.unique(np.concatenate([train_keys, val_keys, test_keys], axis=0))
        if union.size != all_keys.size:
            raise ValueError(
                f"Split union mismatch: union={union.size}, full_unique_pairs={all_keys.size}. "
                "Check train/val/test split files."
            )


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


# -------------------------
# Graph / schedule utils
# -------------------------
def build_base_edge_index(df: pd.DataFrame, num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Build an undirected base graph using UNIQUE drug pairs only (ignore se_id).
    """
    pairs = unique_pairs_from_df(df)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]], axis=0)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]], axis=0)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_nodes
    return edge_index


def linear_ramp(epoch: int, warmup: int, ramp: int, wmax: float) -> float:
    if epoch <= warmup:
        return 0.0
    t = min(1.0, (epoch - warmup) / max(1, ramp))
    return wmax * t


def hard_k_schedule(epoch: int, warmup: int, ramp: int, kmin: int, kmax: int) -> int:
    if epoch <= warmup:
        return kmin
    t = min(1.0, (epoch - warmup) / max(1, ramp))
    return int(round(kmin + (kmax - kmin) * t))


def build_pair_gt_mask(df: pd.DataFrame, num_nodes: int, num_relations: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      uniq_keys: np.ndarray shape [num_pairs] (sorted)
      gt_mask:   np.ndarray shape [num_pairs, R] bool
    pair_key = min(a,b) * num_nodes + max(a,b) for undirected pairs.
    """
    if len(df) == 0:
        return np.empty((0,), dtype=np.int64), np.zeros((0, num_relations), dtype=np.bool_)

    a = df["drug1_id"].to_numpy(np.int64)
    b = df["drug2_id"].to_numpy(np.int64)
    r = df["se_id"].to_numpy(np.int64)

    pair_key = pair_keys_from_pair_array(canonicalize_pairs(a, b), num_nodes=num_nodes)
    uniq_keys, inv = np.unique(pair_key, return_inverse=True)
    gt_mask = np.zeros((len(uniq_keys), num_relations), dtype=np.bool_)
    gt_mask[inv, r] = True
    return uniq_keys, gt_mask


@torch.no_grad()
def mine_hard_negatives(
    z_det: torch.Tensor,
    rel_det: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    uniq_keys: np.ndarray,
    gt_mask: np.ndarray,
    num_nodes: int,
    hard_k: int = 50,
    chunk_size: int = 4096,
    use_fp16: bool = True,
    hard_topm: int = 10,
) -> torch.Tensor:
    za = z_det[a]  # [B,D]
    zb = z_det[b]  # [B,D]
    prod = za * zb  # [B,D]
    relT = rel_det.t().contiguous()  # [D,R]

    batch = prod.size(0)
    k = min(hard_k, rel_det.size(0))

    cand_chunks = []
    for s in range(0, batch, chunk_size):
        e = min(s + chunk_size, batch)
        x = prod[s:e]  # [c,D]

        if use_fp16 and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                scores = x @ relT  # [c,R]
        else:
            scores = x @ relT

        _, cand = torch.topk(scores, k=k, dim=1)  # [c,k]
        cand_chunks.append(cand.cpu().numpy())

    cand_np = np.concatenate(cand_chunks, axis=0)  # [B,k]

    pair_key = pair_keys_from_pair_array(
        canonicalize_pairs(
            torch.minimum(a, b).cpu().numpy(),
            torch.maximum(a, b).cpu().numpy(),
        ),
        num_nodes=num_nodes,
    )
    gt_rows, _ = lookup_gt_rows(pair_key, uniq_keys, gt_mask, num_relations=rel_det.size(0))

    hard = np.empty((cand_np.shape[0],), dtype=np.int64)
    topm = min(hard_topm, cand_np.shape[1])
    rng = np.random.default_rng()

    for i in range(cand_np.shape[0]):
        row = gt_rows[i]
        pool = [int(r) for r in cand_np[i, :topm] if not row[r]]
        if pool:
            hard[i] = int(pool[rng.integers(len(pool))])
            continue

        picked = None
        for r in cand_np[i]:
            if not row[int(r)]:
                picked = int(r)
                break
        hard[i] = int(cand_np[i, -1]) if picked is None else picked

    return torch.tensor(hard, device=a.device, dtype=torch.long)


@torch.no_grad()
def eval_on_pairs(
    encoder: nn.Module,
    decoder: nn.Module,
    edge_index_base: torch.Tensor,
    val_pairs: np.ndarray,
    uniq_keys: np.ndarray,
    gt_mask: np.ndarray,
    num_nodes: int,
    topk: int,
    device: torch.device,
    batch_pairs: int = 1024,
) -> dict:
    encoder.eval()
    decoder.eval()
    bce = torch.nn.BCEWithLogitsLoss()
    val_loss_all = []

    z = encoder(edge_index_base)  # [N,D]
    rel = decoder.rel_emb.weight  # [R,D]
    num_relations = rel.size(0)
    k = min(topk, num_relations)

    val_a = val_pairs[:, 0].astype(np.int64)
    val_b = val_pairs[:, 1].astype(np.int64)

    hit_all, recall_all, overlap_all, found_all, span_all = [], [], [], [], []

    for i in range(0, len(val_a), batch_pairs):
        a_np = val_a[i : i + batch_pairs]
        b_np = val_b[i : i + batch_pairs]

        a = torch.tensor(a_np, device=device, dtype=torch.long)
        b = torch.tensor(b_np, device=device, dtype=torch.long)

        pair_vec = z[a] * z[b]  # [b,D]
        scores = pair_vec @ rel.t()  # [b,R] logits

        vals, idxs = torch.topk(scores, k=k, dim=1)  # [b,K]
        idxs_np = idxs.cpu().numpy()

        pair_key = pair_keys_from_pair_array(canonicalize_pairs(a_np, b_np), num_nodes=num_nodes)
        gt_rows, _ = lookup_gt_rows(pair_key, uniq_keys, gt_mask, num_relations=num_relations)

        gt_tensor = torch.tensor(gt_rows, dtype=torch.float32, device=device)
        val_loss_all.append(float(bce(scores, gt_tensor).item()))

        overlap = gt_rows[np.arange(gt_rows.shape[0])[:, None], idxs_np].sum(axis=1)
        found = gt_rows.sum(axis=1)

        hit = (overlap > 0).astype(np.float32)
        denom = np.minimum(found, k)
        denom = np.where(denom == 0, 1, denom)
        recall = overlap / denom

        span = (vals[:, 0] - vals[:, -1]).detach().cpu().numpy()

        hit_all.append(hit)
        recall_all.append(recall.astype(np.float32))
        overlap_all.append(overlap.astype(np.float32))
        found_all.append(found.astype(np.float32))
        span_all.append(span.astype(np.float32))

    hit = np.concatenate(hit_all) if hit_all else np.array([0.0], dtype=np.float32)
    recall = np.concatenate(recall_all) if recall_all else np.array([0.0], dtype=np.float32)
    overlap = np.concatenate(overlap_all) if overlap_all else np.array([0.0], dtype=np.float32)
    found = np.concatenate(found_all) if found_all else np.array([0.0], dtype=np.float32)
    span = np.concatenate(span_all) if span_all else np.array([0.0], dtype=np.float32)

    return {
        "hit": float(hit.mean()),
        "recall": float(recall.mean()),
        "overlap": float(overlap.mean()),
        "found": float(found.mean()),
        "span": float(span.mean()),
        "val_loss": float(np.mean(val_loss_all) if val_loss_all else 0.0),
    }


def train_full(
    heads: torch.Tensor,
    tails: torch.Tensor,
    rels: torch.Tensor,
    edge_index_base: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: str,
    train_uniq_keys: np.ndarray,
    train_gt_mask: np.ndarray,
    train_eval_pairs: np.ndarray,
    val_pairs: np.ndarray,
    val_uniq_keys: np.ndarray,
    val_gt_mask: np.ndarray,
    eval_every: int = 1,
    eval_topk: int = 15,
    early_metric: str = "recall",
    patience: int = 3,
    min_delta: float = 1e-4,
    hard_k: int = 50,
    hard_warmup: int = 2,
    hard_ramp: int = 4,
    hard_wmax: float = 0.6,
    hard_k_min: int = 10,
    hard_k_max: int = 50,
    hard_topm: int = 10,
    clip_grad: float = 1.0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    best_score = -1e9
    bad_epochs = 0

    encoder = DrugEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
    decoder = DistMultDecoder(num_relations=num_relations, hidden_dim=hidden_dim).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    num_triples = heads.size(0)
    print(f"✅ Train triples (train split): {num_triples:,}")
    print(f"✅ Base graph edges (directed, train graph only): {edge_index_base.size(1):,}")
    print(f"✅ Nodes: {num_nodes}, Relations: {num_relations}")

    heads = heads.to(device)
    tails = tails.to(device)
    rels = rels.to(device)

    history = {"loss": [], "epochs": [], "train_eval_metrics": [], "val_metrics": []}

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        w_hard = linear_ramp(epoch, hard_warmup, hard_ramp, hard_wmax)
        cur_hard_k = hard_k_schedule(epoch, hard_warmup, hard_ramp, hard_k_min, hard_k_max)

        perm = torch.randperm(num_triples, device=device)
        total_loss = 0.0
        steps = 0

        for start in range(0, num_triples, batch_size):
            steps += 1
            end = min(start + batch_size, num_triples)
            batch_idx = perm[start:end]

            a = heads[batch_idx]
            b = tails[batch_idx]
            r_pos = rels[batch_idx]

            z = encoder(edge_index_base)  # [N,D]
            relW = decoder.rel_emb.weight  # [R,D]

            r_rand = torch.randint(0, num_relations, (a.size(0),), device=device)
            r_rand = torch.where(r_rand == r_pos, (r_rand + 1) % num_relations, r_rand)

            if epoch > hard_warmup and w_hard > 0:
                r_hard = mine_hard_negatives(
                    z_det=z.detach(),
                    rel_det=relW.detach(),
                    a=a,
                    b=b,
                    uniq_keys=train_uniq_keys,
                    gt_mask=train_gt_mask,
                    num_nodes=num_nodes,
                    hard_k=cur_hard_k,
                    hard_topm=hard_topm,
                )
            else:
                r_hard = r_rand

            def score(a_ids: torch.Tensor, b_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
                return (z[a_ids] * relW[r_ids] * z[b_ids]).sum(dim=-1)

            pos_logit = score(a, b, r_pos)
            rand_logit = score(a, b, r_rand)
            hard_logit = score(a, b, r_hard)

            loss = (
                bce(pos_logit, torch.ones_like(pos_logit))
                + bce(rand_logit, torch.zeros_like(rand_logit))
                + w_hard * bce(hard_logit, torch.zeros_like(hard_logit))
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=clip_grad)
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch:03d}/{epochs} | avg_loss={avg_loss:.6f}")

        history["loss"].append(avg_loss)
        history["epochs"].append(epoch)

        if epoch % eval_every == 0:
            if len(train_eval_pairs) > 0:
                train_metrics = eval_on_pairs(
                    encoder=encoder,
                    decoder=decoder,
                    edge_index_base=edge_index_base,
                    val_pairs=train_eval_pairs,
                    uniq_keys=train_uniq_keys,
                    gt_mask=train_gt_mask,
                    num_nodes=num_nodes,
                    topk=eval_topk,
                    device=device,
                    batch_pairs=1024,
                )
                train_metrics_copy = {
                    "epoch": epoch,
                    "train_eval_loss": train_metrics["val_loss"],
                    "train_hit": train_metrics["hit"],
                    "train_recall": train_metrics["recall"],
                    "train_overlap": train_metrics["overlap"],
                    "train_found": train_metrics["found"],
                    "train_span": train_metrics["span"],
                }
                history["train_eval_metrics"].append(train_metrics_copy)
                print(
                    f"[TRAIN-EVAL] loss={train_metrics_copy['train_eval_loss']:.6f} | "
                    f"hit@{eval_topk}={train_metrics_copy['train_hit']:.4f} | "
                    f"recall@{eval_topk}={train_metrics_copy['train_recall']:.4f} | "
                    f"overlap={train_metrics_copy['train_overlap']:.2f} | "
                    f"span={train_metrics_copy['train_span']:.6f}"
                )

            metrics = eval_on_pairs(
                encoder=encoder,
                decoder=decoder,
                edge_index_base=edge_index_base,
                val_pairs=val_pairs,
                uniq_keys=val_uniq_keys,
                gt_mask=val_gt_mask,
                num_nodes=num_nodes,
                topk=eval_topk,
                device=device,
                batch_pairs=1024,
            )
            metrics_copy = metrics.copy()
            metrics_copy["epoch"] = epoch
            history["val_metrics"].append(metrics_copy)

            if metrics["span"] < 0.02:
                print("⚠️ span too low -> collapse risk. Consider reducing hard_wmax / hard_k / lr.")

            print(
                f"[VAL] loss={metrics['val_loss']:.6f} | "
                f"hit@{eval_topk}={metrics['hit']:.4f} | "
                f"recall@{eval_topk}={metrics['recall']:.4f} | "
                f"overlap={metrics['overlap']:.2f} | "
                f"span={metrics['span']:.6f}"
            )

            cur = metrics[early_metric]
            improved = cur > best_score + min_delta
            if improved:
                best_score = cur
                bad_epochs = 0
                best_path = os.path.join(out_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state": encoder.state_dict(),
                        "decoder_state": decoder.state_dict(),
                        "num_nodes": num_nodes,
                        "num_relations": num_relations,
                        "hidden_dim": hidden_dim,
                        "val_metrics": metrics,
                    },
                    best_path,
                )
                print(f"✅ New BEST ({early_metric}={best_score:.4f}) saved to: {best_path}")
            else:
                bad_epochs += 1
                print(f"⏳ No improvement. bad_epochs={bad_epochs}/{patience}")
                if bad_epochs >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}. Best {early_metric}={best_score:.4f}")
                    break

        ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "num_nodes": num_nodes,
                "num_relations": num_relations,
                "hidden_dim": hidden_dim,
            },
            ckpt_path,
        )

    final_path = os.path.join(out_dir, "final_model.pt")
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "hidden_dim": hidden_dim,
        },
        final_path,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(out_dir, f"training_history_{timestamp}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"💾 Saved training history: {history_path}")
    print(f"🎉 Done! Saved final model to: {final_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--train-pairs", default=None, help="CSV of train undirected pairs (drug1_id, drug2_id)")
    ap.add_argument("--val-pairs", default="splits/val_pairs.csv", help="CSV of val undirected pairs")
    ap.add_argument("--test-pairs", default=None, help="CSV of test undirected pairs (optional)")
    ap.add_argument("--strict-split", action="store_true", help="Require train/val/test union to match all pairs")
    ap.add_argument("--hard-k", type=int, default=50, help="Top-K candidates used for hard negative mining")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--neg-ratio", type=int, default=1, help="(Legacy arg, currently unused)")
    ap.add_argument("--eval-topk", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument(
        "--train-eval-sample",
        type=int,
        default=1000,
        help="Number of train pairs used to compute train_eval_loss with eval-style BCE (0 disables)",
    )
    ap.add_argument("--train-eval-seed", type=int, default=42, help="Seed for fixed train-eval pair sampling")
    ap.add_argument("--skip-train-eval", action="store_true", help="Skip train-side eval metrics")
    ap.add_argument("--early-metric", choices=["hit", "recall", "overlap"], default="recall")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out-dir", default="runs_decagon_full")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hard-warmup", type=int, default=2)
    ap.add_argument("--hard-ramp", type=int, default=4)
    ap.add_argument("--hard-wmax", type=float, default=0.6)
    ap.add_argument("--hard-k-min", type=int, default=10)
    ap.add_argument("--hard-k-max", type=int, default=50)
    ap.add_argument("--hard-topm", type=int, default=10)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Device:", device)

    df_full = pd.read_parquet(args.parquet)
    for col in ["drug1_id", "drug2_id", "se_id"]:
        if col not in df_full.columns:
            raise ValueError(f"Missing column '{col}' in parquet. Found: {df_full.columns.tolist()}")

    num_nodes = int(max(df_full["drug1_id"].max(), df_full["drug2_id"].max()) + 1)
    num_relations = int(df_full["se_id"].max() + 1)
    full_pair_keys = pair_keys_from_df(df_full, num_nodes=num_nodes)
    full_unique_pair_keys = np.unique(full_pair_keys)

    train_pairs_path = maybe_auto_split_path(args.train_pairs, default_name="train_pairs.csv")
    if train_pairs_path is None:
        warnings.warn(
            "No --train-pairs provided and splits/train_pairs.csv not found. "
            "Falling back to legacy FULL-data training (can leak val/test information)."
        )
        train_pairs = unique_pairs_from_df(df_full)
    else:
        print(f"✅ Using train pairs: {train_pairs_path}")
        train_pairs = load_pairs_csv(train_pairs_path)

    if args.val_pairs is None:
        raise ValueError("--val-pairs is required for validation during training.")
    if not os.path.exists(args.val_pairs):
        raise FileNotFoundError(f"Val pair file not found: {args.val_pairs}")
    val_pairs = load_pairs_csv(args.val_pairs)

    test_pairs_path = maybe_auto_split_path(args.test_pairs, default_name="test_pairs.csv")
    test_pairs = None
    if test_pairs_path is not None:
        print(f"✅ Using test pairs: {test_pairs_path}")
        test_pairs = load_pairs_csv(test_pairs_path)

    train_pair_keys = pair_keys_from_pair_array(train_pairs, num_nodes=num_nodes)
    val_pair_keys = pair_keys_from_pair_array(val_pairs, num_nodes=num_nodes)
    test_pair_keys = None if test_pairs is None else pair_keys_from_pair_array(test_pairs, num_nodes=num_nodes)

    clean_split_mode = train_pairs_path is not None
    if clean_split_mode:
        strict_union = args.strict_split or (test_pair_keys is not None)
        validate_pair_splits(
            train_keys=train_pair_keys,
            val_keys=val_pair_keys,
            test_keys=test_pair_keys,
            all_keys=full_unique_pair_keys,
            strict_union=strict_union,
        )

    df_train = filter_df_by_pair_keys(df_full, full_pair_keys, train_pair_keys)
    df_val = filter_df_by_pair_keys(df_full, full_pair_keys, val_pair_keys)
    df_test = pd.DataFrame(columns=df_full.columns) if test_pair_keys is None else filter_df_by_pair_keys(df_full, full_pair_keys, test_pair_keys)

    if len(df_train) == 0:
        raise ValueError("Train subset is empty after filtering by train pairs.")
    if len(df_val) == 0:
        raise ValueError("Validation subset is empty after filtering by val pairs.")

    print("\n=== Split stats ===")
    print(f"Full triples: {len(df_full):,}")
    print(f"Train triples (used for optimization): {len(df_train):,}")
    print(f"Val triples (used for validation GT):  {len(df_val):,}")
    print(f"Test triples: {len(df_test):,}")
    print(f"Full unique pairs:  {len(full_unique_pair_keys):,}")
    print(f"Train pairs:        {len(train_pair_keys):,}")
    print(f"Val pairs:          {len(val_pair_keys):,}")
    print(f"Test pairs:         {0 if test_pair_keys is None else len(test_pair_keys):,}")
    print("===================\n")

    print("⏳ Building train GT mask for hard negative mining...")
    train_uniq_keys, train_gt_mask = build_pair_gt_mask(df_train, num_nodes=num_nodes, num_relations=num_relations)
    print(f"✅ Train GT mask: pairs={len(train_uniq_keys):,}, relations={num_relations}")

    print("⏳ Building val GT mask for validation...")
    val_uniq_keys, val_gt_mask = build_pair_gt_mask(df_val, num_nodes=num_nodes, num_relations=num_relations)
    print(f"✅ Val GT mask: pairs={len(val_uniq_keys):,}, relations={num_relations}")

    if args.skip_train_eval or args.train_eval_sample <= 0:
        train_eval_pairs = np.empty((0, 2), dtype=np.int64)
        print("⚠️ Train-side eval metrics are disabled (--skip-train-eval or --train-eval-sample <= 0).")
    else:
        train_eval_pairs = sample_pairs_fixed(train_pairs, sample_size=args.train_eval_sample, seed=args.train_eval_seed)
        print(f"✅ Train-eval pairs sample: {len(train_eval_pairs):,} (from {len(train_pairs):,} train pairs)")

    heads = torch.from_numpy(df_train["drug1_id"].to_numpy(dtype=np.int64))
    tails = torch.from_numpy(df_train["drug2_id"].to_numpy(dtype=np.int64))
    rels = torch.from_numpy(df_train["se_id"].to_numpy(dtype=np.int64))

    edge_index_base = build_base_edge_index(df_train, num_nodes=num_nodes, device=device)
    data = Data(num_nodes=num_nodes, edge_index=edge_index_base)
    print("PyG Data (train graph only):")
    print("  num_nodes:", data.num_nodes)
    print("  edge_index shape:", tuple(data.edge_index.shape))

    train_full(
        heads=heads,
        tails=tails,
        rels=rels,
        edge_index_base=edge_index_base,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        out_dir=args.out_dir,
        train_uniq_keys=train_uniq_keys,
        train_gt_mask=train_gt_mask,
        train_eval_pairs=train_eval_pairs,
        val_pairs=val_pairs,
        val_uniq_keys=val_uniq_keys,
        val_gt_mask=val_gt_mask,
        eval_every=args.eval_every,
        eval_topk=args.eval_topk,
        early_metric=args.early_metric,
        patience=args.patience,
        min_delta=args.min_delta,
        hard_k=args.hard_k,
        hard_warmup=args.hard_warmup,
        hard_ramp=args.hard_ramp,
        hard_wmax=args.hard_wmax,
        hard_k_min=args.hard_k_min,
        hard_k_max=args.hard_k_max,
        hard_topm=args.hard_topm,
        clip_grad=args.clip_grad,
    )


if __name__ == "__main__":
    main()
