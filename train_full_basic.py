import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


SCRIPT_DIR = Path(__file__).resolve().parent


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


def resolve_path(path: Optional[str], base_dir: Path = SCRIPT_DIR) -> Optional[str]:
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)
    alt = base_dir / path
    if alt.exists():
        return str(alt)
    return str(p)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA không khả dụng, chuyển sang CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_gib(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)


def canonicalize_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=1)


def build_base_edge_index(drug1: np.ndarray, drug2: np.ndarray, device: torch.device) -> torch.Tensor:
    pairs = np.unique(canonicalize_pairs(drug1, drug2), axis=0)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]], axis=0)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]], axis=0)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    return edge_index


def load_arrays(parquet_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    df = pd.read_parquet(parquet_path, columns=["drug1_id", "drug2_id", "se_id"])
    for col in ["drug1_id", "drug2_id", "se_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in parquet.")

    drug1 = df["drug1_id"].to_numpy(dtype=np.int64, copy=True)
    drug2 = df["drug2_id"].to_numpy(dtype=np.int64, copy=True)
    rels = df["se_id"].to_numpy(dtype=np.int64, copy=True)
    num_nodes = int(max(drug1.max(), drug2.max()) + 1)
    num_relations = int(rels.max() + 1)
    del df
    return drug1, drug2, rels, num_nodes, num_relations


def log_dataset_stats(
    drug1: np.ndarray,
    drug2: np.ndarray,
    rels: np.ndarray,
    edge_index_base: torch.Tensor,
    device: torch.device,
) -> None:
    triple_bytes = drug1.nbytes + drug2.nbytes + rels.nbytes
    edge_bytes = edge_index_base.numel() * edge_index_base.element_size()
    unique_pairs = edge_index_base.size(1) // 2
    print("=== Dataset stats ===")
    print(f"Rows: {len(rels):,}")
    print(f"Nodes: {int(max(drug1.max(), drug2.max()) + 1):,}")
    print(f"Relations: {int(rels.max() + 1):,}")
    print(f"Unique undirected pairs: {unique_pairs:,}")
    print(f"Approx triple-array RAM: {estimate_gib(triple_bytes):.3f} GiB")
    print(f"Approx base-graph RAM: {estimate_gib(edge_bytes):.3f} GiB")
    if device.type == "cpu":
        print("ℹ️ Khuyến nghị ít nhất 16 GB RAM để train CPU ổn định.")
    else:
        print("ℹ️ Nếu VRAM không đủ, giảm --batch-size hoặc chuyển sang CPU.")
    print("=====================")


def save_history(history_path: str, history: Dict[str, List[Dict[str, float]]]) -> None:
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_checkpoint(
    path: str,
    epoch: int,
    encoder: DrugEncoder,
    decoder: DistMultDecoder,
    optimizer: torch.optim.Optimizer,
    num_nodes: int,
    num_relations: int,
    hidden_dim: int,
    history: Dict[str, List[Dict[str, float]]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "hidden_dim": hidden_dim,
            "history": history,
            "args": vars(args),
        },
        path,
    )


def load_torch_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train(args: argparse.Namespace) -> None:
    parquet_path = resolve_path(args.parquet)
    if not parquet_path or not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Không tìm thấy parquet: {args.parquet}")

    device = resolve_device(args.device)
    out_dir_arg = Path(args.out_dir)
    out_dir = out_dir_arg if out_dir_arg.is_absolute() else (SCRIPT_DIR / out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print(f"Device: {device}")
    print(f"Parquet: {parquet_path}")

    drug1, drug2, rels, num_nodes, num_relations = load_arrays(parquet_path)
    edge_index_base = build_base_edge_index(drug1, drug2, device=device)
    log_dataset_stats(drug1, drug2, rels, edge_index_base, device)

    heads_cpu = torch.from_numpy(drug1)
    tails_cpu = torch.from_numpy(drug2)
    rels_cpu = torch.from_numpy(rels)
    if device.type == "cuda":
        heads_cpu = heads_cpu.pin_memory()
        tails_cpu = tails_cpu.pin_memory()
        rels_cpu = rels_cpu.pin_memory()

    del drug1
    del drug2
    del rels

    encoder = DrugEncoder(num_nodes=num_nodes, hidden_dim=args.hidden_dim).to(device)
    decoder = DistMultDecoder(num_relations=num_relations, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss()

    start_epoch = 1
    history: Dict[str, List[Dict[str, float]]] = {"loss": []}

    if args.resume:
        resume_path = resolve_path(args.resume)
        if not resume_path or not os.path.exists(resume_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {args.resume}")
        checkpoint = load_torch_checkpoint(resume_path, torch.device("cpu"))
        if int(checkpoint["num_nodes"]) != num_nodes or int(checkpoint["num_relations"]) != num_relations:
            raise ValueError("Checkpoint không khớp với dataset hiện tại.")
        ckpt_hidden = int(checkpoint["hidden_dim"])
        if ckpt_hidden != args.hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim={ckpt_hidden} nhưng CLI đang dùng hidden_dim={args.hidden_dim}. "
                "Hãy chạy lại với --hidden-dim khớp checkpoint."
            )
        encoder.load_state_dict(checkpoint["encoder_state"])
        decoder.load_state_dict(checkpoint["decoder_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", history)
        print(f"✅ Resume từ checkpoint: {resume_path} (epoch {start_epoch - 1})")

    history_path = out_dir / "training_history.json"
    num_triples = heads_cpu.size(0)
    print(f"Train triples: {num_triples:,}")
    print(f"Base graph edges (directed): {edge_index_base.size(1):,}")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        decoder.train()

        perm = torch.randperm(num_triples)
        total_loss = 0.0
        total_examples = 0
        steps = 0

        for start in range(0, num_triples, args.batch_size):
            end = min(start + args.batch_size, num_triples)
            batch_idx = perm[start:end]

            a = heads_cpu[batch_idx].to(device, non_blocking=device.type == "cuda")
            b = tails_cpu[batch_idx].to(device, non_blocking=device.type == "cuda")
            r_pos = rels_cpu[batch_idx].to(device, non_blocking=device.type == "cuda")

            z = encoder(edge_index_base)
            rel_weight = decoder.rel_emb.weight

            r_neg = torch.randint(0, num_relations, (a.size(0),), device=device)
            r_neg = torch.where(r_neg == r_pos, (r_neg + 1) % num_relations, r_neg)

            pos_logit = (z[a] * rel_weight[r_pos] * z[b]).sum(dim=-1)
            neg_logit = (z[a] * rel_weight[r_neg] * z[b]).sum(dim=-1)

            loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(neg_logit, torch.zeros_like(neg_logit))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
            optimizer.step()

            batch_size_actual = int(a.size(0))
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual
            steps += 1

        avg_loss = total_loss / max(1, total_examples)
        history["loss"].append({"epoch": epoch, "avg_loss": avg_loss, "steps": steps})
        save_history(str(history_path), history)

        checkpoint_last = out_dir / "checkpoint_last.pt"
        save_checkpoint(
            str(checkpoint_last),
            epoch,
            encoder,
            decoder,
            optimizer,
            num_nodes,
            num_relations,
            args.hidden_dim,
            history,
            args,
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            epoch_ckpt = out_dir / f"ckpt_epoch_{epoch:04d}.pt"
            save_checkpoint(
                str(epoch_ckpt),
                epoch,
                encoder,
                decoder,
                optimizer,
                num_nodes,
                num_relations,
                args.hidden_dim,
                history,
                args,
            )

        print(f"Epoch {epoch:03d}/{args.epochs} | avg_loss={avg_loss:.6f} | steps={steps}")

    final_path = out_dir / "final_model.pt"
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "hidden_dim": args.hidden_dim,
        },
        str(final_path),
    )
    print(f"✅ Saved final model: {final_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--out-dir", default="runs_decagon_full")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
