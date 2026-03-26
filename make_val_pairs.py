# make_val_pairs.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def canonicalize_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.stack([lo, hi], axis=1)


def pair_keys(pairs: np.ndarray, stride: int) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0,), dtype=np.int64)
    return pairs[:, 0].astype(np.int64) * stride + pairs[:, 1].astype(np.int64)


def save_pairs(path: Path, pairs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["drug1_id", "drug2_id"]).to_csv(path, index=False)


def pct(n: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(n) / float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--out-dir", default=None, help="Directory to write train/val/test pair CSVs")
    ap.add_argument("--out", default=None, help="(Legacy) val_pairs.csv path")
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-8):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.8f}")
    for name, ratio in [
        ("train-ratio", args.train_ratio),
        ("val-ratio", args.val_ratio),
        ("test-ratio", args.test_ratio),
    ]:
        if ratio < 0:
            raise ValueError(f"{name} must be >= 0, got {ratio}")

    if args.out_dir is None:
        if args.out is not None:
            out_parent = Path(args.out).parent
            out_dir = out_parent if str(out_parent) not in {"", "."} else Path("splits")
        else:
            out_dir = Path("splits")
    else:
        out_dir = Path(args.out_dir)

    train_path = out_dir / "train_pairs.csv"
    val_path = out_dir / "val_pairs.csv"
    test_path = out_dir / "test_pairs.csv"

    df = pd.read_parquet(args.parquet)
    for col in ["drug1_id", "drug2_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.parquet}")

    all_pairs = canonicalize_pairs(
        df["drug1_id"].to_numpy(np.int64),
        df["drug2_id"].to_numpy(np.int64),
    )
    all_pairs = np.unique(all_pairs, axis=0)  # unique undirected pairs
    n_total = int(len(all_pairs))
    if n_total == 0:
        raise ValueError("No pairs found in parquet.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_pairs)

    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train : n_train + n_val]
    test_pairs = all_pairs[n_train + n_val :]

    stride = int(all_pairs[:, 1].max()) + 1
    k_train = pair_keys(train_pairs, stride)
    k_val = pair_keys(val_pairs, stride)
    k_test = pair_keys(test_pairs, stride)

    if np.intersect1d(k_train, k_val).size > 0:
        raise AssertionError("Train/Val overlap detected")
    if np.intersect1d(k_train, k_test).size > 0:
        raise AssertionError("Train/Test overlap detected")
    if np.intersect1d(k_val, k_test).size > 0:
        raise AssertionError("Val/Test overlap detected")

    union_size = np.unique(np.concatenate([k_train, k_val, k_test], axis=0)).size
    if union_size != n_total:
        raise AssertionError(f"Split union mismatch: union={union_size}, total={n_total}")

    save_pairs(train_path, train_pairs)
    save_pairs(val_path, val_pairs)
    save_pairs(test_path, test_pairs)

    # Backward compatibility: old script used --out for val split only.
    if args.out is not None:
        legacy_val_path = Path(args.out)
        if legacy_val_path.resolve() != val_path.resolve():
            save_pairs(legacy_val_path, val_pairs)
            print("✅ Saved legacy val path:", legacy_val_path)

    print("✅ Pair split saved:")
    print("  -", train_path)
    print("  -", val_path)
    print("  -", test_path)
    print("Total unique pairs:", n_total)
    print(f"Train pairs: {len(train_pairs)} ({pct(len(train_pairs), n_total):.2f}%)")
    print(f"Val pairs:   {len(val_pairs)} ({pct(len(val_pairs), n_total):.2f}%)")
    print(f"Test pairs:  {len(test_pairs)} ({pct(len(test_pairs), n_total):.2f}%)")


if __name__ == "__main__":
    main()
