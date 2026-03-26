import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[.,;:/\[\]\{\}\(\)\"'`]+")
_CID_RE = re.compile(r"CID0*(\d+)", re.IGNORECASE)
SCRIPT_DIR = Path(__file__).resolve().parent
_DRUG_NAME_CACHE: Dict[str, List[Dict[str, Optional[str]]]] = {}


def norm_name(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = s.replace("-", " ").replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


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


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def extract_cid(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = _CID_RE.search(s)
    if m:
        return str(int(m.group(1)))
    if s.isdigit():
        return str(int(s))
    return None


def _canon_stitch_from_cid(cid: str) -> str:
    return f"CID{cid.zfill(9)}"


def _pick_stitch(row: Dict[str, str], cols: Sequence[str]) -> Optional[str]:
    for col in cols:
        val = (row.get(col) or "").strip()
        if not val:
            continue
        if col in {"cid", "pubchem_cid"}:
            cid = extract_cid(val)
            if cid:
                return _canon_stitch_from_cid(cid)
            return None
        return val
    return None


def load_drug_names(csv_path: str) -> List[Dict[str, Optional[str]]]:
    csv_path = resolve_path(csv_path)
    if csv_path in _DRUG_NAME_CACHE:
        return _DRUG_NAME_CACHE[csv_path]
    if not os.path.exists(csv_path):
        raise SystemExit(f"❌ Không tìm thấy file drug name map: {csv_path}")

    out: List[Dict[str, Optional[str]]] = []
    seen: Dict[str, Dict[str, Optional[str]]] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "drug_name" not in reader.fieldnames:
            raise SystemExit("❌ File drug name map thiếu cột 'drug_name'.")
        cols = [c for c in ["stitch_id", "stitch", "cid", "pubchem_cid"] if c in reader.fieldnames]
        for row in reader:
            name = (row.get("drug_name") or "").strip()
            if not name:
                continue
            key = norm_name(name)
            if not key:
                continue
            stitch = _pick_stitch(row, cols) if cols else None
            rec = {"name": name, "norm": key, "stitch": stitch, "cid": extract_cid(stitch)}
            existing = seen.get(key)
            if existing:
                if not existing.get("stitch") and stitch:
                    seen[key] = rec
                continue
            seen[key] = rec

    out = list(seen.values())

    if not out:
        raise SystemExit("❌ Không đọc được tên thuốc từ drug name map.")

    _DRUG_NAME_CACHE[csv_path] = out
    return out


def filter_by_drugbank(
    records: Sequence[Dict[str, Optional[str]]],
    index_path: str,
) -> Tuple[List[Dict[str, Optional[str]]], Dict[str, Dict]]:
    index_path = resolve_path(index_path)
    if not os.path.exists(index_path):
        raise SystemExit(f"❌ Không tìm thấy DrugBank index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    name_to_ids = index.get("name_to_ids", {})
    if not isinstance(name_to_ids, dict):
        raise SystemExit("❌ DrugBank index không hợp lệ: thiếu name_to_ids.")
    cid_to_ids = index.get("cid_to_ids", {})
    if not isinstance(cid_to_ids, dict):
        raise SystemExit("❌ DrugBank index không hợp lệ: thiếu cid_to_ids.")

    filtered = []
    for rec in records:
        key = rec.get("norm") or ""
        cid = rec.get("cid")
        if key in name_to_ids or (cid and cid in cid_to_ids):
            filtered.append(rec)
    return filtered, {"name_to_ids": name_to_ids, "cid_to_ids": cid_to_ids}


def _is_resolvable(
    rec: Dict[str, Optional[str]],
    name_to_ids: Dict[str, List[str]],
    cid_to_ids: Dict[str, List[str]],
) -> bool:
    key = rec.get("norm") or ""
    cid = rec.get("cid")
    return key in name_to_ids or (cid and cid in cid_to_ids)


def build_command(
    name_a: str,
    name_b: str,
    topk: int,
    device: str,
    lang: str,
    extra_args: Sequence[str],
    stitch_a: Optional[str] = None,
    stitch_b: Optional[str] = None,
) -> List[str]:
    predict_script = str((SCRIPT_DIR / "predict_pair.py").resolve())
    cmd = [
        sys.executable,
        predict_script,
        "--name-a",
        name_a,
        "--name-b",
        name_b,
        "--topk",
        str(topk),
        "--device",
        device,
        "--lang",
        lang,
    ]
    has_stitch_a = any(arg.startswith("--stitch-a") for arg in extra_args)
    has_stitch_b = any(arg.startswith("--stitch-b") for arg in extra_args)
    if stitch_a and not has_stitch_a:
        cmd.extend(["--stitch-a", stitch_a])
    if stitch_b and not has_stitch_b:
        cmd.extend(["--stitch-b", stitch_b])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def print_run_command(cmd: Sequence[str]) -> None:
    shown = ["python", "predict_pair.py"] + list(cmd[2:])
    quoted = " ".join(shlex.quote(c) for c in shown)
    print(f"Run: {quoted}")


def run_predict(cmd: Sequence[str]) -> None:
    print_run_command(cmd)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"⚠️ predict_pair.py exited with code {proc.returncode}")
        raise SystemExit(proc.returncode)


def run_predict_inproc(cmd: Sequence[str]) -> None:
    print_run_command(cmd)
    try:
        import predict_pair
    except Exception as exc:
        raise SystemExit(f"❌ Không thể import predict_pair để chạy inproc: {exc}")

    prev_argv = sys.argv
    sys.argv = list(cmd[1:])
    try:
        predict_pair.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print(f"⚠️ predict_pair.py exited with code {code}")
        raise SystemExit(code)
    finally:
        sys.argv = prev_argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--lang", choices=["en", "vi"], default="en")
    ap.add_argument("--drug-name-map", default="decagon_processed/drug_id_name_map.csv")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--require-drugbank", action="store_true")
    ap.add_argument("--prefer-drugbank", dest="prefer_drugbank", action="store_true", default=True)
    ap.add_argument("--no-prefer-drugbank", dest="prefer_drugbank", action="store_false")
    ap.add_argument("--drugbank-index", default="cache/drugbank_index.json")
    ap.add_argument("--inproc", action="store_true", help="Chạy predict_pair trong-process để tái sử dụng model/parquet")
    args, extra = ap.parse_known_args()
    runner = run_predict_inproc if args.inproc else run_predict

    if args.seed is not None:
        random.seed(args.seed)

    input_mode = args.name_a and args.name_b
    if input_mode:
        names = [
            {"name": args.name_a, "norm": norm_name(args.name_a), "stitch": None, "cid": None},
            {"name": args.name_b, "norm": norm_name(args.name_b), "stitch": None, "cid": None},
        ]
    else:
        records = load_drug_names(args.drug_name_map)
        name_to_ids = {}
        cid_to_ids = {}
        resolvable = None
        if args.require_drugbank or args.prefer_drugbank:
            resolvable, index = filter_by_drugbank(records, args.drugbank_index)
            name_to_ids = index["name_to_ids"]
            cid_to_ids = index["cid_to_ids"]
            if args.require_drugbank:
                if len(resolvable) < 2:
                    raise SystemExit(
                        "❌ Không đủ >=2 thuốc khớp DrugBank index. "
                        "Hãy thử tắt --require-drugbank hoặc kiểm tra lại index."
                    )
                records = resolvable
        names = records

    if input_mode:
        print("Mode: input")
        name_a, name_b = args.name_a, args.name_b
        print(f"Chosen pair: {name_a}  <->  {name_b}")
        cmd = build_command(name_a, name_b, args.topk, args.device, args.lang, extra)
        runner(cmd)
        return

    print("Mode: random")
    if args.n <= 0:
        raise SystemExit("❌ --n phải >= 1")

    pool = list(names)
    if len(pool) < 2:
        raise SystemExit("❌ Không đủ >=2 tên thuốc để random.")

    for i in range(args.n):
        rec_a = rec_b = None
        if args.prefer_drugbank and not args.require_drugbank and name_to_ids and cid_to_ids:
            resolvable_pool = resolvable if resolvable is not None else [r for r in pool if _is_resolvable(r, name_to_ids, cid_to_ids)]
            if len(resolvable_pool) >= 2:
                rec_a, rec_b = random.sample(resolvable_pool, 2)
            else:
                for _ in range(50):
                    ra, rb = random.sample(pool, 2)
                    if _is_resolvable(ra, name_to_ids, cid_to_ids) and _is_resolvable(rb, name_to_ids, cid_to_ids):
                        rec_a, rec_b = ra, rb
                        break
                if rec_a is None:
                    warnings.warn("Không tìm được cặp resolvable theo DrugBank, fallback random bất kỳ.")

        if rec_a is None or rec_b is None:
            rec_a, rec_b = random.sample(pool, 2)

        print(f"Chosen pair [{i + 1}/{args.n}]: {rec_a['name']}  <->  {rec_b['name']}")
        cmd = build_command(
            rec_a["name"],
            rec_b["name"],
            args.topk,
            args.device,
            args.lang,
            extra,
            stitch_a=rec_a.get("stitch"),
            stitch_b=rec_b.get("stitch"),
        )
        runner(cmd)


if __name__ == "__main__":
    main()
