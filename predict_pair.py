import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
import torch.nn.functional as F
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch import nn
from torch_geometric.nn import SAGEConv
from rich.console import Console
from rich.table import Table
from drugbank_ref import DrugBankRef
from local_translator import LocalTranslator, protect_drug_names, restore_drug_names


SCRIPT_DIR = Path(__file__).resolve().parent

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, nn.Module, int, int, int]] = {}
_DATA_CACHE: Dict[Tuple[str, int, str], Tuple[pd.DataFrame, torch.Tensor]] = {}
_MAPS_CACHE: Dict[Tuple[str, str, str], Tuple[dict, dict, dict, dict]] = {}
_NAME_DF_CACHE: Dict[str, pd.DataFrame] = {}
_STITCH_NAME_CACHE: Dict[str, dict] = {}
_ID2NAME_CACHE: Dict[str, dict] = {}
_TRANSLATOR_CACHE: Dict[Tuple[str, str, bool], LocalTranslator] = {}
_PLATT_CACHE: Dict[str, Tuple[float, float]] = {}


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


def load_torch_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_platt_cached(path: Optional[str]) -> Tuple[float, float]:
    if not path or not os.path.exists(path):
        return 1.0, 0.0
    if path in _PLATT_CACHE:
        return _PLATT_CACHE[path]
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    a_platt = float(j.get("a", 1.0))
    b_platt = float(j.get("b", 0.0))
    _PLATT_CACHE[path] = (a_platt, b_platt)
    return a_platt, b_platt


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA không khả dụng, chuyển sang CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_translator(model_name: str, device: torch.device, offline_only: bool = True) -> LocalTranslator:
    key = (model_name, str(device), offline_only)
    if key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[key]
    tr = LocalTranslator(model_name=model_name, device=str(device), offline_only=offline_only)
    _TRANSLATOR_CACHE[key] = tr
    return tr


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

def build_base_edge_index(df: pd.DataFrame, num_nodes: int, device: torch.device) -> torch.Tensor:
    d1 = df["drug1_id"].to_numpy(dtype=np.int64)
    d2 = df["drug2_id"].to_numpy(dtype=np.int64)

    lo = np.minimum(d1, d2)
    hi = np.maximum(d1, d2)
    pairs = np.stack([lo, hi], axis=1)
    pairs = np.unique(pairs, axis=0)

    src = np.concatenate([pairs[:, 0], pairs[:, 1]], axis=0)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]], axis=0)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    assert edge_index.max().item() < num_nodes
    return edge_index

def load_maps(drug_map_csv: str, se_map_csv: str, se_name_map_csv: str):
    drug_map = pd.read_csv(drug_map_csv)
    se_map = pd.read_csv(se_map_csv)

    stitch2id = dict(zip(drug_map["stitch_id"].astype(str), drug_map["drug_id"].astype(int)))
    id2stitch = dict(zip(drug_map["drug_id"].astype(int), drug_map["stitch_id"].astype(str)))

    se_id2cui = dict(zip(se_map["se_id"].astype(int), se_map["side_effect_cui"].astype(str)))

    se_id2name = {}
    if os.path.exists(se_name_map_csv):
        se_name = pd.read_csv(se_name_map_csv)
        se_id2name = dict(zip(se_name["se_id"].astype(int), se_name["side_effect_name"].astype(str)))
    else:
        se_id2name = {k: v for k, v in se_id2cui.items()}

    return stitch2id, id2stitch, se_id2name, se_id2cui


def load_maps_cached(drug_map_csv: str, se_map_csv: str, se_name_map_csv: str):
    key = (drug_map_csv, se_map_csv, se_name_map_csv)
    if key in _MAPS_CACHE:
        return _MAPS_CACHE[key]
    data = load_maps(drug_map_csv, se_map_csv, se_name_map_csv)
    _MAPS_CACHE[key] = data
    return data


def load_model_cached(model_path: str, device: torch.device):
    key = (model_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ckpt = load_torch_checkpoint(model_path, device)
    num_nodes = int(ckpt["num_nodes"])
    num_relations = int(ckpt["num_relations"])
    hidden_dim = int(ckpt["hidden_dim"])

    encoder = DrugEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
    decoder = DistMultDecoder(num_relations=num_relations, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()

    _MODEL_CACHE[key] = (encoder, decoder, num_nodes, num_relations, hidden_dim)
    return _MODEL_CACHE[key]


def load_parquet_cached(parquet_path: str, num_nodes: int, device: torch.device):
    key = (parquet_path, num_nodes, str(device))
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    df = pd.read_parquet(parquet_path)
    edge_index_base = build_base_edge_index(df, num_nodes=num_nodes, device=device)
    _DATA_CACHE[key] = (df, edge_index_base)
    return _DATA_CACHE[key]


def get_drug_name_df(drug_name_map_csv: str) -> pd.DataFrame:
    if drug_name_map_csv in _NAME_DF_CACHE:
        return _NAME_DF_CACHE[drug_name_map_csv]
    df = pd.read_csv(drug_name_map_csv)
    _NAME_DF_CACHE[drug_name_map_csv] = df
    return df

def name_to_stitch(name: str, drug_name_map_csv: str, strict: bool = True, df_cache: Optional[pd.DataFrame] = None):
    df = df_cache if df_cache is not None else pd.read_csv(drug_name_map_csv)
    if "drug_name" not in df.columns:
        raise ValueError(f'Không tìm thấy cột "drug_name" trong {drug_name_map_csv}')

    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = " ".join(s.split())
        return s

    if "_drug_name_norm" not in df.columns:
        df["_drug_name_norm"] = df["drug_name"].astype(str).map(_norm)
    name_norm = _norm(name)
    if not name_norm:
        raise ValueError("Tên thuốc trống.")
    hit = df[df["_drug_name_norm"] == name_norm]
    if len(hit) == 1:
        return str(hit.iloc[0]["stitch_id"])
    alias_cols = [c for c in ["synonyms", "aliases"] if c in df.columns]
    if alias_cols:
        for col in alias_cols:
            norm_col = f"_alias_norm_{col}"
            if norm_col not in df.columns:
                df[norm_col] = df[col].astype(str).map(_norm)
            hit_alias = df[df[norm_col] == name_norm]
            if len(hit_alias) == 1:
                return str(hit_alias.iloc[0]["stitch_id"])
            if len(hit_alias) > 1:
                hit = hit_alias
                break
    if len(hit) == 0:
        hit = df[df["_drug_name_norm"].str.contains(name_norm, regex=False, na=False)]

    if len(hit) == 0:
        if strict:
            raise ValueError(
                f'Không tìm thấy tên "{name}" trong {drug_name_map_csv}. '
                'Hãy thử dùng --stitch-a/--stitch-b hoặc tên cụ thể hơn.'
            )
        return None

    if len(hit) > 1:
        name_len = len(name_norm)
        hit = hit.copy()
        hit["_len_diff"] = (hit["_drug_name_norm"].str.len() - name_len).abs()
        hit["_starts"] = hit["_drug_name_norm"].str.startswith(name_norm)
        hit = hit.sort_values(by=["_len_diff", "_starts"], ascending=[True, False])
        warnings.warn(
            f'Có nhiều kết quả cho "{name}". '
            'Đang chọn kết quả gần nhất; hãy dùng --stitch-a/--stitch-b hoặc tên cụ thể hơn nếu cần.'
        )
        return str(hit.iloc[0]["stitch_id"])

    return str(hit.iloc[0]["stitch_id"])

def load_stitch_name_map(drug_name_map_csv: str):
    if not os.path.exists(drug_name_map_csv):
        return {}
    if drug_name_map_csv in _STITCH_NAME_CACHE:
        return _STITCH_NAME_CACHE[drug_name_map_csv]
    df = pd.read_csv(drug_name_map_csv)
    if "stitch_id" not in df.columns or "drug_name" not in df.columns:
        return {}
    df["stitch_id"] = df["stitch_id"].astype(str)
    df["drug_name"] = df["drug_name"].astype(str)
    mp = df.dropna(subset=["stitch_id", "drug_name"]).drop_duplicates("stitch_id").set_index("stitch_id")["drug_name"].to_dict()
    _STITCH_NAME_CACHE[drug_name_map_csv] = mp
    return mp

def display_drug_label(stitch_id: str, provided_name: str, stitch2name: dict):
    if provided_name:
        return provided_name
    if stitch_id in stitch2name:
        return stitch2name[stitch_id]
    return stitch_id

def short_text(s: str, max_len: int = 40) -> str:
    s = " ".join((s or "").strip().split())
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return s[:max_len]
    return f"{s[: max_len - 1]}…"

def load_id2drug_name(drug_name_map_csv: str):
    if not os.path.exists(drug_name_map_csv):
        return {}
    if drug_name_map_csv in _ID2NAME_CACHE:
        return _ID2NAME_CACHE[drug_name_map_csv]
    df = pd.read_csv(drug_name_map_csv)
    if "drug_id" not in df.columns or "drug_name" not in df.columns:
        return {}
    df = df.dropna(subset=["drug_id", "drug_name"])
    df["drug_id"] = df["drug_id"].astype(int)
    df["drug_name"] = df["drug_name"].astype(str)
    mp = dict(zip(df["drug_id"], df["drug_name"]))
    _ID2NAME_CACHE[drug_name_map_csv] = mp
    return mp


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "vi"], default="en")
    ap.add_argument("--translator-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--se-name-map", default="decagon_processed/side_effect_id_name_map.csv")
    ap.add_argument("--model", default="runs_decagon_full/final_model.pt")
    ap.add_argument("--parquet", default="decagon_processed/decagon_polypharmacy_mapped.parquet")
    ap.add_argument("--hard-warmup", type=int, default=2, help="Số epoch đầu chỉ random negatives")
    ap.add_argument("--hard-ramp", type=int, default=4, help="Số epoch để tăng dần weight hard")
    ap.add_argument("--hard-wmax", type=float, default=0.6, help="Max weight cho hard negative loss")
    ap.add_argument("--hard-k-min", type=int, default=10, help="hard_k nhỏ lúc đầu")
    ap.add_argument("--hard-k-max", type=int, default=50, help="hard_k lớn dần")
    ap.add_argument("--hard-topm", type=int, default=10, help="chọn ngẫu nhiên trong top-m hard candidates (đa dạng hơn)")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping để tránh training nổ")
    ap.add_argument("--drug-map", default="decagon_processed/drug_id_map.csv")
    ap.add_argument("--se-map", default="decagon_processed/side_effect_id_map.csv")
    ap.add_argument("--stitch-a", default=None, help='VD: "CID000002173"')
    ap.add_argument("--stitch-b", default=None, help='VD: "CID000003345"')
    ap.add_argument("--name-a", default=None, help='VD: "Prilosec" (case-insensitive)')
    ap.add_argument("--name-b", default=None, help='VD: "Lansoprazole" (case-insensitive)')
    ap.add_argument("--drug-name-map", default="decagon_processed/drug_id_name_map.csv")
    ap.add_argument("--drugbank-json", default="drugbank_ddi.json")
    ap.add_argument("--db-index", default="cache/drugbank_index.json")
    ap.add_argument("--drugbank-a-name", default=None, help="Tên thuốc A để tra DrugBank (nếu không có sẽ dùng drug name từ mapping nếu có)")
    ap.add_argument("--drugbank-b-name", default=None, help="Tên thuốc B để tra DrugBank")
    ap.add_argument("--db-debug", action="store_true", help="In debug cho DrugBank resolver (không đổi format bảng)")
    ap.add_argument("--display-maxlen", type=int, default=40, help="Giới hạn độ dài hiển thị tên thuốc")
    ap.add_argument("--debug", action="store_true", help="In debug chung cho model scoring")
    ap.add_argument("--debug-names", action="store_true", help="In debug tên/candidates DrugBank (không đổi format bảng)")
    ap.add_argument("--debug-drugbank", action="store_true", help="In full candidates DrugBank ngoài bảng (không đổi format bảng)")
    ap.add_argument("--hf-online", action="store_true", help="Cho phép tải model dịch từ HuggingFace nếu cache local chưa có")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--platt", default="calib/platt_best.json", help="Path to calib/platt_*.json")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    if (args.stitch_a is None or args.stitch_b is None) and (args.name_a is None or args.name_b is None):
        raise SystemExit("❌ Bạn phải truyền (--stitch-a & --stitch-b) hoặc (--name-a & --name-b).")
    args.model = resolve_path(args.model)
    args.parquet = resolve_path(args.parquet)
    args.drug_map = resolve_path(args.drug_map)
    args.se_map = resolve_path(args.se_map)
    args.se_name_map = resolve_path(args.se_name_map)
    args.drug_name_map = resolve_path(args.drug_name_map)
    args.drugbank_json = resolve_path(args.drugbank_json)
    args.db_index = resolve_path(args.db_index)
    if args.platt:
        args.platt = resolve_path(args.platt)

    device = resolve_device(args.device)
    print("\nDevice:", device)

    decagon_ok = True
    miss_msgs = []

    try:
        stitch2id, id2stitch, se_id2name, se_id2cui = load_maps_cached(args.drug_map, args.se_map, args.se_name_map)
    except Exception as e:
        stitch2id, id2stitch, se_id2name, se_id2cui = {}, {}, {}, {}
        decagon_ok = False
        miss_msgs.append(f"⚠️ Không đọc được mapping Decagon/TWOSIDES: {e}")
    name_df = None
    if args.drug_name_map and os.path.exists(args.drug_name_map):
        name_df = get_drug_name_df(args.drug_name_map)
    elif args.name_a or args.name_b:
        decagon_ok = False
        miss_msgs.append(f"⚠️ Không tìm thấy file drug_name_map: {args.drug_name_map}")

    if name_df is not None and args.name_a and args.stitch_a is None:
        args.stitch_a = name_to_stitch(args.name_a, args.drug_name_map, strict=False, df_cache=name_df)
        if args.stitch_a is None:
            decagon_ok = False
            miss_msgs.append(f'⚠️ Không tìm thấy thuốc "{args.name_a}" trong dataset Decagon/TWOSIDES nên không thể dự đoán tác dụng phụ từ mô hình.')
    if name_df is not None and args.name_b and args.stitch_b is None:
        args.stitch_b = name_to_stitch(args.name_b, args.drug_name_map, strict=False, df_cache=name_df)
        if args.stitch_b is None:
            decagon_ok = False
            miss_msgs.append(f'⚠️ Không tìm thấy thuốc "{args.name_b}" trong dataset Decagon/TWOSIDES nên không thể dự đoán tác dụng phụ từ mô hình.')
    if args.stitch_a not in stitch2id or args.stitch_b not in stitch2id:
        decagon_ok = False
        if args.stitch_a not in stitch2id:
            miss_msgs.append(f'⚠️ Không tìm thấy STITCH ID cho thuốc A: {args.stitch_a}')
        if args.stitch_b not in stitch2id:
            miss_msgs.append(f'⚠️ Không tìm thấy STITCH ID cho thuốc B: {args.stitch_b}')

    stitch2name = load_stitch_name_map(args.drug_name_map)
    id2drug_name = load_id2drug_name(args.drug_name_map)

    nameA = args.name_a or args.stitch_a
    nameB = args.name_b or args.stitch_b

    if decagon_ok:
        a = stitch2id[args.stitch_a]
        b = stitch2id[args.stitch_b]
        nameA = id2drug_name.get(a, nameA)
        nameB = id2drug_name.get(b, nameB)

    queryA = args.drugbank_a_name or nameA
    queryB = args.drugbank_b_name or nameB

    db = DrugBankRef(args.drugbank_json, index_path=args.db_index)
    candidatesA = db.resolve_candidates(queryA, args.stitch_a)
    candidatesB = db.resolve_candidates(queryB, args.stitch_b)
    headerA = db.format_candidates_label(candidatesA) or queryA
    headerB = db.format_candidates_label(candidatesB) or queryB

    if args.debug_names:
        def _fmt_full(dbids):
            out = []
            for d in dbids or []:
                nm = db.dbid_to_name(d) or "N/A"
                out.append(f"{d} ({nm})")
            return out

        print("\n[Name debug]")
        print(f"- inputA: {args.name_a}")
        print(f"- stitchA: {args.stitch_a}")
        print(f"- candidatesA: {_fmt_full(candidatesA)}")
        print(f"- inputB: {args.name_b}")
        print(f"- stitchB: {args.stitch_b}")
        print(f"- candidatesB: {_fmt_full(candidatesB)}")

    translator = None
    if args.lang == "vi":
        translator_candidate = get_translator(args.translator_model, device, offline_only=not args.hf_online)
        if translator_candidate.available:
            translator = translator_candidate
        else:
            print(f"⚠️ {translator_candidate.warning_message()}")

    if decagon_ok:
        a = stitch2id[args.stitch_a]
        b = stitch2id[args.stitch_b]
        print(f"Input: {args.stitch_a} -> {a}, {args.stitch_b} -> {b}\n")

        encoder, decoder, num_nodes, num_relations, _ = load_model_cached(args.model, device)
        df, edge_index_base = load_parquet_cached(args.parquet, num_nodes=num_nodes, device=device)

        with torch.inference_mode():
            z = encoder(edge_index_base)
            za = z[a]
            zb = z[b]
            rel = decoder.rel_emb.weight
            scores = (rel * (za * zb)).sum(dim=-1)

            topk = min(args.topk, num_relations)
            vals, idxs = torch.topk(scores, k=topk)

            console = Console()
            sid_list = idxs.cpu().tolist()
            sc_list = vals.cpu().tolist()

            a_platt, b_platt = load_platt_cached(args.platt)
            if args.debug:
                s = scores.detach().float().cpu()
                v = vals.detach().float().cpu()
                print(f"[Score stats] min = {s.min():.4f}, max = {s.max():.4f}, mean = {s.mean():.4f}, std = {s.std():.6f}")
                print(f"[TopK stats] min = {v.min():.4f}, max = {v.max():.4f}, mean = {v.mean():.4f}, std = {v.std():.6f}, span = {float(v.max() - v.min()):.6f}")
                print(f"[Platt] a = {a_platt:.6f}, b = {b_platt:.6f}")

            se_names_en = [se_id2name.get(sid, f"se_id={sid}") for sid in sid_list]
            if translator:
                se_names_show = translator.translate_many(se_names_en, batch_size=16)
            else:
                se_names_show = se_names_en

            a_label_short = short_text(headerA or display_drug_label(args.stitch_a, args.name_a, stitch2name), max_len=args.display_maxlen)
            b_label_short = short_text(headerB or display_drug_label(args.stitch_b, args.name_b, stitch2name), max_len=args.display_maxlen)
            table = Table(title=f"\n\nTop-{topk} các tác dụng phụ được dự đoán\nGiữa {a_label_short} và {b_label_short}")
            table.add_column("Rank", justify="center")
            table.add_column("Polypharmacy\nSide Effect ID", justify="center")
            table.add_column("Raw Score", justify="center")
            table.add_column("Risk Score", justify="center")
            table.add_column("Side Effect Name", overflow="fold", justify="left")
            table.add_column("CUI", justify="center")

            for i, (sid, sc) in enumerate(zip(sid_list, sc_list), start=1):
                logit = float(sc)
                se_name_show = se_names_show[i-1]
                cui = se_id2cui.get(sid, "")
                risk_pct = 100.0 * (1.0 / (1.0 + np.exp(-(a_platt * logit + b_platt))))

                table.add_row(
                    str(i),
                    str(sid),
                    f"{sc:.4f}",
                    f"{risk_pct:.2f}%",
                    se_name_show,
                    cui
                )

            console.print(table)

            gt = df[
                ((df["drug1_id"] == a) & (df["drug2_id"] == b)) |
                ((df["drug1_id"] == b) & (df["drug2_id"] == a))
            ]["se_id"]

            gt_set = set(gt.astype(int).tolist())
            pred_set = set(idxs.cpu().tolist())
            inter = gt_set.intersection(pred_set)

            summary = Table(title="\nSummary")
            summary.add_column("Metric")
            summary.add_column("Value", justify="right")
            summary.add_row("Found in dataset", str(len(gt_set)))
            summary.add_row(f"Predicted correctly over Top {topk}", str(len(inter)))
            console.print(summary)

            print("\n")
            print("=" * 150)
            source_block = (
                "[Nguồn]\n"
                "- Nghiên cứu này sử dụng bộ dữ liệu tác dụng phụ dược phẩm từ TWOSIDES, ban đầu được giới thiệu trong khuôn khổ Decagon.\n"
                "- Bộ dữ liệu được xây dựng bằng cách khai thác Hệ thống Báo cáo Sự cố Bất lợi của FDA (FAERS) và chứa các tác dụng phụ được báo cáo liên quan đến các cặp thuốc.\n"
                "- Bộ dữ liệu bao gồm khoảng 4,6 triệu liên kết thuốc–thuốc–tác dụng phụ, bao phủ 645 loại thuốc và 1.317 tác dụng phụ."
            )
            console.print(source_block)
            print("=" * 150)
            print("\n\n")
    else:
        for msg in miss_msgs:
            print(msg)
        print("➡️ Sẽ bỏ qua bước dự đoán tác dụng phụ bằng mô hình và chuyển sang tra cứu DrugBank...\n\n\n")
        
    res = db.lookup_pair(
        queryA,
        queryB,
        stitchA=args.stitch_a,
        stitchB=args.stitch_b,
        debug=args.db_debug,
    )
    candA = res.get("candidatesA") or []
    candB = res.get("candidatesB") or []
    if len(candA) > 1:
        displayA_for_db = res.get("candidatesA_label") or db.format_candidates_label(candA)
    else:
        displayA_for_db = res.get("drugA_name") or res.get("candidatesA_label") or queryA
    if len(candB) > 1:
        displayB_for_db = res.get("candidatesB_label") or db.format_candidates_label(candB)
    else:
        displayB_for_db = res.get("drugB_name") or res.get("candidatesB_label") or queryB

    displayA_short = short_text(displayA_for_db, max_len=args.display_maxlen)
    displayB_short = short_text(displayB_for_db, max_len=args.display_maxlen)
    inputA_short = short_text(queryA, max_len=args.display_maxlen)
    inputB_short = short_text(queryB, max_len=args.display_maxlen)

    displayA_query = displayA_short
    displayB_query = displayB_short
    if inputA_short and displayA_short and displayA_short != inputA_short:
        displayA_query = f"{displayA_short} (aka {inputA_short})"
    if inputB_short and displayB_short and displayB_short != inputB_short:
        displayB_query = f"{displayB_short} (aka {inputB_short})"
    if args.db_debug:
        def _fmt_list(vals, max_items=5):
            vals = vals or []
            if len(vals) <= max_items:
                return str(vals)
            return f"{vals[:max_items]} ... (+{len(vals) - max_items} more)"

        print("\n[DrugBank debug]")
        print(f"- candidatesA: {_fmt_list(res.get('candidatesA'))}")
        print(f"- candidatesB: {_fmt_list(res.get('candidatesB'))}")
        print(f"- chosen_pair: {res.get('chosen_pair')}")
        print(f"- matched_pairs_count: {res.get('matched_pairs_count', 0)}")

    if getattr(args, "debug_drugbank", False):
        def _full_list(dbids):
            dbids = dbids or []
            out = []
            for d in dbids:
                nm = db.dbid_to_name(d) or "N/A"
                out.append(f"{d} ({nm})")
            return out

        print("\n[DrugBank candidates]")
        print("A:")
        for line in _full_list(res.get("candidatesA")):
            print(f"  - {line}")
        print("B:")
        for line in _full_list(res.get("candidatesB")):
            print(f"  - {line}")

    console = Console()
    db_table = Table(title="DrugBank – Tài liệu tham chiếu DDI được tuyển chọn")
    db_table.add_column("Field", justify="left")
    db_table.add_column("Value", overflow="fold")

    if not res.get("found", False):
        reason = res.get("reason", "")
        status = "Không tìm thấy tương tác trong DrugBank"
        db_table.add_row("Status", status)
        db_table.add_row("Query", f"A: {displayA_query}\nB: {displayB_query}")
        if reason:
            if reason == "name/cid not resolved":
                db_table.add_row("Reason", "DrugBank không resolve được tên/CID")
            elif reason == "no curated DDI found for resolved IDs":
                db_table.add_row("Reason", "DrugBank không có cảnh báo tương tác cho cặp này (curated DDI)")
            else:
                db_table.add_row("Reason", reason)
    else:
        if res.get("items"):
            db_table.add_row("Status", "Found")
        else:
            db_table.add_row("Status", "Không ghi nhận tương tác được tuyển chọn")
        db_table.add_row("DrugBank IDs", f"{res.get('drugA_dbid','?')}  <->  {res.get('drugB_dbid','?')}")
        db_table.add_row("Query", f"A: {displayA_query}\nB: {displayB_query}")

        seen = set()
        lines = []
        for it in res.get("items", []):
            desc = (it.get("description", "") or "").strip()
            if not desc:
                continue
            key = " ".join(desc.lower().split())
            if key in seen:
                continue
            seen.add(key)
            lines.append(desc)

        if len(lines) == 0:
            db_table.add_row("Curated warning(s)", "(No description available)")
        else:
            warn_out = []
            for desc in lines[:3]:
                if translator:
                    protected, mp = protect_drug_names(desc, displayA_short, displayB_short)
                    vi = translator.translate_one(protected)
                    vi = restore_drug_names(vi, mp)
                    warn_out.append(f"- {vi}\n  (EN: {desc})")
                else:
                    warn_out.append(f"- {desc}")

            db_table.add_row("Curated warning(s)", "\n".join(warn_out))
            
    console.print(db_table)

    drugbank_note = (
        "\n[Về DrugBank]\n"
        "- DrugBank là một cơ sở tri thức y sinh được biên tập và duy trì bởi các chuyên gia trong lĩnh vực.\n"
        "- Nền tảng này cung cấp các mô tả DDI theo định hướng lâm sàng (thường ngắn gọn), khác với các bộ dữ liệu được suy ra từ FAERS (ví dụ: TWOSIDES) "
        "vốn chứa nhiều tác dụng phụ được báo cáo ở mức chi tiết.\n"
        "- Trong bản demo này, DrugBank được sử dụng như một nguồn tham chiếu để đối chiếu với các tác dụng phụ của thuốc do mô hình dự đoán.\n"
    )
    console.print(drugbank_note)
    print("\n")


if __name__ == "__main__":
    main()
