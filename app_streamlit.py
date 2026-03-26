import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from drugbank_ref import DrugBankRef
from local_translator import LocalTranslator, protect_drug_names, restore_drug_names


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSLATOR_MODEL = "facebook/nllb-200-distilled-600M"


def resolve_path(path: str, base_dir: Path = SCRIPT_DIR) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def load_torch_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@st.cache_data
def load_drug_records(drug_map_csv: str, drug_name_map_csv: str):
    drug_map_csv = resolve_path(drug_map_csv)
    drug_name_map_csv = resolve_path(drug_name_map_csv)
    if os.path.exists(drug_name_map_csv):
        df = pd.read_csv(drug_name_map_csv)
        df["stitch_id"] = df["stitch_id"].astype(str)
        df["drug_id"] = df["drug_id"].astype(int)
        df["drug_name"] = df["drug_name"].fillna(df["stitch_id"]).astype(str)
    else:
        df = pd.read_csv(drug_map_csv)
        df["stitch_id"] = df["stitch_id"].astype(str)
        df["drug_id"] = df["drug_id"].astype(int)
        df["drug_name"] = df["stitch_id"]
    stitch2id = dict(zip(df["stitch_id"], df["drug_id"]))
    id2name = dict(zip(df["drug_id"], df["drug_name"]))
    stitch2name = dict(zip(df["stitch_id"], df["drug_name"]))
    records = df[["stitch_id", "drug_id", "drug_name"]].to_dict("records")
    return records, stitch2id, id2name, stitch2name


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
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]], axis=0)
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]], axis=0)
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)


def load_maps(drug_map_csv: str, se_map_csv: str, se_name_map_csv: str):
    drug_map = pd.read_csv(drug_map_csv)
    se_map = pd.read_csv(se_map_csv)
    stitch2id = dict(zip(drug_map["stitch_id"].astype(str), drug_map["drug_id"].astype(int)))
    id2stitch = dict(zip(drug_map["drug_id"].astype(int), drug_map["stitch_id"].astype(str)))
    se_id2cui = dict(zip(se_map["se_id"].astype(int), se_map["side_effect_cui"].astype(str)))
    if os.path.exists(se_name_map_csv):
        se_name = pd.read_csv(se_name_map_csv)
        se_id2name = dict(zip(se_name["se_id"].astype(int), se_name["side_effect_name"].astype(str)))
    else:
        se_id2name = {k: v for k, v in se_id2cui.items()}
    return stitch2id, id2stitch, se_id2name, se_id2cui


@st.cache_resource
def load_all(
    model_path: str,
    parquet_path: str,
    drug_map_csv: str,
    se_map_csv: str,
    se_name_map_csv: str,
    device_str: str,
):
    model_path = resolve_path(model_path)
    parquet_path = resolve_path(parquet_path)
    drug_map_csv = resolve_path(drug_map_csv)
    se_map_csv = resolve_path(se_map_csv)
    se_name_map_csv = resolve_path(se_name_map_csv)
    device = resolve_device(device_str)
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
    df = pd.read_parquet(parquet_path)
    edge_index_base = build_base_edge_index(df, num_nodes=num_nodes, device=device)
    stitch2id, id2stitch, se_id2name, se_id2cui = load_maps(drug_map_csv, se_map_csv, se_name_map_csv)
    with torch.no_grad():
        z = encoder(edge_index_base)
        rel = decoder.rel_emb.weight
    return {
        "device": device,
        "z": z,
        "rel": rel,
        "stitch2id": stitch2id,
        "id2stitch": id2stitch,
        "se_id2name": se_id2name,
        "se_id2cui": se_id2cui,
        "df": df,
    }


def predict_topk(z, rel, a_id: int, b_id: int, topk: int):
    za = z[a_id]
    zb = z[b_id]
    scores = (rel * (za * zb)).sum(dim=-1)
    probs = torch.sigmoid(scores)
    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    return idxs.detach().cpu().numpy(), vals.detach().cpu().numpy()


@st.cache_resource
def get_translator(model_name: str, device_str: str):
    translator = LocalTranslator(model_name=model_name, device=str(resolve_device(device_str)))
    return translator if translator.available else None


st.set_page_config(
    page_title="Decagon DDI Side-Effect Predictor",
    initial_sidebar_state="collapsed",
)
st.title("💊 Decagon – Dự đoán tác dụng phụ khi phối hợp 2 thuốc")
st.caption("⚠️ Đây là mô hình học từ tín hiệu báo cáo (TWOSIDES/Decagon). 'Risk Score' = sigmoid(logit) × 100%, mang tính tham khảo, không phải xác suất y khoa chuẩn.")

with st.sidebar:
    st.header("⚙️ Config")
    model_path = st.text_input("Model path", resolve_path("runs_decagon_full/final_model.pt"))
    parquet_path = st.text_input("Mapped parquet", resolve_path("decagon_processed/decagon_polypharmacy_mapped.parquet"))
    drug_map_csv = st.text_input("drug_id_map.csv", resolve_path("decagon_processed/drug_id_map.csv"))
    se_map_csv = st.text_input("side_effect_id_map.csv", resolve_path("decagon_processed/side_effect_id_map.csv"))
    se_name_map_csv = st.text_input("side_effect_id_name_map.csv", resolve_path("decagon_processed/side_effect_id_name_map.csv"))
    drug_name_map_csv = st.text_input("drug_id_name_map.csv", resolve_path("decagon_processed/drug_id_name_map.csv"))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = st.selectbox("Device", ["cuda", "cpu"], index=0 if default_device == "cuda" else 1)
    topk = st.slider("Top-K", 5, 50, 15, step=1)
    lang_choice = st.selectbox("Ngôn ngữ cảnh báo", ["Tiếng Anh", "Tiếng Việt"], index=1)

st.subheader("Chọn 2 thuốc")

records, stitch2id_light, id2name_light, stitch2name = load_drug_records(drug_map_csv, drug_name_map_csv)


def format_drug_option(stitch_id: str) -> str:
    drug_id = stitch2id_light.get(stitch_id, "-")
    name = stitch2name.get(stitch_id, "")
    name_part = f" {name}" if name and name != stitch_id else ""
    return f"{stitch_id} (id:{drug_id}){name_part}"


def display_drug_label(stitch_id: str) -> str:
    name = stitch2name.get(stitch_id, "")
    if name and name != stitch_id:
        return name
    return f"{stitch_id} (id:{stitch2id_light.get(stitch_id, '-')})"


def filter_records(query: str):
    query = (query or "").strip().lower()
    if not query:
        return records
    return [
        record
        for record in records
        if query in f"{record['drug_name']} {record['stitch_id']} {record['drug_id']}".lower()
    ]


col1, col2 = st.columns(2)

with col1:
    query_a = st.text_input("🔎 Search drug A (name/CID)", "aspirin")
    filtered_a = filter_records(query_a)
    if len(filtered_a) == 0:
        st.warning("Không tìm thấy drug A theo search. Thử từ khoá khác.")
        filtered_a = records[:50]
    options_a = [record["stitch_id"] for record in filtered_a]
    stitch_a = st.selectbox("Drug A", options_a, index=0, format_func=format_drug_option)

with col2:
    query_b = st.text_input("🔎 Search drug B (name/CID)", "ibuprofen")
    filtered_b = [record for record in filter_records(query_b) if record["stitch_id"] != stitch_a]
    if len(filtered_b) == 0:
        st.warning("Không tìm thấy drug B (hoặc bị loại do trùng drug A). Thử từ khoá khác.")
        filtered_b = [record for record in records[:50] if record["stitch_id"] != stitch_a]
    options_b = [record["stitch_id"] for record in filtered_b]
    stitch_b = st.selectbox("Drug B", options_b, index=0, format_func=format_drug_option)

if st.button("🔍 Dự đoán"):
    try:
        pack = load_all(model_path, parquet_path, drug_map_csv, se_map_csv, se_name_map_csv, device_str)
        if device_str == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA không khả dụng, ứng dụng đã chuyển sang CPU.")
        stitch2id = pack["stitch2id"]
        if stitch_a not in stitch2id or stitch_b not in stitch2id:
            st.error("❌ STITCH ID không có trong mapping. Hãy kiểm tra lại (đúng format CID...).")
        else:
            a_id = stitch2id[stitch_a]
            b_id = stitch2id[stitch_b]
            idxs, probs = predict_topk(pack["z"], pack["rel"], a_id, b_id, topk)
            translator = None
            if lang_choice == "Tiếng Việt":
                translator = get_translator(TRANSLATOR_MODEL, device_str)
                if translator is None:
                    st.warning("Không tìm thấy model dịch trong local cache, tiếp tục hiển thị tiếng Anh.")
            rows = []
            for sid, prob in zip(idxs, probs):
                rows.append(
                    {
                        "Rank": len(rows) + 1,
                        "Polypharmacy Side Effect ID": int(sid),
                        "Risk Score": f"{float(prob) * 100.0:.2f}%",
                        "Side Effect Name": pack["se_id2name"].get(int(sid), ""),
                        "CUI": pack["se_id2cui"].get(int(sid), ""),
                    }
                )
            out_df = pd.DataFrame(rows)
            a_label = display_drug_label(stitch_a)
            b_label = display_drug_label(stitch_b)
            st.success(f"✅ Input: {format_drug_option(stitch_a)} + {format_drug_option(stitch_b)}")
            st.markdown(f"**Top-{topk} predicted side effects**")
            st.markdown(f"Between {a_label} and {b_label}")
            st.dataframe(out_df, width="stretch")

            df = pack["df"]
            gt = df[
                ((df["drug1_id"] == a_id) & (df["drug2_id"] == b_id))
                | ((df["drug1_id"] == b_id) & (df["drug2_id"] == a_id))
            ]["se_id"]
            gt_set = set(gt.astype(int).tolist())
            pred_set = set(int(x) for x in idxs.tolist())
            inter = gt_set.intersection(pred_set)
            summary_df = pd.DataFrame(
                [
                    {"Metric": "Found in dataset", "Value": len(gt_set)},
                    {"Metric": f"Predicted correctly over Top {topk}", "Value": len(inter)},
                ]
            )
            st.subheader("Summary")
            st.table(summary_df)

            source_block = (
                "[Nguồn]\n"
                "- Nghiên cứu này sử dụng bộ dữ liệu tác dụng phụ dược phẩm từ TWOSIDES, ban đầu được giới thiệu trong khuôn khổ Decagon.\n"
                "- Bộ dữ liệu được xây dựng bằng cách khai thác Hệ thống Báo cáo Sự cố Bất lợi của FDA (FAERS) và chứa các tác dụng phụ được báo cáo liên quan đến các cặp thuốc.\n"
                "- Bộ dữ liệu bao gồm khoảng 4,6 triệu liên kết thuốc–thuốc–tác dụng phụ, bao phủ 645 loại thuốc và 1.317 tác dụng phụ."
            )
            st.info(source_block)

            db = DrugBankRef(resolve_path("drugbank_ddi.json"), index_path=resolve_path("cache/drugbank_index.json"))
            query_a_db = id2name_light.get(a_id, stitch_a)
            query_b_db = id2name_light.get(b_id, stitch_b)
            res = db.lookup_pair(query_a_db, query_b_db)
            db_rows = []
            if not res.get("found", False):
                db_rows.append({"Field": "Status", "Value": "Không tìm thấy tương tác trong DrugBank"})
                db_rows.append({"Field": "Query", "Value": f"A: {query_a_db}\nB: {query_b_db}"})
                reason = res.get("reason", "")
                if reason:
                    db_rows.append({"Field": "Reason", "Value": reason})
            else:
                db_rows.append({"Field": "Status", "Value": "Found"})
                db_rows.append({"Field": "DrugBank IDs", "Value": f"{res.get('drugA_dbid','?')}  <->  {res.get('drugB_dbid','?')}"})
                db_rows.append({"Field": "Query", "Value": f"A: {query_a_db}\nB: {query_b_db}"})
                seen = set()
                lines = []
                for item in res.get("items", []):
                    desc = (item.get("description", "") or "").strip()
                    if not desc:
                        continue
                    key = " ".join(desc.lower().split())
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(desc)
                if len(lines) == 0:
                    warn_text = "(No description available)"
                else:
                    warn_out = []
                    for desc in lines[:3]:
                        if translator:
                            protected, mapping = protect_drug_names(desc, query_a_db, query_b_db)
                            translated = translator.translate_one(protected)
                            translated = restore_drug_names(translated, mapping)
                            warn_out.append(f"- {translated}\n  (EN: {desc})")
                        else:
                            warn_out.append(f"- {desc}")
                    warn_text = "\n".join(warn_out)
                db_rows.append({"Field": "Curated warning(s)", "Value": warn_text})
            st.subheader("DrugBank – Tài liệu tham chiếu DDI được tuyển chọn")
            st.table(pd.DataFrame(db_rows))

            drugbank_note = (
                "[Về DrugBank]\n"
                "- DrugBank là một cơ sở tri thức y sinh được biên tập và duy trì bởi các chuyên gia trong lĩnh vực.\n"
                "- Nền tảng này cung cấp các mô tả DDI theo định hướng lâm sàng (thường ngắn gọn), khác với các bộ dữ liệu được suy ra từ FAERS (ví dụ: TWOSIDES) vốn chứa nhiều tác dụng phụ được báo cáo ở mức chi tiết.\n"
                "- Trong bản demo này, DrugBank được sử dụng như một nguồn tham chiếu để đối chiếu với các tác dụng phụ của thuốc do mô hình dự đoán.\n"
            )
            st.info(drugbank_note)
    except Exception as exc:
        st.error(str(exc))
