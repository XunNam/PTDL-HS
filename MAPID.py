import os
import pandas as pd

# ========= CONFIG =========
INPUT_PATH = "ChChSe-Decagon_polypharmacy.csv.gz"   # <-- đổi theo file của cậu
OUT_DIR = "decagon_processed"
# Nếu muốn test nhanh trước khi chạy full 4.6M dòng:
DEBUG_NROWS = None   # ví dụ: 200000  (200k dòng). Để None là chạy full.
SAVE_FORMAT = "parquet"  # "parquet" (khuyên) hoặc "csvgz"
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Read data
df = pd.read_csv(INPUT_PATH, nrows=DEBUG_NROWS)
# Chuẩn hoá tên cột: strip spaces, thay nhiều khoảng trắng thành 1 khoảng trắng
df.columns = [" ".join(c.strip().split()) for c in df.columns]

# 2) Detect columns (tương thích nếu tên cột hơi khác)
# Bạn đang có: "# STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"
col_drug1 = None
col_drug2 = None
col_se = None

for c in df.columns:
    lc = c.lower()
    if "stitch" in lc and "1" in lc:
        col_drug1 = c
    elif "stitch" in lc and "2" in lc:
        col_drug2 = c
    elif "polypharmacy" in lc and "side effect" in lc:
        col_se = c

if col_drug1 is None or col_drug2 is None or col_se is None:
    raise ValueError(f"Không tìm thấy cột phù hợp. Columns hiện có: {df.columns.tolist()}")

# 3) Build ID maps
# Lưu ý: concat 2 cột drug rồi unique
all_drugs = pd.Index(pd.concat([df[col_drug1], df[col_drug2]], ignore_index=True).astype(str).unique())
drug2id = pd.Series(range(len(all_drugs)), index=all_drugs)

all_ses = pd.Index(df[col_se].astype(str).unique())
se2id = pd.Series(range(len(all_ses)), index=all_ses)

# 4) Apply mapping
df_mapped = pd.DataFrame({
    "drug1_id": df[col_drug1].astype(str).map(drug2id).astype("int32"),
    "drug2_id": df[col_drug2].astype(str).map(drug2id).astype("int32"),
    "se_id": df[col_se].astype(str).map(se2id).astype("int32"),
})

# 5) Save mapping tables (để sau này decode/giải thích)
drug_map = pd.DataFrame({"stitch_id": all_drugs.values, "drug_id": drug2id.values})
se_map = pd.DataFrame({"side_effect_cui": all_ses.values, "se_id": se2id.values})

drug_map_path = os.path.join(OUT_DIR, "drug_id_map.csv")
se_map_path = os.path.join(OUT_DIR, "side_effect_id_map.csv")
drug_map.to_csv(drug_map_path, index=False)
se_map.to_csv(se_map_path, index=False)

# 6) Save mapped edges
if SAVE_FORMAT == "parquet":
    out_edges_path = os.path.join(OUT_DIR, "decagon_polypharmacy_mapped.parquet")
    df_mapped.to_parquet(out_edges_path, index=False)
elif SAVE_FORMAT == "csvgz":
    out_edges_path = os.path.join(OUT_DIR, "decagon_polypharmacy_mapped.csv.gz")
    df_mapped.to_csv(out_edges_path, index=False, compression="gzip")
else:
    raise ValueError("SAVE_FORMAT chỉ nhận 'parquet' hoặc 'csvgz'")

print("✅ Done!")
print("Rows:", len(df_mapped))
print("Num drugs:", len(all_drugs))
print("Num side effects:", len(all_ses))
print("Saved:")
print("-", out_edges_path)
print("-", drug_map_path)
print("-", se_map_path)
