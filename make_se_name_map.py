import pandas as pd

# ====== đổi đúng đường dẫn của cậu ======
MAPPED_SE = "decagon_processed/side_effect_id_map.csv"     # se_id + CUI
RAW_CSVGZ = "ChChSe-Decagon_polypharmacy.csv.gz"           # file gốc có Side Effect Name
OUT_PATH  = "decagon_processed/side_effect_id_name_map.csv"
# =======================================

# 1) Load mapping se_id <-> CUI
se_map = pd.read_csv(MAPPED_SE)  # columns: side_effect_cui, se_id
se_map["side_effect_cui"] = se_map["side_effect_cui"].astype(str)
se_map["se_id"] = se_map["se_id"].astype(int)

# 2) Load only 2 columns from raw file (nhẹ)
raw = pd.read_csv(
    RAW_CSVGZ,
    usecols=["Polypharmacy Side Effect", "Side Effect Name"],
    dtype=str
)
raw.columns = ["poly_cui", "se_name"]

# 3) Deduplicate by CUI (cùng CUI thì name thường giống, lấy cái đầu)
raw = raw.dropna()
raw = raw.drop_duplicates(subset=["poly_cui"], keep="first")

# 4) Join to create se_id -> English name
merged = se_map.merge(raw, left_on="side_effect_cui", right_on="poly_cui", how="left")
out = merged[["se_id", "side_effect_cui", "se_name"]].rename(columns={"se_name": "side_effect_name"})

# 5) Save
out.to_csv(OUT_PATH, index=False)
print("✅ Saved:", OUT_PATH)
print(out.head())
print("Missing names:", out["side_effect_name"].isna().sum(), "/", len(out))
