import time
import pandas as pd
import requests

DRUG_MAP_IN = "decagon_processed/drug_id_map.csv"   # stitch_id, drug_id
OUT_PATH    = "decagon_processed/drug_id_name_map.csv"

SLEEP_SEC = 0.2   # giảm tải PubChem, tránh bị rate-limit

drug_map = pd.read_csv(DRUG_MAP_IN)
drug_map["stitch_id"] = drug_map["stitch_id"].astype(str)
drug_map["drug_id"] = drug_map["drug_id"].astype(int)

names = []
cids = []

for i, stitch in enumerate(drug_map["stitch_id"].tolist(), start=1):
    # stitch dạng: CID000002173 -> cid = 2173
    cid = int(stitch.replace("CID", ""))
    cids.append(cid)

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
        title = js["PropertyTable"]["Properties"][0]["Title"]
    except Exception:
        title = None

    names.append(title)

    if i % 50 == 0:
        print(f"Fetched {i}/{len(drug_map)} ...")
    time.sleep(SLEEP_SEC)

out = drug_map.copy()
out["pubchem_cid"] = cids
out["drug_name"] = names

out.to_csv(OUT_PATH, index=False)
print("✅ Saved:", OUT_PATH)
print("Missing names:", out["drug_name"].isna().sum(), "/", len(out))
print(out.head())
