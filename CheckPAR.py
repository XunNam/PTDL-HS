import pandas as pd
df = pd.read_parquet("decagon_processed/decagon_polypharmacy_mapped.parquet")

all_ids = pd.concat([df["drug1_id"], df["drug2_id"]]).unique()
print("Num unique drug IDs used in edges:", len(all_ids))
print("Min/Max used:", all_ids.min(), all_ids.max())

missing = set(range(all_ids.max()+1)) - set(all_ids)
print("Missing IDs in [0..max] count:", len(missing))
print("Sample missing:", sorted(list(missing))[:20])
