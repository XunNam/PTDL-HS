import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DRUGBANK_XML = "./drugbank_all_full_database.xml/full_database.xml"          # <-- đổi path
OUT_JSON = "drugbank_ddi.json"

# DrugBank XML thường có namespace; ta lấy root và strip namespace khi cần
def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def main():
    xml_path = Path(DRUGBANK_XML)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # index: normalized drug name -> drugbank_id
    name_to_dbid = {}
    # ddi: (dbid1, dbid2) sorted tuple -> list[interaction]
    ddi = {}

    # iterate drugs
    for drug in root:
        if strip_ns(drug.tag) != "drug":
            continue

        # drugbank-id (primary)
        dbid = None
        for x in drug.findall(".//"):
            if strip_ns(x.tag) == "drugbank-id" and x.attrib.get("primary") == "true":
                dbid = (x.text or "").strip()
                break
        if not dbid:
            continue

        # name
        name_el = None
        for x in drug:
            if strip_ns(x.tag) == "name":
                name_el = x
                break
        drug_name = (name_el.text or "").strip() if name_el is not None else ""
        if drug_name:
            name_to_dbid[norm(drug_name)] = dbid

        # synonyms (tùy dataset; có thể nhiều)
        for syn in drug.findall(".//"):
            if strip_ns(syn.tag) == "synonym":
                s = (syn.text or "").strip()
                if s:
                    name_to_dbid[norm(s)] = dbid

        # drug-interactions
        for inter in drug.findall(".//"):
            if strip_ns(inter.tag) != "drug-interaction":
                continue

            # partner drugbank-id + name + description
            partner_id = None
            partner_name = ""
            desc = ""

            for ch in inter:
                t = strip_ns(ch.tag)
                if t == "drugbank-id":
                    partner_id = (ch.text or "").strip()
                elif t == "name":
                    partner_name = (ch.text or "").strip()
                elif t == "description":
                    desc = (ch.text or "").strip()

            if not partner_id:
                continue

            key = tuple(sorted([dbid, partner_id]))
            # --- DEDUPE theo description (tránh A->B và B->A bị trùng) ---
            desc_norm = norm(desc)
            if not desc_norm:
                continue

            # lưu thêm 1 set để check trùng
            seen = ddi.setdefault(f"{key[0]}|{key[1]}|_seen", set())
            if desc_norm in seen:
                continue
            seen.add(desc_norm)

            ddi.setdefault(key, []).append({
                "drugA_dbid": dbid,
                "drugB_dbid": partner_id,
                "drugB_name": partner_name,
                "description": desc,
            })

    # cleanup keys _seen (set không ghi được ra json)
    for k in list(ddi.keys()):
        if isinstance(k, str) and k.endswith("|_seen"):
            ddi.pop(k, None)

    out = {
        "name_to_dbid": name_to_dbid,
        "ddi": ddi,
    }

    # json không support tuple key → convert sang "DB0001|DB0002"
    ddi2 = {}
    for (a,b), items in ddi.items():
        ddi2[f"{a}|{b}"] = items
    out["ddi"] = ddi2

    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("✅ Saved:", OUT_JSON)
    print("name_to_dbid:", len(name_to_dbid))
    print("ddi pairs:", len(ddi2))

if __name__ == "__main__":
    main()
