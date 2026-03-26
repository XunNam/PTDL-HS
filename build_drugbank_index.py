#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from collections import defaultdict

import xml.etree.ElementTree as ET


# -----------------------------
# Normalization
# -----------------------------
_DOSE_TOKENS = {
    "mg", "g", "mcg", "ug", "µg", "kg",
    "ml", "l",
    "iu", "u", "units",
    "%", "percent",
    "xr", "sr", "er", "cr", "dr",
    "tab", "tabs", "tablet", "tablets",
    "cap", "caps", "capsule", "capsules",
    "susp", "suspension",
    "inj", "injection",
    "iv", "im", "po",
}

_SALT_TOKENS = {
    "hydrochloride", "hcl",
    "sulfate", "sulphate",
    "sodium", "potassium", "calcium", "magnesium",
    "phosphate", "acetate", "tartrate", "citrate", "maleate",
    "mesylate", "besylate", "fumarate", "succinate", "nitrate",
    "hydrobromide", "bromide",
    "chloride",
    "oxalate", "lactate",
}

_punct_re = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_space_re = re.compile(r"\s+")

def normalize_name(s: str, drop_salts: bool = True) -> str:
    if not s:
        return ""
    s = s.strip().lower()

    # replace common separators with space
    s = s.replace("/", " ").replace("-", " ")
    s = _punct_re.sub(" ", s)
    s = _space_re.sub(" ", s).strip()

    if not s:
        return ""

    parts = s.split()
    cleaned = []
    for p in parts:
        if p in _DOSE_TOKENS:
            continue
        if drop_salts and p in _SALT_TOKENS:
            continue
        # drop pure numbers like 500, 10
        if p.isdigit():
            continue
        cleaned.append(p)

    s2 = " ".join(cleaned).strip()
    return s2


# -----------------------------
# XML helpers
# -----------------------------
def strip_ns(tag: str) -> str:
    # "{namespace}tag" -> "tag"
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def child_text(elem, wanted: str):
    for ch in list(elem):
        if strip_ns(ch.tag) == wanted:
            if ch.text:
                return ch.text.strip()
            return ""
    return ""

def iter_texts(elem, path: list):
    """
    Very small path-walker:
      path=["synonyms","synonym"] yields all <synonym> text under <synonyms>
    """
    cur = [elem]
    for level in path:
        nxt = []
        for e in cur:
            for ch in list(e):
                if strip_ns(ch.tag) == level:
                    nxt.append(ch)
        cur = nxt
        if not cur:
            return []
    out = []
    for e in cur:
        if e.text and e.text.strip():
            out.append(e.text.strip())
    return out

def extract_pubchem_cids(drug_elem) -> list:
    """
    Look for:
      <external-identifiers>
        <external-identifier>
          <resource>PubChem Compound</resource>
          <identifier>3715</identifier>
    """
    cids = []
    for ext_ids in list(drug_elem):
        if strip_ns(ext_ids.tag) != "external-identifiers":
            continue
        for ext in list(ext_ids):
            if strip_ns(ext.tag) != "external-identifier":
                continue
            resource = child_text(ext, "resource")
            if resource.strip().lower() in {"pubchem compound", "pubchem compound id"}:
                ident = child_text(ext, "identifier")
                ident = ident.strip()
                if ident.isdigit():
                    cids.append(int(ident))
    return cids


def build_index_from_xml(xml_path: str,
                         drop_salts: bool = True,
                         include_brands: bool = True,
                         include_products: bool = False,
                         max_keys_per_drug: int = 200):

    t0 = time.time()
    cid_to_ids = defaultdict(set)     # int CID -> {DBID,...}
    name_to_ids = defaultdict(set)    # normalized name -> {DBID,...}
    id_to_name = {}                  # DBID -> canonical name

    # streaming parse to avoid RAM blowup
    ctx = ET.iterparse(xml_path, events=("end",))
    drug_count = 0
    key_count = 0

    for event, elem in ctx:
        if strip_ns(elem.tag) != "drug":
            continue

        # --- DrugBank primary ID ---
        dbid = None
        for ch in list(elem):
            if strip_ns(ch.tag) == "drugbank-id":
                primary = ch.attrib.get("primary", "").lower() == "true"
                if ch.text and ch.text.strip():
                    if primary:
                        dbid = ch.text.strip()
                        break
                    if dbid is None:
                        dbid = ch.text.strip()
        if not dbid:
            elem.clear()
            continue

        # --- Canonical name ---
        name = child_text(elem, "name")
        if name:
            id_to_name[dbid] = name

        # --- Collect keys (name + synonyms + brands/products) ---
        keys = set()
        if name:
            keys.add(normalize_name(name, drop_salts=drop_salts))

        # synonyms
        for s in iter_texts(elem, ["synonyms", "synonym"]):
            keys.add(normalize_name(s, drop_salts=drop_salts))

        # international brands (optional)
        if include_brands:
            # path: international-brands -> international-brand -> name
            # our iter_texts only handles fixed tags, so do a small manual walk
            for ibs in list(elem):
                if strip_ns(ibs.tag) != "international-brands":
                    continue
                for ib in list(ibs):
                    if strip_ns(ib.tag) != "international-brand":
                        continue
                    nm = child_text(ib, "name")
                    if nm:
                        keys.add(normalize_name(nm, drop_salts=drop_salts))

        # products (optional, may explode)
        if include_products:
            for prods in list(elem):
                if strip_ns(prods.tag) != "products":
                    continue
                for p in list(prods):
                    if strip_ns(p.tag) != "product":
                        continue
                    nm = child_text(p, "name")
                    if nm:
                        keys.add(normalize_name(nm, drop_salts=drop_salts))

        # remove empties + cap
        keys = [k for k in keys if k]
        if len(keys) > max_keys_per_drug:
            keys = keys[:max_keys_per_drug]

        for k in keys:
            name_to_ids[k].add(dbid)
        key_count += len(keys)

        # --- PubChem CID ---
        for cid in extract_pubchem_cids(elem):
            cid_to_ids[cid].add(dbid)

        drug_count += 1
        if drug_count % 500 == 0:
            print(f"… parsed {drug_count} drugs | keys≈{key_count:,} | cids={len(cid_to_ids):,}")

        # IMPORTANT: clear element to free memory
        elem.clear()

    dt = time.time() - t0
    print(f"✅ Done parsing XML: drugs={drug_count:,}, keys≈{key_count:,}, unique_cids={len(cid_to_ids):,} in {dt:.1f}s")

    # convert sets -> sorted lists, and int keys -> str for JSON
    cid_to_ids_json = {str(cid): sorted(list(v)) for cid, v in cid_to_ids.items()}
    name_to_ids_json = {k: sorted(list(v)) for k, v in name_to_ids.items()}

    return {
        "meta": {
            "source": os.path.basename(xml_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drop_salts": drop_salts,
            "include_brands": include_brands,
            "include_products": include_products,
            "max_keys_per_drug": max_keys_per_drug,
        },
        "cid_to_ids": cid_to_ids_json,
        "name_to_ids": name_to_ids_json,
        "id_to_name": id_to_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="drugbank_all_full_database.xml/full_database.xml")
    ap.add_argument("--out", default="cache/drugbank_index.json")
    ap.add_argument("--drop-salts", action="store_true", help="Remove common salt words (hydrochloride, sodium, ...)")
    ap.add_argument("--keep-salts", action="store_true", help="Do NOT remove salt words")
    ap.add_argument("--include-brands", action="store_true", help="Index international brand names")
    ap.add_argument("--include-products", action="store_true", help="Index product names (can be huge)")
    ap.add_argument("--max-keys-per-drug", type=int, default=200)
    args = ap.parse_args()

    xml_path = args.xml
    if not os.path.exists(xml_path):
        raise SystemExit(f"❌ XML not found: {xml_path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    drop_salts = True
    if args.keep_salts:
        drop_salts = False
    elif args.drop_salts:
        drop_salts = True

    index = build_index_from_xml(
        xml_path=xml_path,
        drop_salts=drop_salts,
        include_brands=args.include_brands,
        include_products=args.include_products,
        max_keys_per_drug=args.max_keys_per_drug,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))

    print(f"✅ Saved index -> {args.out}")
    print(f"   name_keys={len(index['name_to_ids']):,} | cid_keys={len(index['cid_to_ids']):,} | drugs={len(index['id_to_name']):,}")


if __name__ == "__main__":
    main()
