import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[.,;:/\[\]\{\}\(\)\"'`]+")
_PAREN_RE = re.compile(r"\([^)]*\)")
_CID_RE = re.compile(r"CID0*(\d+)", re.IGNORECASE)

_SALT_SUFFIXES = {
    "sulfate",
    "sulphate",
    "hydrochloride",
    "hcl",
    "phosphate",
    "acetate",
    "citrate",
    "maleate",
    "tartrate",
    "succinate",
    "nitrate",
    "chloride",
}

_FORM_SUFFIXES = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "injection",
    "injections",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def _legacy_norm(s: str) -> str:
    return _SPACE_RE.sub(" ", (s or "").strip().lower())


def _strip_parens(s: str) -> str:
    return _PAREN_RE.sub(" ", s)


def normalize_name(
    s: str,
    drop_salt_suffix: bool = True,
    drop_form_suffix: bool = True,
    strip_parens: bool = False,
) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    if strip_parens:
        s = _strip_parens(s)
    s = s.replace("-", " ").replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    if not s:
        return ""
    parts = s.split()
    if drop_salt_suffix and parts and parts[-1] in _SALT_SUFFIXES:
        parts = parts[:-1]
    if drop_form_suffix and parts and parts[-1] in _FORM_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip()


def _candidate_keys(query: Optional[str]) -> List[str]:
    if not query:
        return []
    variants = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]
    out: List[str] = []
    seen = set()
    for drop_salt, drop_form, strip_parens in variants:
        key = normalize_name(
            query,
            drop_salt_suffix=drop_salt,
            drop_form_suffix=drop_form,
            strip_parens=strip_parens,
        )
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def normalize_key(s: str) -> str:
    return normalize_name(s, drop_salt_suffix=True, drop_form_suffix=True, strip_parens=False)


def _pair_stats(items: Iterable[Dict]) -> Tuple[int, int, int]:
    seen = set()
    total_len = 0
    count = 0
    for it in items:
        count += 1
        desc = (it.get("description", "") or "").strip()
        if not desc:
            continue
        key = " ".join(desc.lower().split())
        if key in seen:
            continue
        seen.add(key)
        total_len += len(desc)
    return count, total_len, len(seen)


class DrugBankRef:
    def __init__(self, json_path: str = "drugbank_ddi.json", index_path: str = "cache/drugbank_index.json"):
        self.json_path = self._resolve_path(json_path)
        self.index_path = self._resolve_path(index_path) if index_path else None
        self._index: Optional[Dict] = None
        self.data = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.name_to_dbid = self.data.get("name_to_dbid", {})
        self.ddi = self.data.get("ddi", {})

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return (SCRIPT_DIR / p).resolve()

    def _load_index(self) -> Dict:
        if self._index is not None:
            return self._index
        if self.index_path and self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self._index = {}
        return self._index

    def dbid_to_name(self, dbid: str) -> Optional[str]:
        if not dbid:
            return None
        index = self._load_index()
        id_to_name = index.get("id_to_name", {}) if isinstance(index, dict) else {}
        return id_to_name.get(dbid)

    def format_dbids_short(self, dbids: List[str], max_items: int = 2) -> str:
        names = []
        seen = set()
        for dbid in dbids or []:
            nm = self.dbid_to_name(dbid) or dbid
            if nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
        if not names:
            return ""
        if len(names) <= max_items:
            return ", ".join(names)
        return ", ".join(names[:max_items]) + f" (+{len(names) - max_items})"

    def format_candidates_label(self, dbids: List[str]) -> str:
        if not dbids:
            return ""
        first = dbids[0]
        base = self.dbid_to_name(first) or first
        extra = max(0, len(dbids) - 1)
        if extra:
            return f"{base} (+{extra})"
        return base

    def _extract_cid(self, stitch_or_cid: Optional[str]) -> Optional[str]:
        if not stitch_or_cid:
            return None
        s = str(stitch_or_cid).strip()
        if not s:
            return None
        m = _CID_RE.search(s)
        if m:
            return str(int(m.group(1)))
        if s.isdigit():
            return str(int(s))
        return None

    def _debug_enabled(self, debug: bool) -> bool:
        if debug:
            return True
        flag = os.getenv("DRUGBANK_DEBUG", "").strip().lower()
        return flag in {"1", "true", "yes", "y"}

    def resolve_candidates(self, query: Optional[str], stitch_or_cid: Optional[str] = None) -> List[str]:
        cid = self._extract_cid(stitch_or_cid)
        keys = _candidate_keys(query)
        if not cid and not keys:
            return []

        index = self._load_index()
        name_to_ids = index.get("name_to_ids", {}) if isinstance(index, dict) else {}
        cid_to_ids = index.get("cid_to_ids", {}) if isinstance(index, dict) else {}

        out: List[str] = []
        seen = set()

        if cid:
            for dbid in cid_to_ids.get(cid, []):
                if dbid not in seen:
                    out.append(dbid)
                    seen.add(dbid)

        for key in keys:
            for dbid in name_to_ids.get(key, []):
                if dbid not in seen:
                    out.append(dbid)
                    seen.add(dbid)

        if query:
            legacy_key = _legacy_norm(query)
            legacy = self.name_to_dbid.get(legacy_key)
            if legacy and legacy not in seen:
                out.append(legacy)
                seen.add(legacy)

        return out

    def _resolve_ids(self, query: Optional[str], stitch_id: Optional[str] = None) -> List[str]:
        return self.resolve_candidates(query, stitch_or_cid=stitch_id)

    def resolve_dbid(self, drug_name: Optional[str], stitch_or_cid: Optional[str] = None) -> Optional[str]:
        ids = self.resolve_candidates(drug_name, stitch_or_cid)
        return ids[0] if ids else None

    def lookup_pair(
        self,
        name_a: str,
        name_b: str,
        stitchA: Optional[str] = None,
        stitchB: Optional[str] = None,
        debug: bool = False,
    ):
        debug_on = self._debug_enabled(debug)
        ids_a = self.resolve_candidates(name_a, stitchA)
        ids_b = self.resolve_candidates(name_b, stitchB)
        cand_a_label = self.format_candidates_label(ids_a)
        cand_b_label = self.format_candidates_label(ids_b)

        if not ids_a or not ids_b:
            res = {
                "found": False,
                "reason": "name/cid not resolved",
                "items": [],
                "candidatesA": ids_a,
                "candidatesB": ids_b,
                "candidatesA_label": cand_a_label,
                "candidatesB_label": cand_b_label,
            }
            if debug_on:
                cid_a = self._extract_cid(stitchA)
                cid_b = self._extract_cid(stitchB)
                print(
                    f"[DrugBankRef] resolve failed: candidatesA={ids_a[:5]} candidatesB={ids_b[:5]} "
                    f"cidA={cid_a} cidB={cid_b}",
                    file=sys.stderr,
                )
                res["chosen_pair"] = None
                res["matched_pairs_count"] = 0
            return res

        matched = []
        all_items: List[Dict] = []
        matched_pairs_count = 0
        for ida in ids_a:
            for idb in ids_b:
                key = "|".join(sorted([ida, idb]))
                items = self.ddi.get(key, [])
                if items:
                    stats = _pair_stats(items)
                    matched.append((ida, idb, items, stats))
                    all_items.extend(items)
                    matched_pairs_count += 1

        if not matched:
            res = {
                "found": False,
                "reason": "no curated DDI found for resolved IDs",
                "items": [],
                "candidatesA": ids_a,
                "candidatesB": ids_b,
                "candidatesA_label": cand_a_label,
                "candidatesB_label": cand_b_label,
            }
            if debug_on:
                cid_a = self._extract_cid(stitchA)
                cid_b = self._extract_cid(stitchB)
                print(
                    f"[DrugBankRef] no DDI found: candidatesA={ids_a[:5]} candidatesB={ids_b[:5]} "
                    f"cidA={cid_a} cidB={cid_b}",
                    file=sys.stderr,
                )
                res["chosen_pair"] = None
                res["matched_pairs_count"] = 0
            return res

        best_a, best_b, best_items, best_stats = matched[0]
        for ida, idb, items, stats in matched[1:]:
            if stats > best_stats:
                best_a, best_b, best_items, best_stats = ida, idb, items, stats

        seen_desc = set()
        deduped: List[Dict] = []
        for it in all_items:
            desc = (it.get("description", "") or "").strip()
            key = " ".join(desc.lower().split())
            if key and key in seen_desc:
                continue
            if key:
                seen_desc.add(key)
            deduped.append(it)

        res = {
            "found": True,
            "drugA_dbid": best_a,
            "drugB_dbid": best_b,
            "drugA_name": self.dbid_to_name(best_a),
            "drugB_name": self.dbid_to_name(best_b),
            "items": deduped,
            "candidatesA": ids_a,
            "candidatesB": ids_b,
            "candidatesA_label": cand_a_label,
            "candidatesB_label": cand_b_label,
        }
        if debug_on:
            res["chosen_pair"] = (best_a, best_b)
            res["matched_pairs_count"] = matched_pairs_count
        return res
