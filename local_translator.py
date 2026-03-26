from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
import re
import warnings

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


_FRENCH_CHARS_RE = re.compile(r"[àâäçéèêëîïôöùûüÿœ]", re.IGNORECASE)
_FRENCH_WORDS_RE = re.compile(
    r"\b(la|le|les|des|de|du|une|un|est|sont|maladie|thérapie|arthérosclérose|sténose|sinusite|protéinosis|agénèse)\b",
    re.IGNORECASE,
)


def _looks_french(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        _FRENCH_CHARS_RE.search(t)
        or _FRENCH_WORDS_RE.search(t)
        or "l'" in t
        or "d'" in t
    )


@dataclass
class LocalTranslator:
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    batch_size: int = 16
    src_lang: str = "eng_Latn"
    tgt_lang: str = "vie_Latn"
    quiet: bool = True
    auto_src_lang: bool = True
    offline_only: bool = True
    available: bool = field(init=False, default=False)
    load_error: Optional[str] = field(init=False, default=None)
    tokenizer: Optional[AutoTokenizer] = field(init=False, default=None, repr=False)
    model: Optional[AutoModelForSeq2SeqLM] = field(init=False, default=None, repr=False)
    _warning_emitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        os.environ["HF_HUB_OFFLINE"] = "1" if self.offline_only else "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "1" if self.offline_only else "0"
        if self.quiet:
            try:
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            except Exception:
                pass
            try:
                from huggingface_hub import logging as hub_logging

                hub_logging.set_verbosity_error()
            except Exception:
                pass
            warnings.filterwarnings(
                "ignore",
                message=r".*tied weights mapping.*|.*tie_word_embeddings=False.*",
            )
        self.cache: Dict[Tuple[str, str, str], str] = {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                local_files_only=self.offline_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                local_files_only=self.offline_only,
            ).to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:
            self.available = False
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
        self.is_nllb = bool(
            self.available
            and self.tokenizer is not None
            and ("nllb" in self.model_name.lower() or hasattr(self.tokenizer, "lang_code_to_id"))
        )
        if self.is_nllb and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.src_lang

    def warning_message(self) -> str:
        if self.offline_only:
            base = "Không tìm thấy model dịch trong local cache; tiếp tục hiển thị tiếng Anh."
        else:
            base = "Không tải được model dịch; tiếp tục hiển thị tiếng Anh."
        if not self.load_error:
            return base
        return f"{base} Chi tiết: {self.load_error}"

    def _warn_unavailable(self) -> None:
        if self.available or self._warning_emitted:
            return
        warnings.warn(self.warning_message())
        self._warning_emitted = True

    def _set_src_lang(self, lang: str):
        self.src_lang = lang
        if self.is_nllb and self.tokenizer is not None and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = lang

    def _forced_bos_id(self) -> Optional[int]:
        if not self.is_nllb or self.tokenizer is None:
            return None
        if hasattr(self.tokenizer, "lang_code_to_id") and isinstance(getattr(self.tokenizer, "lang_code_to_id"), dict):
            mp = getattr(self.tokenizer, "lang_code_to_id")
            if self.tgt_lang in mp:
                return mp[self.tgt_lang]
        try:
            tid = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
            if isinstance(tid, int) and tid >= 0:
                return tid
        except Exception:
            pass
        return None

    def _gen_kwargs(self):
        forced_bos = self._forced_bos_id()
        return {"forced_bos_token_id": forced_bos} if forced_bos is not None else {}

    def _guess_src_lang(self, text: str) -> str:
        return "fra_Latn" if _looks_french(text) else "eng_Latn"

    def translate_one(self, text: str, src_lang: Optional[str] = None) -> str:
        text = (text or "").strip()
        if not text:
            return text
        if not self.available or self.tokenizer is None or self.model is None:
            self._warn_unavailable()
            return text
        lang = src_lang
        if lang is None and self.auto_src_lang:
            lang = self._guess_src_lang(text)
        if lang:
            self._set_src_lang(lang)
        key = (self.src_lang, self.tgt_lang, text)
        if key in self.cache:
            return self.cache[key]
        try:
            with torch.inference_mode():
                inp = self.tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(self.device)
                out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, **self._gen_kwargs())
                vi = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            self.cache[key] = text
            return text
        self.cache[key] = vi
        return vi

    def translate_many(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        src_lang: Optional[str] = None,
    ) -> List[str]:
        bs = batch_size or self.batch_size
        norm_texts = [(t or "").strip() for t in texts]
        out: List[Optional[str]] = [None] * len(norm_texts)
        if not self.available or self.tokenizer is None or self.model is None:
            self._warn_unavailable()
            return norm_texts
        groups: Dict[str, List[int]] = {}
        for i, text in enumerate(norm_texts):
            if not text:
                out[i] = text
                continue
            lang = src_lang
            if lang is None and self.auto_src_lang:
                lang = self._guess_src_lang(text)
            lang = lang or self.src_lang
            key = (lang, self.tgt_lang, text)
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                groups.setdefault(lang, []).append(i)
        for lang, indices in groups.items():
            self._set_src_lang(lang)
            to_run = [norm_texts[i] for i in indices]
            for j in range(0, len(to_run), bs):
                batch = to_run[j : j + bs]
                try:
                    with torch.inference_mode():
                        inp = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(self.device)
                        gen = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, **self._gen_kwargs())
                        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
                except Exception:
                    decoded = batch
                for src_text, vi in zip(batch, decoded):
                    self.cache[(lang, self.tgt_lang, src_text)] = vi
            for i in indices:
                out[i] = self.cache[(lang, self.tgt_lang, norm_texts[i])]
        return [x if x is not None else "" for x in out]


def protect_drug_names(text: str, drugA: str, drugB: str):
    if not text:
        return text, {}
    mapping = {
        "__DRUGA__": drugA,
        "__DRUGB__": drugB,
    }

    def repl(src: str, token: str, value: str) -> str:
        if not src:
            return value
        pat = re.compile(re.escape(src), flags=re.IGNORECASE)
        return pat.sub(token, value)

    value = text
    value = repl(drugA, "__DRUGA__", value)
    value = repl(drugB, "__DRUGB__", value)
    return value, mapping


def restore_drug_names(text: str, mapping: Dict[str, str]):
    if not text:
        return text
    for token, name in mapping.items():
        text = text.replace(token, name)
    return text
