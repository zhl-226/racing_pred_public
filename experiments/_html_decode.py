"""modules/html_decode.py — robust auto-detect decoder for Japanese sites.

Handles the encodings netkeiba and JRA.jp have served over time:
  - UTF-8 (netkeiba as of 2026-11; most modern pages)
  - EUC-JP (netkeiba historically)
  - Shift-JIS / cp932 (JRA.jp pages, some legacy netkeiba)
  - ISO-2022-JP (rare, email-era)

Detection order (strongest signal first):
  1. HTTP Content-Type charset= hint (passed in)
  2. HTML <meta charset="..."> or <meta http-equiv="Content-Type" ...>
  3. UTF-8 BOM
  4. Heuristic: try each candidate, pick the one that produces the most
     Japanese characters (hiragana/katakana/kanji) without replacement chars.

This replaces the hardcoded `.decode("euc-jp", errors="replace")` that
produced mojibake like "2026綛�11���13��ャ" (= UTF-8 bytes decoded as
EUC-JP; actual text was "2026年11月13日").
"""
from __future__ import annotations

import re


# Order matters: try strict decoders first (no fallback) so we can
# distinguish "clean" from "mostly-replacement-chars"
CANDIDATE_ENCODINGS = ("utf-8", "euc_jp", "shift_jis", "cp932", "iso-2022-jp")

_META_CHARSET_RE = re.compile(
    rb'<meta[^>]*charset\s*=\s*["\']?\s*([a-zA-Z0-9_\-]+)',
    re.IGNORECASE,
)
_META_HTTPEQUIV_RE = re.compile(
    rb'<meta[^>]*http-equiv\s*=\s*["\']?content-type[^>]*content\s*=\s*'
    rb'["\'][^"\']*charset\s*=\s*([a-zA-Z0-9_\-]+)',
    re.IGNORECASE,
)

# Japanese-script Unicode ranges: hiragana, katakana, half-width katakana,
# CJK unified ideographs, CJK compat ideographs
_JA_RE = re.compile(
    r'[぀-ゟ゠-ヿ･-ﾟ一-鿿豈-﫿]'
)


def _normalise(enc: str) -> str:
    enc = enc.lower().replace('_', '-').replace(' ', '')
    aliases = {
        'shift-jis': 'shift_jis',
        'shift-js':  'shift_jis',
        'sjis':      'shift_jis',
        'euc-jp':    'euc_jp',
        'eucjp':     'euc_jp',
        'iso2022jp': 'iso-2022-jp',
        'utf8':      'utf-8',
    }
    return aliases.get(enc, enc)


def _score(text: str, body_len: int) -> float:
    """Higher = better decode. Penalises replacement chars; rewards Japanese."""
    if not text:
        return -1.0
    ja_count = len(_JA_RE.findall(text))
    repl_count = text.count('�')
    # Normalise by body length so long pages aren't automatically better
    return (ja_count - 10 * repl_count) / max(body_len / 1024, 1.0)


def detect_encoding(body: bytes, content_type: str | None = None) -> str:
    """Return the best-guess encoding name for a bytestring."""
    # 1. BOM
    if body.startswith(b'\xef\xbb\xbf'):
        return 'utf-8'

    # 2. HTTP Content-Type
    if content_type:
        m = re.search(r'charset\s*=\s*([a-zA-Z0-9_\-]+)', content_type, re.IGNORECASE)
        if m:
            return _normalise(m.group(1))

    # 3. Meta tags (scan first 4KB; charset is near the top)
    head = body[:4096]
    m = _META_CHARSET_RE.search(head) or _META_HTTPEQUIV_RE.search(head)
    if m:
        return _normalise(m.group(1).decode('ascii', errors='ignore'))

    # 4. Score each candidate
    best_enc, best_score = 'utf-8', -1e9
    body_len = len(body)
    for enc in CANDIDATE_ENCODINGS:
        try:
            text = body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            # Also try with replace to still score partial matches
            try:
                text = body.decode(enc, errors='replace')
            except Exception:
                continue
        score = _score(text, body_len)
        if score > best_score:
            best_score, best_enc = score, enc
    return best_enc


def decode_html(body: bytes, content_type: str | None = None,
                fallback_encoding: str = 'utf-8') -> str:
    """Decode an HTML byte-string with auto-detected encoding.

    Always returns a str; on total failure falls back to decoding
    ``fallback_encoding`` with ``errors='replace'``.
    """
    if not body:
        return ''
    enc = detect_encoding(body, content_type)
    try:
        return body.decode(enc)
    except (UnicodeDecodeError, LookupError):
        try:
            return body.decode(enc, errors='replace')
        except Exception:
            return body.decode(fallback_encoding, errors='replace')
