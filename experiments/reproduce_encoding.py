"""demo/encoding_reproduction.py — Japanese encoding autodetect.

Sites like netkeiba.com have inconsistent encodings across pages
(UTF-8 since 2026, EUC-JP historically, some pages still Shift-JIS).
A single hardcoded decoder produces garbage ('mojibake') when the
server charset changes under you.

This reproductions the autodetect decoder that picks per-request:
  1. HTTP Content-Type charset
  2. HTML <meta charset="..."> or <meta http-equiv>
  3. UTF-8 BOM
  4. Heuristic score: Japanese-char count minus replacement-char penalty

Regression-tested against the exact mojibake signature we hit in production:
    `2026綛�11���13��ャ` (UTF-8 bytes decoded as EUC-JP;
    actual text was `2026年11月13日`).

Runs <1 second.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _html_decode import decode_html, detect_encoding


def main():
    print('=== Encoding autodetect reproduction ===')
    print()
    ORIGINAL = '2026年11月13日 1回東京1日目 新馬戦'

    # Case 1: modern netkeiba (UTF-8 server, no meta tag)
    body = ORIGINAL.encode('utf-8')
    enc = detect_encoding(body)
    out = decode_html(body)
    print(f'1. Bare UTF-8 (no meta, no header):')
    print(f'     detected encoding: {enc}')
    print(f'     round-trip OK: {out == ORIGINAL}')
    print()

    # Case 2: historical netkeiba (EUC-JP + meta tag)
    body = (b'<meta http-equiv="Content-Type" content="text/html; charset=euc-jp">'
            + ORIGINAL.encode('euc_jp'))
    enc = detect_encoding(body)
    out = decode_html(body)
    print(f'2. EUC-JP w/ <meta http-equiv>:')
    print(f'     detected encoding: {enc}')
    print(f'     round-trip OK: {ORIGINAL in out}')
    print()

    # Case 3: JRA.jp (Shift-JIS + HTTP charset header)
    body = ORIGINAL.encode('shift_jis')
    enc = detect_encoding(body, content_type='text/html; charset=Shift_JIS')
    out = decode_html(body, content_type='text/html; charset=Shift_JIS')
    print(f'3. Shift-JIS w/ HTTP Content-Type header:')
    print(f'     detected encoding: {enc}')
    print(f'     round-trip OK: {out == ORIGINAL}')
    print()

    # Case 4: regression — the exact mojibake signature from a real bug
    body = ORIGINAL.encode('utf-8')
    wrong_decode = body.decode('euc_jp', errors='replace')
    out = decode_html(body)   # autodetect; no meta/BOM/charset
    print(f'4. REGRESSION: UTF-8 bytes in the wild (what the old `.decode("euc-jp")` produced):')
    print(f'     old wrong output:  {wrong_decode[:40]!r}  ← mojibake 綛/ャ pattern')
    print(f'     new autodetect:    {out[:40]!r}')
    print(f'     fixed:             {"✅" if "綛" not in out and "年" in out else "❌"}')


if __name__ == '__main__':
    main()
