"""Regex helpers for PII in financial regulatory text.

Each entry is `(pattern, replacement_tag)`. Matches are replaced
left-to-right by `PiiTagRedactor` in `utils.py`.

Design notes
------------
- The `Account`-keyword pattern below is deliberately strict: it requires
  the identifier token to START WITH A DIGIT (or a letter followed by
  digits). Earlier versions used `[A-Z0-9]{6,20}` case-insensitively, which
  matched the English word "account" followed by any six-letter word
  (e.g. "account holder", "account opening", "correspondent account the
  foreign…"). That destroyed AML terms of art ("correspondent account",
  "payable-through account", "shell account", "funnel account") by
  rewriting them to "[ACCOUNT_ID]" and stripped material typology
  vocabulary from the CPT corpus.
- IBAN / SWIFT / routing-number patterns remain unchanged — they are
  format-specific and do not have the same false-positive profile.
"""

from __future__ import annotations

import re

EIN_PATTERN = re.compile(
    r"\b\d{2}-\d{7}\b"
)

# Must start with a digit OR a letter immediately followed by digits. That
# excludes plain English words (which are all-letter) while still catching
# typical account-number shapes like "Account 12345678", "Acct# A12345678",
# "A/C: 1234-5678-90".
ACCOUNT_ID_PATTERNS = [
    re.compile(
        r"\b(?:Account|Acct\.?|A/C)[\s#:.]*"
        r"(\d[A-Z0-9 \-]{5,19}|[A-Z]\d[A-Z0-9 \-]{4,18})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bIBAN[\s:]*[A-Z0-9]{15,34}\b"),
    re.compile(r"\bSWIFT[\s:]*[A-Z]{6}[A-Z0-9]{2,5}\b"),
]

ROUTING_NUMBER_PATTERN = re.compile(
    r"\b(?:routing|aba)[\s#:]*\d{9}\b", re.IGNORECASE
)


def build_regex_replacements(enable_ein: bool, enable_account_id: bool) -> list[tuple[re.Pattern[str], str]]:
    """Return (pattern, replacement) tuples in priority order."""
    out: list[tuple[re.Pattern[str], str]] = []
    if enable_ein:
        out.append((EIN_PATTERN, "[EIN]"))
    if enable_account_id:
        for p in ACCOUNT_ID_PATTERNS:
            out.append((p, "[ACCOUNT_ID]"))
        out.append((ROUTING_NUMBER_PATTERN, "[ACCOUNT_ID]"))
    return out
