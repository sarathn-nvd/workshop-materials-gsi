"""Stage 3 — LLM-extract prompts for prose-source paths (Records 4 + 6).

Record_4 is SARSum (label=True; suspicious activity, narrative carries red flags
explicitly).
Record_6 is CFPB consumer complaints (label=False; benign by construction —
the consumer is reporting an issue with their bank, not engaging in money
laundering).

The earlier version of this prompt did not differentiate by label, which led
to ~47% of CFPB-derived records being flagged `evidence_actually_suspicious`
by the LLM judge — the extractor produced sub-CTR cash splitting and other
classic red flags that contradicted the benign-label semantics of the source.

This prompt is now LABEL-AWARE: when `is_suspicious=false`, the extractor is
instructed to produce transactions that look like ordinary banking activity
consistent with the declared profile, NOT structuring / smurfing / pass-through
patterns.
"""

LLM_EXTRACT_TX_SYSTEM = (
    "You extract a list of transactions from a free-text narrative. The user "
    "message tells you whether the case is SUSPICIOUS (`is_suspicious=true`) "
    "or BENIGN (`is_suspicious=false`); the transactions you emit MUST reflect "
    "that label.\n\n"
    "Output a JSON array where each element has exactly:\n"
    "  {date, amount, currency, counterparty, channel, notes}\n"
    "with channel ∈ {wire, ach, cash, card, cheque, crypto}.\n\n"
    "Severity → count: light=1-5, medium=6-15, heavy=16-50.\n\n"
    "BENIGN-MODE CONSTRAINTS (when is_suspicious=false):\n"
    "  • Total transaction value SHOULD be within 1.0× to 1.5× the declared "
    "    monthly volume — NOT 5×, NOT 30×.\n"
    "  • Cash transactions: each must be UNDER $5,000 (well below the $10K "
    "    CTR threshold AND below the $5K SAR threshold). Do NOT produce "
    "    multiple sub-$10K cash deposits across multiple branches — that is "
    "    structuring, which is the OPPOSITE of benign.\n"
    "  • Counterparties: ordinary names (utility companies, retailers, "
    "    mainstream banks, employer payroll, family members). Do NOT use "
    "    foreign / sanctioned-jurisdiction counterparties unless the "
    "    narrative explicitly mentions them as routine activity.\n"
    "  • Channels: weighted toward ach (recurring bills, payroll, transfers), "
    "    card (everyday purchases), cheque (rent, traditional payments). "
    "    Wire / crypto / large cash should be rare or absent.\n"
    "  • Pattern: ordinary, dispersed activity. Do NOT cluster transactions "
    "    on a single date, do NOT use repeated identical amounts, do NOT "
    "    show rapid-velocity transfers to a single counterparty.\n\n"
    "SUSPICIOUS-MODE CONSTRAINTS (when is_suspicious=true):\n"
    "  • Reflect the typology in the activity pattern (structuring → sub-CTR "
    "    cash; smurfing → many small cash deposits; layering → rapid wire "
    "    movement; trade_based_ml → over/under-invoicing; etc.).\n"
    "  • Total may exceed declared volume by 3–6× depending on typology.\n\n"
    "Output ONLY the JSON array."
)

LLM_EXTRACT_TX_USER = (
    "Narrative:\n{{ notes }}\n\n"
    "is_suspicious: {{ is_suspicious }}\n"
    "Severity bucket: {{ severity }}\n"
    "Typology hypothesis: {{ typology }}\n"
    "Entity declared monthly volume: ${{ expected_monthly_volume }}\n\n"
    "Extract the transactions list as a JSON array. "
    "Honor the BENIGN-MODE or SUSPICIOUS-MODE constraints from the system "
    "prompt according to is_suspicious."
)
