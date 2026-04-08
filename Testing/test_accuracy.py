"""
test_accuracy.py — Retrieval Accuracy Evaluation Suite
========================================================
Evaluates retrieval accuracy on **real-world** documents only (Type E-I).

Real-world documents are read from ``Testing/pdf_cache/`` only.
Ground truth is resolved dynamically by
matching content keywords against extracted page text.

Document types
--------------
Type E — IRS Publication 17 (Federal Tax Guide).
Type F — NIST SP 800-63B (Digital Identity Guidelines).
Type G — NASA Apollo 11 Mission Report.
Type H — Apple 10-K Annual Report (SEC filing).
Type I — ArXiv: "Attention Is All You Need" (Transformer paper).

Evaluation metrics
------------------
Precision@k   — fraction of top-k results that are correct
Recall@k      — fraction of correct pages found in top-k
MRR           — Mean Reciprocal Rank (1/rank of first correct hit)
nDCG@5        — normalised Discounted Cumulative Gain @ 5
Distractor    — % of queries where top-1 is a plausible wrong page
Exact-match   — BM25-critical: query contains literal string from doc
Semantic-match— Dense-critical: query uses synonym/paraphrase

Usage
-----
    python test_accuracy.py                    # all real docs (E-I), adaptive mode
    python test_accuracy.py --verbose          # show per-query detail
    python test_accuracy.py --type E           # run only real doc E
    python test_accuracy.py --mode adaptive    # single mode (HyDE/CE applied per-case)
    python test_accuracy.py --no-generative    # deprecated alias; kept for compatibility
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# PyMuPDF compatibility: newer versions use `pymupdf`, older use `fitz`
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:
    try:
        import fitz            # PyMuPDF < 1.24.3
    except ImportError:
        fitz = None            # type: ignore[assignment]

PDF_CACHE_DIR = _HERE / "pdf_cache"

_t0 = time.time()


def _elapsed() -> str:
    """Return a human-readable elapsed time since the test started."""
    secs = time.time() - _t0
    if secs < 60:
        return f"{secs:.0f}s"
    mins, secs = divmod(secs, 60)
    return f"{int(mins)}m{int(secs):02d}s"


def _status(msg: str) -> None:
    """Print a timestamped status line."""
    print(f"  [{_elapsed():>7s}] {msg}", flush=True)


# ===========================================================================
# Synthetic document content  (Types A–D)
# ===========================================================================

# Each entry: (page_number, text_content)
TYPE_A_PAGES: list[tuple[int, str]] = [
    (1, textwrap.dedent("""\
        AS568 O-Ring Standard Dimensions — Table 1
        Dash No. | ID (in) | Cross-section (in) | Material | Shore A
        -004     | 0.070   | 0.070              | NBR      | 70
        -009     | 0.145   | 0.070              | NBR      | 70
        -110     | 0.489   | 0.103              | FKM      | 75
        -210     | 1.239   | 0.139              | EPDM     | 70
        -425     | 4.239   | 0.275              | Silicone | 60
        Note: Dash -004 not recommended for dynamic applications.
    """)),
    (2, textwrap.dedent("""\
        Material Compatibility — Table 2: Pressure and Temperature Ratings
        Material | Max Pressure (PSI) | Max Temp (°F) | Min Temp (°F) | Application
        NBR      | 3000               | 250           | -40           | Hydraulic
        FKM      | 3000               | 400           | -15           | Chemical
        EPDM     | 1500               | 300           | -65           | Steam
        Silicone | 500                | 450           | -175          | High temp
        PTFE     | 5000               | 500           | -300          | Universal
        NBR is not recommended for steam or hot water environments.
    """)),
    (3, textwrap.dedent("""\
        Compound Selection Guide — Table 3: Chemical Resistance
        Chemical          | NBR  | FKM  | EPDM | Silicone
        Hydraulic oil     | A    | A    | D    | C
        Phosphate ester   | D    | A    | D    | D
        Ethylene glycol   | A    | B    | A    | C
        Gasoline          | B    | A    | D    | D
        Steam             | D    | C    | A    | D
        A=Excellent B=Good C=Fair D=Not recommended
        FKM (Viton) is suitable for most hydrocarbon and chemical applications.
    """)),
    (4, textwrap.dedent("""\
        Dimensional Tolerances and Surface Finish Requirements
        O-ring grooves must comply with AS568A tolerances.
        Surface finish: 16 to 32 micro-inch Ra for dynamic seals.
        Surface finish: 32 to 63 micro-inch Ra for static seals.
        Groove width should exceed cross-section by 20-30%.
        Diametral clearance must not exceed 0.005 inches at max pressure.
        Lubrication with compatible grease reduces installation damage by 60%.
    """)),
]

TYPE_B_PAGES: list[tuple[int, str]] = [
    (1, textwrap.dedent("""\
        Introduction to Hydraulic Seal Systems
        Hydraulic sealing systems are critical components in fluid power equipment.
        They prevent leakage of pressurised fluid between stationary and moving parts.
        The selection of appropriate sealing materials depends on operating pressure,
        temperature range, fluid compatibility, and dynamic or static application.
        Poor seal selection is the leading cause of hydraulic system failure in
        industrial machinery, accounting for approximately 40% of unplanned downtime.
    """)),
    (2, textwrap.dedent("""\
        Elastomer Degradation Mechanisms
        Rubber compounds degrade through several mechanisms including oxidation,
        chemical attack, compression set, and thermal aging. Nitrile rubber (NBR)
        undergoes oxidative crosslinking above 250°F, becoming brittle and prone
        to cracking. Fluorocarbon compounds such as Viton maintain elasticity at
        temperatures up to 400°F due to their fluorine-carbon bond stability.
        EPDM excels in water and steam environments but is incompatible with
        petroleum-based fluids due to excessive swelling.
    """)),
    (3, textwrap.dedent("""\
        Installation Best Practices for O-Ring Seals
        Proper installation is as important as correct material selection.
        Always lubricate o-rings with a compatible lubricant before installation.
        Avoid stretching the o-ring beyond 50% of its inside diameter.
        Inspect mating surfaces for burrs, scratches, and contamination.
        Use installation tools rather than sharp implements to avoid nicking.
        After installation, perform a low-pressure leak test before full operation.
        Replace any o-ring that shows signs of extrusion or nibbling after use.
    """)),
    (4, textwrap.dedent("""\
        Failure Analysis and Troubleshooting
        Common failure modes include: extrusion (gap too large or pressure too high),
        compression set (wrong material or over-compression), spiral failure (dynamic
        rotation causing twisting), chemical degradation, and abrasive wear.
        Extrusion failures show characteristic nibbling at the low-pressure face.
        Compression set appears as flat spots equal to the groove width.
        Chemical attack typically produces swelling, softening, or surface cracking.
        Thermal failures result in hardening, cracking, or loss of elastic recovery.
    """)),
]

TYPE_C_PAGES: list[tuple[int, str]] = [
    (1, textwrap.dedent("""\
        Annual Financial Summary — FY2023
        Total Revenue: $4.2 billion (up 12% year-over-year)
        Gross Profit: $1.8 billion (margin 42.9%)
        Operating Income: $620 million
        Net Income: $410 million
        Earnings Per Share: $3.42
        Capital Expenditure: $280 million invested in manufacturing expansion.
        The industrial sealing division grew 18% driven by aerospace demand.
    """)),
    (2, textwrap.dedent("""\
        Segment Performance Analysis
        The sealing products segment contributed 38% of total revenue at $1.6 billion.
        Aerospace sealing revenue increased 31% to $480 million due to
        commercial aircraft production ramp-up and defense contracts.
        Automotive OEM revenue declined 4% due to EV transition inventory reduction.
        Industrial MRO (maintenance, repair, overhaul) grew 9% to $640 million.
        Operating margin for the sealing segment was 24.1% versus 21.3% prior year.
    """)),
    (3, textwrap.dedent("""\
        Revenue by Geography — Chart Description
        North America: $1.9 billion (45% of total), grew 8%.
        Europe: $1.1 billion (26%), flat versus prior year.
        Asia Pacific: $0.9 billion (21%), grew 22% led by China EV battery sealing.
        Rest of World: $0.3 billion (8%).
        Foreign exchange headwind of $85 million impacted reported results.
        Constant-currency growth was 14% versus 12% reported.
    """)),
    (4, textwrap.dedent("""\
        Outlook and Guidance for FY2024
        Management guides revenue of $4.6 to $4.8 billion for fiscal 2024.
        Gross margin expected to improve 50-100 basis points through pricing and
        operational efficiency programmes. Capital expenditure of $320 million
        planned, focused on high-performance fluoropolymer capacity expansion.
        The company expects to repurchase $200 million of shares in fiscal 2024.
        Key risks include raw material cost inflation and geopolitical uncertainty.
    """)),
]

TYPE_D_PAGES: list[tuple[int, str]] = [
    (1, textwrap.dedent("""\
        Abstract: Fluorocarbon Elastomer Thermal Stability at Extreme Temperatures
        We investigate the thermomechanical properties of FKM grade compounds
        subject to isothermal aging at temperatures between 150°C and 250°C.
        Tensile strength, elongation at break, and compression set were measured
        after 24h, 72h, and 168h exposure intervals. Results indicate that standard
        FKM retains 85% of initial tensile strength after 168h at 200°C, while
        perfluoroelastomer (FFKM) retains 94% under identical conditions.
    """)),
    (2, textwrap.dedent("""\
        Methodology: Sample Preparation and Testing Protocol
        FKM and FFKM specimens were compression-moulded per ASTM D395 Method B.
        Shore A hardness was measured per ISO 7619-1 after 23°C conditioning.
        Tensile specimens (type 2 dumbbell) were tested per ISO 37 at 200mm/min.
        Thermal aging was performed in a forced-air oven per ASTM D573.
        Statistical analysis used ANOVA with 95% confidence intervals.
        Three independent batches were evaluated with n=5 specimens per condition.
    """)),
    (3, textwrap.dedent("""\
        Results: Compression Set and Elastic Recovery
        Standard FKM at 150°C, 22h: compression set 8.2% (ASTM D395 Method B).
        Standard FKM at 200°C, 22h: compression set 14.7%.
        FFKM at 200°C, 22h: compression set 5.1%, showing superior elastic recovery.
        The Arrhenius activation energy for FKM oxidative degradation is 92 kJ/mol,
        giving a predicted service life at 175°C of approximately 2,500 hours.
        These results are consistent with published values by Smith et al. (2019).
    """)),
    (4, textwrap.dedent("""\
        Discussion and Conclusions
        The superior thermal stability of FFKM versus standard FKM is attributable
        to the fully-substituted fluorine backbone which resists oxidative attack.
        Standard FKM remains the cost-effective choice for applications below 200°C.
        FFKM is recommended for semiconductor processing, aerospace, and chemical
        plant applications where temperatures exceed 200°C or exposure to aggressive
        solvents requires maximum chemical inertness. Future work will investigate
        hybrid laminates combining FFKM face seals with FKM body compounds.
    """)),
]

# Human-readable names for report output (keys match QAItem.doc_type)
DOC_TYPE_LABELS: dict[str, str] = {
    "A": "Engineering datasheet (table-heavy)",
    "B": "Technical manual (prose-heavy)",
    "C": "Financial report (mixed)",
    "D": "Scientific abstract (academic prose)",
    "E": "IRS Publication 17 (tax guide)",
    "F": "NIST SP 800-63B (auth guidelines)",
    "G": "Apollo 11 Mission Report",
    "H": "Apple 10-K Annual Report",
    "I": "Attention Is All You Need (arXiv)",
}

# ===========================================================================
# Real-world PDF definitions  (Types E–I)
# ===========================================================================

REAL_PDF_SOURCES: dict[str, dict] = {
    "E": {
        "url":  "https://www.irs.gov/pub/irs-prior/p17--2024.pdf",
        "file": "irs_pub17_2024.pdf",
        "desc": "IRS Publication 17 — Your Federal Income Tax (2024)",
    },
    "F": {
        "url":  "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "file": "nist_sp800_63b.pdf",
        "desc": "NIST SP 800-63B — Digital Identity Guidelines",
    },
    "G": {
        "url":  "https://ntrs.nasa.gov/api/citations/19710015566/downloads/19710015566.pdf",
        "file": "apollo11_mission_report.pdf",
        "desc": "NASA Apollo 11 Mission Report",
    },
    "H": {
        "url":  "https://investor.apple.com/files/doc_earnings/2023/q4/filing/_10-K-Q4-2023-As-Filed.pdf",
        "file": "apple_10k_2023.pdf",
        "desc": "Apple Inc. 10-K FY2023",
    },
    "I": {
        "url":  "https://arxiv.org/pdf/1706.03762v7",
        "file": "attention_is_all_you_need.pdf",
        "desc": "Attention Is All You Need (Vaswani et al. 2017)",
    },
}


def _get_cached_pdf(doc_type: str, *, verbose: bool = False) -> Path | None:
    """Return cached real PDF from pdf_cache/; do not download."""
    info = REAL_PDF_SOURCES.get(doc_type)
    if info is None:
        return None
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_CACHE_DIR / info["file"]
    if dest.exists() and dest.stat().st_size > 1000:
        _status(f"PDF cache hit: {info['desc']}")
        return dest
    _status(
        f"Missing cached PDF for {info['desc']}: expected at {dest}. "
        "Place the file in Testing/pdf_cache/ and rerun."
    )
    return None


# ===========================================================================
# Ground-truth QA dataset
# ===========================================================================

@dataclass
class QAItem:
    """A single test query with expected answer page(s)."""
    query:           str
    expected_pages:  list[int]           # 1-based; may be empty for real docs
    doc_type:        str                 # "A"–"I"
    retrieval_type:  str                 # "exact", "semantic", "negation", "numeric"
    note:            str = ""
    difficulty:      str = "medium"      # "easy", "medium", "hard"
    query_length:    str = "medium"      # "short", "medium", "long"
    expected_content: list[str] = field(default_factory=list)
    distractor_keywords: list[str] = field(default_factory=list)


# ── Synthetic queries (Types A–D, 33 total) ─────────────────────────────────

QA_DATASET: list[QAItem] = [
    # ── Type A: Engineering datasheet ──────────────────────────────────────
    QAItem("dash -004 o-ring dimensions",        [1], "A", "exact",
           "Literal part code — BM25 critical",
           difficulty="easy", query_length="short"),
    QAItem("NBR maximum temperature rating",     [2], "A", "numeric",
           "Requires linking material to temperature table",
           difficulty="medium", query_length="short"),
    QAItem("FKM pressure rating PSI",            [2], "A", "numeric",
           "Numeric spec lookup",
           difficulty="easy", query_length="short"),
    QAItem("o-ring for steam application",       [2], "A", "semantic",
           "Steam -> EPDM mapping",
           difficulty="medium", query_length="short"),
    QAItem("not suitable for hydraulic oil",     [3], "A", "negation",
           "Negation: should avoid FKM for hydraulic oil match",
           difficulty="hard", query_length="short"),
    QAItem("silicone high temperature seal",     [1, 2], "A", "semantic",
           "Silicone compound and temperature",
           difficulty="medium", query_length="short"),
    QAItem("groove surface finish static seal",  [4], "A", "semantic",
           "Surface finish requirements page",
           difficulty="medium", query_length="short"),
    QAItem("nitrile rubber chemical resistance", [2, 3], "A", "semantic",
           "Synonym: nitrile = NBR",
           difficulty="medium", query_length="short"),
    QAItem("AS568 standard 0.489 inside diameter", [1], "A", "exact",
           "Exact numeric match via BM25",
           difficulty="easy", query_length="medium"),
    QAItem("avoid EPDM for petroleum",           [3], "A", "negation",
           "Negation with chemical name",
           difficulty="hard", query_length="short"),

    # ── Type B: Technical manual (prose) ───────────────────────────────────
    QAItem("how to install an o-ring correctly", [3], "B", "semantic",
           "Paraphrase of installation best practices",
           difficulty="medium", query_length="medium"),
    QAItem("Viton temperature resistance",       [2], "B", "semantic",
           "Viton = FKM synonym",
           difficulty="medium", query_length="short"),
    QAItem("what causes compression set failure",[4], "B", "semantic",
           "Failure mode identification",
           difficulty="medium", query_length="medium"),
    QAItem("elastomer oxidation above 250 degrees", [2], "B", "numeric",
           "Specific temperature in prose",
           difficulty="medium", query_length="medium"),
    QAItem("spiral failure rotating seal",       [4], "B", "semantic",
           "Specific failure mode in narrative",
           difficulty="medium", query_length="short"),
    QAItem("lubricant installation damage",      [3], "B", "semantic",
           "Prose fact: 60% reduction",
           difficulty="medium", query_length="short"),
    QAItem("EPDM incompatible with petroleum",   [2], "B", "negation",
           "Incompatibility in prose",
           difficulty="hard", query_length="short"),
    QAItem("hydraulic system unplanned downtime",[1], "B", "semantic",
           "Statistic in introduction",
           difficulty="medium", query_length="short"),

    # ── Type C: Financial report ──────────────────────────────────────────
    QAItem("total revenue fiscal year 2023",     [1], "C", "exact",
           "Exact number lookup",
           difficulty="easy", query_length="medium"),
    QAItem("aerospace sealing revenue growth",   [2], "C", "semantic",
           "Segment performance",
           difficulty="medium", query_length="short"),
    QAItem("Asia Pacific revenue percentage",    [3], "C", "numeric",
           "Geographic breakdown",
           difficulty="medium", query_length="short"),
    QAItem("EV electric vehicle sealing demand", [3], "C", "semantic",
           "Synonym: EV = electric vehicle",
           difficulty="medium", query_length="medium"),
    QAItem("earnings per share",                 [1], "C", "exact",
           "Financial metric — exact",
           difficulty="easy", query_length="short"),
    QAItem("guidance outlook next year revenue", [4], "C", "semantic",
           "Forward-looking statement",
           difficulty="medium", query_length="medium"),
    QAItem("fluoropolymer capacity expansion",   [4], "C", "semantic",
           "Capex detail in guidance",
           difficulty="medium", query_length="short"),
    QAItem("constant currency growth rate",      [3], "C", "semantic",
           "FX-adjusted metric in prose",
           difficulty="hard", query_length="short"),

    # ── Type D: Scientific abstract ───────────────────────────────────────
    QAItem("FKM compression set at 200 degrees", [3], "D", "numeric",
           "Exact experimental result",
           difficulty="easy", query_length="medium"),
    QAItem("FFKM superior thermal stability reason", [4], "D", "semantic",
           "Explanation in discussion",
           difficulty="medium", query_length="medium"),
    QAItem("Arrhenius activation energy fluorocarbon", [3], "D", "exact",
           "Specific numeric parameter",
           difficulty="easy", query_length="short"),
    QAItem("semiconductor processing high temperature seal", [4], "D", "semantic",
           "Application recommendation",
           difficulty="medium", query_length="medium"),
    QAItem("ASTM tensile test methodology",      [2], "D", "exact",
           "Standard reference exact match",
           difficulty="easy", query_length="short"),
    QAItem("service life prediction 175 degrees",[3], "D", "numeric",
           "Calculated lifetime value",
           difficulty="medium", query_length="medium"),
    QAItem("FFKM versus standard FKM comparison",[3, 4], "D", "semantic",
           "Comparison across pages",
           difficulty="hard", query_length="medium"),
]


# ── Real-world queries (Types E–I, ~30 each = 150 total) ────────────────────

REAL_QA_DATASET: list[QAItem] = [
    # ==================================================================
    # Type E — IRS Publication 17  (30 queries)
    # ==================================================================
    # Easy / exact
    QAItem("standard deduction amount", [], "E", "exact",
           difficulty="easy", query_length="short",
           expected_content=["standard deduction"]),
    QAItem("filing requirements", [], "E", "exact",
           difficulty="easy", query_length="short",
           expected_content=["filing requirement"]),
    QAItem("earned income credit", [], "E", "exact",
           difficulty="easy", query_length="short",
           expected_content=["earned income"]),
    QAItem("child tax credit amount", [], "E", "exact",
           difficulty="easy", query_length="short",
           expected_content=["child tax credit"]),
    QAItem("capital gains tax rate", [], "E", "numeric",
           difficulty="easy", query_length="short",
           expected_content=["capital gain"]),
    # Medium / semantic
    QAItem("what income is not taxable", [], "E", "negation",
           difficulty="medium", query_length="medium",
           expected_content=["not taxable", "nontaxable", "excluded from income"],
           distractor_keywords=["taxable income"]),
    QAItem("how to file if you are self-employed", [], "E", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["self-employ"]),
    QAItem("what medical expenses can I deduct from taxes", [], "E", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["medical", "deduct"]),
    QAItem("retirement savings contribution tax benefit", [], "E", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["retirement", "contribution"]),
    QAItem("itemized deductions versus standard deduction", [], "E", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["itemized deduction"]),
    QAItem("dependents claimed on tax return", [], "E", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["dependent"]),
    QAItem("tax withholding from wages", [], "E", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["withholding"]),
    QAItem("alimony payments tax treatment", [], "E", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["alimony"]),
    QAItem("gambling winnings reporting requirement", [], "E", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["gambling"]),
    QAItem("home mortgage interest deduction rules", [], "E", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["mortgage interest"]),
    # Hard / multi-hop / negation / paraphrase
    QAItem("which education expenses are not deductible for individual taxpayers", [], "E", "negation",
           difficulty="hard", query_length="long",
           expected_content=["education"],
           distractor_keywords=["deduction"]),
    QAItem("penalty for early withdrawal from retirement account", [], "E", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["early", "withdrawal", "penalty"]),
    QAItem("foreign earned income exclusion requirements for US citizens living abroad", [], "E", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["foreign", "earned income"]),
    QAItem("estimated tax payment deadlines", [], "E", "numeric",
           difficulty="medium", query_length="short",
           expected_content=["estimated tax"]),
    QAItem("social security benefits taxability threshold", [], "E", "numeric",
           difficulty="hard", query_length="medium",
           expected_content=["social security"]),
    QAItem("casualty loss deduction requirements after disaster", [], "E", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["casualty", "loss"]),
    QAItem("tax filing status married filing separately", [], "E", "exact",
           difficulty="medium", query_length="medium",
           expected_content=["married filing separately"]),
    QAItem("alternative minimum tax", [], "E", "exact",
           difficulty="medium", query_length="short",
           expected_content=["alternative minimum tax"]),
    QAItem("charitable contribution deduction limits for individuals", [], "E", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["charitable", "contribution"]),
    QAItem("IRA contribution limits", [], "E", "numeric",
           difficulty="medium", query_length="short",
           expected_content=["individual retirement", "IRA"]),
    QAItem("what records should I keep for taxes", [], "E", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["record"]),
    QAItem("penalties for not filing a tax return on time", [], "E", "negation",
           difficulty="hard", query_length="long",
           expected_content=["penalty", "failure to file"]),
    QAItem("head of household filing requirements and qualifications", [], "E", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["head of household"]),
    QAItem("student loan interest deduction", [], "E", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["student loan"]),
    QAItem("qualifying widow widower filing status", [], "E", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["qualifying", "widow"]),

    # ==================================================================
    # Type F — NIST SP 800-63B  (30 queries)
    # ==================================================================
    # Easy / exact
    QAItem("memorized secret requirements", [], "F", "exact",
           difficulty="easy", query_length="short",
           expected_content=["memorized secret"]),
    QAItem("authenticator assurance level", [], "F", "exact",
           difficulty="easy", query_length="short",
           expected_content=["authenticator assurance level"]),
    QAItem("multi-factor authentication", [], "F", "exact",
           difficulty="easy", query_length="short",
           expected_content=["multi-factor"]),
    QAItem("minimum password length", [], "F", "exact",
           difficulty="easy", query_length="short",
           expected_content=["minimum", "character"]),
    QAItem("look-up secret authenticator", [], "F", "exact",
           difficulty="easy", query_length="short",
           expected_content=["look-up secret"]),
    # Medium / semantic
    QAItem("when is SMS verification not permitted for authentication", [], "F", "negation",
           difficulty="medium", query_length="long",
           expected_content=["restricted", "PSTN", "SMS", "out-of-band"],
           distractor_keywords=["authenticator"]),
    QAItem("requirements for biometric authentication", [], "F", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["biometric"]),
    QAItem("session management after authentication", [], "F", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["session"]),
    QAItem("what authenticators satisfy AAL2", [], "F", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["AAL2", "assurance level 2"]),
    QAItem("cryptographic authenticator requirements", [], "F", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["cryptographic"]),
    QAItem("verifier requirements for password storage", [], "F", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["verifier", "password"]),
    QAItem("single-factor OTP device", [], "F", "exact",
           difficulty="medium", query_length="short",
           expected_content=["single-factor", "OTP"]),
    QAItem("rate limiting failed authentication attempts", [], "F", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["rate limit", "throttl"]),
    QAItem("reauthentication timeout requirements", [], "F", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["reauthentication"]),
    QAItem("token binding requirements for authenticators", [], "F", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["binding"]),
    # Hard / paraphrase / negation
    QAItem("what password policies SHALL NOT be implemented by verifiers", [], "F", "negation",
           difficulty="hard", query_length="long",
           expected_content=["SHALL NOT"],
           distractor_keywords=["SHALL"]),
    QAItem("resistance to man-in-the-middle attacks during authentication", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["man-in-the-middle", "verifier impersonation"]),
    QAItem("physical authenticator destruction and revocation procedures", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["revocation", "loss"]),
    QAItem("how to handle compromised authenticator credentials", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["compromised", "breach"]),
    QAItem("federation and assertion requirements across identity providers", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["federation", "assertion"]),
    QAItem("requirements that do NOT apply to AAL1 authentication", [], "F", "negation",
           difficulty="hard", query_length="long",
           expected_content=["AAL1"],
           distractor_keywords=["AAL2", "AAL3"]),
    QAItem("password composition rules that should not be enforced", [], "F", "negation",
           difficulty="hard", query_length="long",
           expected_content=["composition", "SHALL NOT"]),
    QAItem("out-of-band authenticator verification methods", [], "F", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["out-of-band"]),
    QAItem("activation secret for multi-factor authenticator", [], "F", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["activation"]),
    QAItem("verifier impersonation resistance at AAL3", [], "F", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["verifier impersonation", "AAL3"]),
    QAItem("hardware cryptographic device requirements for authentication", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["hardware", "cryptographic"]),
    QAItem("claimant identity verification during enrollment", [], "F", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["enrollment", "identity"]),
    QAItem("replay resistance mechanisms", [], "F", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["replay"]),
    QAItem("entropy requirements for random authenticator secrets", [], "F", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["entropy"]),
    QAItem("restricted authenticator definition", [], "F", "exact",
           difficulty="medium", query_length="short",
           expected_content=["restricted authenticator"]),

    # ==================================================================
    # Type G — Apollo 11 Mission Report  (30 queries)
    # ==================================================================
    # Easy / exact
    QAItem("Apollo 11 launch date and time", [], "G", "exact",
           difficulty="easy", query_length="short",
           expected_content=["July", "16", "1969"]),
    QAItem("lunar module name", [], "G", "exact",
           difficulty="easy", query_length="short",
           expected_content=["Eagle", "lunar module"]),
    QAItem("command module name", [], "G", "exact",
           difficulty="easy", query_length="short",
           expected_content=["Columbia", "command module"]),
    QAItem("crew members of Apollo 11", [], "G", "exact",
           difficulty="easy", query_length="short",
           expected_content=["Armstrong", "Aldrin", "Collins"]),
    QAItem("landing site coordinates", [], "G", "exact",
           difficulty="easy", query_length="short",
           expected_content=["tranquility", "landing"]),
    # Medium / semantic & numeric
    QAItem("total mission elapsed time duration", [], "G", "numeric",
           difficulty="medium", query_length="medium",
           expected_content=["mission", "elapsed"]),
    QAItem("extravehicular activity duration on lunar surface", [], "G", "numeric",
           difficulty="medium", query_length="long",
           expected_content=["extravehicular", "EVA"]),
    QAItem("lunar surface experiments deployed by astronauts", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["experiment"]),
    QAItem("spacecraft trajectory during translunar injection", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["translunar"]),
    QAItem("command module splashdown location", [], "G", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["splashdown", "recovery"]),
    QAItem("lunar orbit insertion burn parameters", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["lunar orbit"]),
    QAItem("propulsion system performance during mission", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["propulsion"]),
    QAItem("communications systems and blackout periods", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["communication"]),
    QAItem("environmental control system cabin temperature", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["environmental control", "temperature"]),
    QAItem("lunar sample collection and return mass", [], "G", "numeric",
           difficulty="medium", query_length="medium",
           expected_content=["sample", "lunar"]),
    # Hard / multi-hop / paraphrase
    QAItem("anomalies or problems encountered during descent to lunar surface", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["anomal", "descent"]),
    QAItem("fuel remaining at lunar landing when engines were shut down", [], "G", "numeric",
           difficulty="hard", query_length="long",
           expected_content=["fuel", "remaining", "landing"]),
    QAItem("computer program alarms during powered descent 1202 1201", [], "G", "exact",
           difficulty="hard", query_length="long",
           expected_content=["1202", "alarm", "program"]),
    QAItem("rendezvous procedures between lunar module and command module", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["rendezvous"]),
    QAItem("thermal protection system performance during reentry", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["thermal", "reentry", "heat shield"]),
    QAItem("guidance navigation and control system accuracy", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["guidance", "navigation"]),
    QAItem("what systems were powered down to conserve energy during the mission", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["power"]),
    QAItem("lunar module ascent engine performance", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["ascent", "engine"]),
    QAItem("passive seismic experiment results", [], "G", "semantic",
           difficulty="hard", query_length="short",
           expected_content=["seismic"]),
    QAItem("crew health and medical observations", [], "G", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["medical", "crew"]),
    QAItem("S-IVB stage separation and performance", [], "G", "exact",
           difficulty="medium", query_length="medium",
           expected_content=["S-IVB"]),
    QAItem("electrical power system fuel cell performance", [], "G", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["electrical", "fuel cell"]),
    QAItem("reaction control system thruster usage", [], "G", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["reaction control"]),
    QAItem("docking mechanism operation between modules", [], "G", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["docking"]),
    QAItem("photographic equipment and film usage during mission", [], "G", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["photograph", "camera"]),

    # ==================================================================
    # Type H — Apple 10-K FY2023  (30 queries)
    # ==================================================================
    # Easy / exact
    QAItem("total net sales revenue", [], "H", "exact",
           difficulty="easy", query_length="short",
           expected_content=["net sales", "revenue"]),
    QAItem("iPhone revenue", [], "H", "exact",
           difficulty="easy", query_length="short",
           expected_content=["iPhone"]),
    QAItem("net income", [], "H", "numeric",
           difficulty="easy", query_length="short",
           expected_content=["net income"]),
    QAItem("earnings per share diluted", [], "H", "exact",
           difficulty="easy", query_length="short",
           expected_content=["earnings per share", "diluted"]),
    QAItem("total assets", [], "H", "numeric",
           difficulty="easy", query_length="short",
           expected_content=["total assets"]),
    # Medium / semantic
    QAItem("services segment revenue growth year over year", [], "H", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["Services"]),
    QAItem("research and development expenditure", [], "H", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["research and development"]),
    QAItem("share repurchase program buyback amount", [], "H", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["repurchase", "shares"]),
    QAItem("operating segments geographic breakdown", [], "H", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["Americas", "Europe", "Greater China"]),
    QAItem("long-term debt maturity schedule", [], "H", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["long-term debt", "maturit"]),
    QAItem("goodwill and intangible assets", [], "H", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["goodwill", "intangible"]),
    QAItem("capital expenditures for property plant and equipment", [], "H", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["capital expenditure", "property"]),
    QAItem("dividends declared per share", [], "H", "numeric",
           difficulty="medium", query_length="short",
           expected_content=["dividend"]),
    QAItem("deferred revenue balance", [], "H", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["deferred revenue"]),
    QAItem("effective income tax rate", [], "H", "numeric",
           difficulty="medium", query_length="short",
           expected_content=["effective tax rate", "income tax"]),
    # Hard / multi-hop / negation
    QAItem("what risk factors does Apple identify related to supply chain disruptions", [], "H", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["risk", "supply chain"]),
    QAItem("revenue from wearables home and accessories segment versus prior year", [], "H", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["Wearables", "Home and Accessories"]),
    QAItem("foreign currency exchange rate impact on international revenue", [], "H", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["foreign currency", "exchange"]),
    QAItem("which product categories experienced revenue decline year over year", [], "H", "negation",
           difficulty="hard", query_length="long",
           expected_content=["decrease", "decline"],
           distractor_keywords=["increase", "growth"]),
    QAItem("legal proceedings and contingent liabilities", [], "H", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["legal proceedings", "contingent"]),
    QAItem("operating lease right-of-use assets", [], "H", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["operating lease", "right-of-use"]),
    QAItem("gross margin percentage for products versus services", [], "H", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["gross margin"]),
    QAItem("critical accounting estimates and judgments", [], "H", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["critical accounting"]),
    QAItem("Mac revenue", [], "H", "exact",
           difficulty="easy", query_length="short",
           expected_content=["Mac"]),
    QAItem("customer concentration risk and major customers", [], "H", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["customer", "concentration"]),
    QAItem("inventory valuation method", [], "H", "semantic",
           difficulty="hard", query_length="short",
           expected_content=["inventor"]),
    QAItem("employee headcount or workforce size", [], "H", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["employee", "full-time"]),
    QAItem("iPad revenue and unit sales", [], "H", "exact",
           difficulty="easy", query_length="short",
           expected_content=["iPad"]),
    QAItem("marketable securities fair value", [], "H", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["marketable securities"]),
    QAItem("cash and cash equivalents balance", [], "H", "exact",
           difficulty="easy", query_length="medium",
           expected_content=["cash and cash equivalents"]),

    # ==================================================================
    # Type I — Attention Is All You Need  (30 queries)
    # ==================================================================
    # Easy / exact
    QAItem("Transformer model architecture", [], "I", "exact",
           difficulty="easy", query_length="short",
           expected_content=["Transformer"]),
    QAItem("self-attention mechanism", [], "I", "exact",
           difficulty="easy", query_length="short",
           expected_content=["self-attention", "Attention"]),
    QAItem("BLEU score results", [], "I", "exact",
           difficulty="easy", query_length="short",
           expected_content=["BLEU"]),
    QAItem("multi-head attention", [], "I", "exact",
           difficulty="easy", query_length="short",
           expected_content=["multi-head", "Multi-Head"]),
    QAItem("positional encoding", [], "I", "exact",
           difficulty="easy", query_length="short",
           expected_content=["positional encoding"]),
    # Medium / semantic
    QAItem("how does the model handle sequential position information without recurrence", [], "I", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["positional", "encoding", "sinusoidal"]),
    QAItem("encoder decoder architecture description of the proposed model", [], "I", "semantic",
           difficulty="medium", query_length="long",
           expected_content=["encoder", "decoder"]),
    QAItem("training hyperparameters and optimizer used", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["optimizer", "Adam"]),
    QAItem("English to German translation performance", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["English-to-German", "WMT"]),
    QAItem("comparison with recurrent neural network models", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["recurrent"]),
    QAItem("scaled dot-product attention formula", [], "I", "exact",
           difficulty="medium", query_length="medium",
           expected_content=["Scaled Dot-Product"]),
    QAItem("feed-forward network in transformer layer", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["feed-forward", "position-wise"]),
    QAItem("dropout regularization used during training", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["dropout"]),
    QAItem("English to French translation BLEU score", [], "I", "numeric",
           difficulty="medium", query_length="medium",
           expected_content=["English-to-French", "French"]),
    QAItem("label smoothing regularization technique", [], "I", "semantic",
           difficulty="medium", query_length="medium",
           expected_content=["label smoothing"]),
    # Hard / paraphrase / multi-hop
    QAItem("why does self-attention have lower computational complexity than recurrent layers for typical sequence lengths", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["complexity", "self-attention"]),
    QAItem("residual connections and layer normalization in the model", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["residual", "layer normalization"]),
    QAItem("number of attention heads and model dimension used in the base model", [], "I", "numeric",
           difficulty="hard", query_length="long",
           expected_content=["head", "dimension", "512"]),
    QAItem("training data and preprocessing for WMT translation task", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["WMT", "training data"]),
    QAItem("how does the model generalize to English constituency parsing", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["constituency parsing", "English"]),
    QAItem("key query value projections in attention mechanism", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["query", "key", "value"]),
    QAItem("effect of varying attention head count on translation quality", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["head", "variation"]),
    QAItem("training cost in GPU hours for the base and big model", [], "I", "numeric",
           difficulty="hard", query_length="long",
           expected_content=["GPU", "training"]),
    QAItem("byte pair encoding vocabulary size used", [], "I", "exact",
           difficulty="medium", query_length="medium",
           expected_content=["byte pair", "BPE", "vocabulary"]),
    QAItem("beam search decoding parameters", [], "I", "semantic",
           difficulty="medium", query_length="short",
           expected_content=["beam"]),
    QAItem("attention visualization and interpretability analysis", [], "I", "semantic",
           difficulty="hard", query_length="medium",
           expected_content=["attention", "visualization"]),
    QAItem("masked attention in decoder to preserve autoregressive property", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["mask", "autoregressive"]),
    QAItem("sinusoidal versus learned positional embeddings comparison", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["sinusoidal", "learned", "positional"]),
    QAItem("number of parameters in the Transformer model", [], "I", "numeric",
           difficulty="medium", query_length="medium",
           expected_content=["parameter"]),
    QAItem("future work and limitations mentioned by the authors", [], "I", "semantic",
           difficulty="hard", query_length="long",
           expected_content=["future"]),
]


# ===========================================================================
# Content validation
# ===========================================================================

def _validate_extraction(
    pdf_path: Path,
    qa_items: list[QAItem],
    *,
    verbose: bool = False,
) -> tuple[list[QAItem], list[str]]:
    """Verify that the PDF extraction found expected content.

    For real-doc QAItems (expected_pages empty, expected_content non-empty),
    scan the PDF page texts and resolve expected_pages dynamically.

    Returns (resolved_items, warnings).
    """
    warnings: list[str] = []
    try:
        import pdfplumber
    except ImportError:
        warnings.append("pdfplumber not available for content validation")
        return qa_items, warnings

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_texts: list[tuple[int, str]] = [
                (i + 1, (page.extract_text() or "").lower())
                for i, page in enumerate(pdf.pages)
            ]
    except Exception as e:
        warnings.append(f"Could not read PDF for validation: {e}")
        return qa_items, warnings

    resolved: list[QAItem] = []
    for item in qa_items:
        if item.expected_pages and not item.expected_content:
            resolved.append(item)
            continue

        if not item.expected_content:
            resolved.append(item)
            continue

        matching_pages: list[int] = []
        for pnum, ptext in page_texts:
            if any(kw.lower() in ptext for kw in item.expected_content):
                matching_pages.append(pnum)

        if not matching_pages:
            warnings.append(
                f"  Content validation FAIL: no page matches {item.expected_content!r} "
                f"for query '{item.query[:50]}'"
            )
            continue

        new_item = QAItem(
            query=item.query,
            expected_pages=matching_pages,
            doc_type=item.doc_type,
            retrieval_type=item.retrieval_type,
            note=item.note,
            difficulty=item.difficulty,
            query_length=item.query_length,
            expected_content=item.expected_content,
            distractor_keywords=item.distractor_keywords,
        )
        resolved.append(new_item)

    if verbose and warnings:
        for w in warnings:
            print(w)

    return resolved, warnings


# ===========================================================================
# Prose indexing (test harness only)
# ============================================================================

def _split_prose_chunks(text: str, max_chars: int = 600) -> list[str]:
    """Split page text into chunks suitable for Row nodes and dense indexing."""
    text = text.strip()
    if not text:
        return []
    paras = re.split(r"\n\s*\n+", text)
    chunks: list[str] = []
    for para in paras:
        p = para.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        sents = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) + 1 <= max_chars:
                buf = f"{buf} {s}".strip() if buf else s
            else:
                if buf:
                    chunks.append(buf)
                buf = s if len(s) <= max_chars else s[:max_chars]
        if buf:
            chunks.append(buf)
    return chunks


def _add_prose_row_nodes(G, pages: list[tuple[int, str]]) -> None:
    """
    Add pseudo-Row nodes from full page text so step5 build_index()
    encodes narrative prose (Row/Image only). Page nodes are not indexed.
    """
    for page_num, page_text in pages:
        text = (page_text or "").strip()
        if not text:
            continue
        page_node = f"page::{page_num}"
        if not G.has_node(page_node):
            G.add_node(
                page_node,
                type="Page",
                label=f"Page {page_num}",
                page=page_num,
                prose=text,
                text=text,
            )
        chunks = _split_prose_chunks(text, max_chars=600)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 40:
                continue
            row_id = f"prose::p{page_num}::c{i}"
            if G.has_node(row_id):
                continue
            G.add_node(
                row_id,
                type="Row",
                label=f"Prose {i + 1}",
                entity="",
                page=page_num,
                section="Document text",
                table_caption="",
                text=chunk,
            )
            G.add_edge(row_id, page_node, relation="on_page")


# ===========================================================================
# Extraction PDF builder (lined tables for pdfplumber)
# ===========================================================================

def _draw_table_grid_fitz(page, rows: list[list[str]], x0: float, y0: float,
                          col_w: float, row_h: float) -> float:
    """Draw a simple grid with PyMuPDF and place cell text. Returns y below table."""
    if fitz is None:
        return y0
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    for r in rows:
        while len(r) < ncols:
            r.append("")
    for ri in range(nrows):
        for ci in range(ncols):
            rect = fitz.Rect(
                x0 + ci * col_w,
                y0 + ri * row_h,
                x0 + (ci + 1) * col_w,
                y0 + (ri + 1) * row_h,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.4)
            txt = (rows[ri][ci] or "")[:40]
            tx = rect.x0 + 3
            ty = rect.y0 + 11
            page.insert_text((tx, ty), txt, fontsize=7, fontname="helv", color=(0, 0, 0))
    return y0 + nrows * row_h + 12


def _build_extraction_pdf(pages: list[tuple[int, str]], path: Path) -> bool:
    """
    Write a PDF with visible table borders so pdfplumber can find tables.
    Uses the same logical content as TYPE_*_PAGES but draws pipe-rows as grids.
    """
    if fitz is None:
        return False
    doc = fitz.open()
    for page_num, content in pages:
        page = doc.new_page(width=595, height=842)
        y = 48
        page.insert_text((50, y), f"Page {page_num}", fontsize=12, fontname="helv")
        y += 22
        lines = content.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "|" in line:
                block: list[str] = []
                while i < len(lines) and "|" in lines[i]:
                    block.append(lines[i].strip())
                    i += 1
                if len(block) >= 2:
                    rows = [[c.strip() for c in ln.split("|")] for ln in block]
                    ncols = max(len(r) for r in rows)
                    col_w = min(95.0, max(40.0, (490.0) / max(ncols, 1)))
                    row_h = 28 if ncols <= 6 else 22
                    y = _draw_table_grid_fitz(page, rows, 50, y, col_w, row_h)
                else:
                    page.insert_text((50, y), line, fontsize=9, fontname="helv")
                    y += 14
                    i += 1
            else:
                if line:
                    page.insert_text((50, y), line[:100], fontsize=9, fontname="helv")
                    y += 14
                i += 1
            if y > 780:
                break
    doc.save(str(path))
    doc.close()
    return True


# ===========================================================================
# Synthetic PDF builder
# ===========================================================================

def _build_synthetic_pdf(pages: list[tuple[int, str]], path: Path) -> None:
    """Write a minimal text-based PDF using PyMuPDF."""
    if fitz is not None:
        doc = fitz.open()
        font = fitz.Font("helv")
        for page_num, content in pages:
            page = doc.new_page(width=595, height=842)
            y = 60
            page.insert_text(
                (50, y), f"Page {page_num}", fontsize=14,
                fontname="helv", color=(0, 0, 0),
            )
            y += 24
            for line in content.strip().splitlines():
                if y > 800:
                    break
                page.insert_text(
                    (50, y), line.strip(), fontsize=10,
                    fontname="helv", color=(0, 0, 0),
                )
                y += 14
        doc.save(str(path))
        doc.close()
    else:
        # Fallback: write a plain .txt file and mock the rest
        path.with_suffix(".txt").write_text(
            "\n\n".join(f"=== Page {p} ===\n{c}" for p, c in pages)
        )
        # Create an empty pdf placeholder
        path.write_bytes(b"%PDF-1.4\n1 0 obj\n<</Type/Catalog>>\nendobj\n"
                          b"xref\n0 2\n0000000000 65535 f\n0000000009 00000 n\n"
                          b"trailer\n<</Root 1 0 R/Size 2>>\nstartxref\n47\n%%EOF\n")


# ===========================================================================
# Evaluation helpers
# ===========================================================================

@dataclass
class EvalResult:
    query:           str
    expected_pages:  list[int]
    got_pages:       list[int]
    rank_of_correct: Optional[int]   # None if not found
    doc_type:        str
    retrieval_type:  str
    note:            str
    difficulty:      str = "medium"
    query_length:    str = "medium"
    is_distractor:   bool = False     # True if top-1 is a plausible wrong page


def _mrr(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def _precision_at_k(got: list[int], expected: list[int], k: int = 5) -> float:
    hits = sum(1 for p in got[:k] if p in expected)
    return hits / min(k, len(got)) if got else 0.0


def _recall_at_k(got: list[int], expected: list[int], k: int = 5) -> float:
    hits = sum(1 for p in got[:k] if p in expected)
    return hits / len(expected) if expected else 0.0


def _first_correct_rank(got: list[int], expected: list[int]) -> Optional[int]:
    for i, p in enumerate(got, 1):
        if p in expected:
            return i
    return None


def _ndcg_at_k(got: list[int], expected: list[int], k: int = 5) -> float:
    """Normalised Discounted Cumulative Gain @ k."""
    def _dcg(ranks: list[int], expected_set: set[int], k: int) -> float:
        score = 0.0
        for i, p in enumerate(ranks[:k]):
            rel = 1.0 if p in expected_set else 0.0
            score += rel / math.log2(i + 2)
        return score
    if not got or not expected:
        return 0.0
    expected_set = set(expected)
    dcg  = _dcg(got, expected_set, k)
    ideal = sorted([1.0] * min(len(expected), k) + [0.0] * max(0, k - len(expected)),
                    reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ===========================================================================
# Main evaluation runner
# ===========================================================================

def build_test_kg(pages: list[tuple[int, str]], pdf_path: Path):
    """
    Build KG + index from a synthetic document.
    Returns (G, index) or (None, None) on failure.
    """
    import sys
    # Add App and Program to path
    app_dir     = _HERE / "App"
    program_dir = _HERE / "Program"
    for d in (str(_HERE), str(app_dir), str(program_dir)):
        if d not in sys.path:
            sys.path.insert(0, d)

    try:
        extracted = []
        pages_data: dict[int, dict] = {}
        for page_num, text in pages:
            pages_data[page_num] = {"prose": text, "blocks": []}
            lines  = [l.strip() for l in text.splitlines() if "|" in l]
            if len(lines) >= 2:
                raw_table = [row.split("|") for row in lines]
                raw_table = [[cell.strip() for cell in row] for row in raw_table]
                extracted.append({
                    "page":            page_num,
                    "bbox":            (50, 80, 545, 400),
                    "section_heading": f"Section page {page_num}",
                    "caption_above":   f"Table on page {page_num}",
                    "table":           raw_table,
                    "notes_below":     "",
                    "footnotes":       {},
                })

        if not extracted:
            extracted.append({
                "page":            1,
                "bbox":            (50, 80, 545, 400),
                "section_heading": "Introduction",
                "caption_above":   "",
                "table":           [["Item", "Value"], ["—", "—"]],
                "notes_below":     "",
                "footnotes":       {},
            })

        extracted[0]["_toc"]   = {"toc_page_numbers": set(), "entries": []}
        extracted[0]["_pages"] = pages_data

        from Program.step4_graph_construction import build_knowledge_graph
        G = build_knowledge_graph(extracted)

        import networkx as nx
        for page_num, text in pages:
            pn = f"page::{page_num}"
            if G.has_node(pn):
                G.nodes[pn]["text"]  = text
                G.nodes[pn]["prose"] = text
            else:
                G.add_node(pn, type="Page", label=f"Page {page_num}",
                           page=page_num, prose=text, text=text)

        _add_prose_row_nodes(G, pages)

        from Program.step5_query_helpers import build_index
        index = build_index(G)
        return G, index

    except Exception as e:
        print(f"  KG build failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None, None


def build_kg_from_extracted_pdf(pdf_path: Path):
    """
    Build KG + index using real pdfplumber extraction (step1) + graph + index.
    Prose rows are added from pdfplumber full-page text so retrieval matches
    the PDF content (not injected pipe-split tables).
    """
    import sys
    app_dir     = _HERE / "App"
    program_dir = _HERE / "Program"
    for d in (str(_HERE), str(app_dir), str(program_dir)):
        if d not in sys.path:
            sys.path.insert(0, d)

    try:
        import pdfplumber
        from Program.step1_pdf_extraction import extract_tables_with_context
        from Program.step4_graph_construction import build_knowledge_graph
        from Program.step5_query_helpers import build_index

        extracted = extract_tables_with_context(str(pdf_path))
        if not extracted:
            return None, None
        G = build_knowledge_graph(extracted)
        with pdfplumber.open(pdf_path) as pdf:
            pages_tuples = [
                (i + 1, page.extract_text() or "")
                for i, page in enumerate(pdf.pages)
            ]
        _add_prose_row_nodes(G, pages_tuples)
        index = build_index(G)
        return G, index
    except Exception as e:
        print(f"  KG build error: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None, None


def build_kg_for_pipeline(
    doc_type:     str,
    pages:        list[tuple[int, str]],
    *,
    document_title: str = "",
    pipeline:     str = "mock",
    pdf_path:     Path | None = None,
    qa_items:     list[QAItem] | None = None,
    verbose:      bool = False,
) -> tuple[object, dict, list[QAItem] | None]:
    """Build a KG + index for a given pipeline/doc_type.

    Returns (G, index, resolved_qa_items).  resolved_qa_items is only
    meaningful for the 'real' pipeline (where content validation resolves
    expected_pages); for other pipelines it is returned as-is.
    Returns (None, None, qa_items) on failure.
    """
    title = document_title.strip() or f"Type {doc_type} document"

    if pipeline == "real" and pdf_path is not None:
        if qa_items:
            _status(f"Validating content for {title}...")
            qa_items, val_warnings = _validate_extraction(
                pdf_path, qa_items, verbose=verbose,
            )
            if verbose:
                for w in val_warnings:
                    print(w)
            resolved = sum(1 for q in qa_items if q.expected_pages)
            _status(f"Content validation done: {resolved}/{len(qa_items)} queries resolved")

        _status(f"Building KG from real PDF: {title}...")
        G, index = build_kg_from_extracted_pdf(pdf_path)
        if G is None or index is None:
            _status(f"KG build FAILED for {title}")
            return None, None, qa_items
        _status(f"KG ready - {G.number_of_nodes()} nodes, "
                f"{len(index.get('ids', []))} indexed")
    elif pipeline == "extract":
        _status(f"Building extraction PDF for Type {doc_type}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            synth_pdf = Path(tmpdir) / f"test_type_{doc_type}.pdf"
            if not _build_extraction_pdf(pages, synth_pdf):
                _status(f"PDF build failed for Type {doc_type} (PyMuPDF missing?)")
                return None, None, qa_items
            _status(f"Extracting + building KG for Type {doc_type}...")
            G, index = build_kg_from_extracted_pdf(synth_pdf)
        if G is None or index is None:
            _status(f"KG build FAILED for Type {doc_type}")
            return None, None, qa_items
        _status(f"KG ready - {G.number_of_nodes()} nodes, "
                f"{len(index.get('ids', []))} indexed")
    else:
        _status(f"Building mock KG for Type {doc_type}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            synth_pdf = Path(tmpdir) / f"test_type_{doc_type}.pdf"
            _build_synthetic_pdf(pages, synth_pdf)
            G, index = build_test_kg(pages, synth_pdf)
        if G is None or index is None:
            _status(f"KG build FAILED for Type {doc_type}")
            return None, None, qa_items
        _status(f"KG ready - {G.number_of_nodes()} nodes, "
                f"{len(index.get('ids', []))} indexed")

    return G, index, qa_items


def run_queries(
    G,
    index:        dict,
    qa_items:     list[QAItem],
    *,
    document_title: str = "",
    use_generative: bool = False,
    top_k:          int = 5,
    verbose:        bool = False,
    on_query_done:  Callable[[], None] | None = None,
) -> list[EvalResult]:
    """Run queries against an already-built KG + index."""
    title = document_title.strip() or "document"

    @contextlib.contextmanager
    def _quiet_context():
        if verbose:
            yield
            return
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield

    results: list[EvalResult] = []

    try:
        from Program.step5_query_helpers import query as kg_query
    except Exception as e:
        print(f"  Cannot import query helper: {e}")
        return []

    _status(f"Running {len(qa_items)} queries for {title}...")
    for item in qa_items:
        if not item.expected_pages:
            if on_query_done is not None:
                on_query_done()
            continue

        try:
            with _quiet_context():
                hits = kg_query(
                    G, index, item.query,
                    top_k=top_k,
                    use_generative=use_generative,
                )
            got_pages = [r["page"] for r in hits if r.get("page")]
        except Exception as e:
            got_pages = []
            if verbose:
                print(f"  Query error for '{item.query}': {e}")

        rank = _first_correct_rank(got_pages, item.expected_pages)

        is_distractor = (
            len(got_pages) > 0
            and got_pages[0] not in item.expected_pages
        )

        res = EvalResult(
            query=item.query,
            expected_pages=item.expected_pages,
            got_pages=got_pages,
            rank_of_correct=rank,
            doc_type=item.doc_type,
            retrieval_type=item.retrieval_type,
            note=item.note,
            difficulty=item.difficulty,
            query_length=item.query_length,
            is_distractor=is_distractor,
        )
        results.append(res)

        if verbose:
            status = "OK" if rank else "XX"
            dist   = " [DISTRACTOR]" if is_distractor else ""
            print(f"  {status} [{item.retrieval_type:8s}|{item.difficulty:6s}] "
                  f"{item.query[:50]:50s}"
                  f" | expected={item.expected_pages} got={got_pages[:5]}{dist}")

        if on_query_done is not None:
            on_query_done()

    return results


def print_report(
    all_results: list[EvalResult],
    *,
    title: str = "",
    include_overall: bool = True,
) -> dict:
    """Print a full accuracy report and return summary metrics."""
    print(f"\n{'='*70}")
    print(f"  ACCURACY REPORT  ({len(all_results)} total queries)")
    if title:
        print(f"  {title}")
    print(f"{'='*70}")

    def _group(results, key_fn):
        groups: dict[str, list[EvalResult]] = {}
        for r in results:
            k = key_fn(r)
            groups.setdefault(k, []).append(r)
        return groups

    def _metrics(group: list[EvalResult]) -> dict:
        n       = len(group)
        mrr     = sum(_mrr(r.rank_of_correct) for r in group) / n
        p1      = sum(_precision_at_k(r.got_pages, r.expected_pages, 1) for r in group) / n
        p5      = sum(_precision_at_k(r.got_pages, r.expected_pages, 5) for r in group) / n
        r5      = sum(_recall_at_k(r.got_pages, r.expected_pages, 5) for r in group) / n
        ndcg5   = sum(_ndcg_at_k(r.got_pages, r.expected_pages, 5) for r in group) / n
        found   = sum(1 for r in group if r.rank_of_correct)
        dist    = sum(1 for r in group if r.is_distractor)
        dist_rt = dist / n if n else 0.0
        return {"n": n, "found": found, "mrr": mrr,
                "p@1": p1, "p@5": p5, "r@5": r5, "nDCG@5": ndcg5,
                "distractor_rate": dist_rt}

    overall = _metrics(all_results)
    if include_overall:
        print(f"\n  Overall     n={overall['n']:3d}  found={overall['found']:3d}"
              f"  MRR={overall['mrr']:.3f}  P@1={overall['p@1']:.3f}"
              f"  P@5={overall['p@5']:.3f}  R@5={overall['r@5']:.3f}"
              f"  nDCG@5={overall['nDCG@5']:.3f}"
              f"  distractor={overall['distractor_rate']:.1%}")

    # By document type
    doc_col_w = max((len(s) for s in DOC_TYPE_LABELS.values()), default=20) + 2
    print(f"\n  {'Document':<{doc_col_w}} {'N':>4} {'Found':>6} {'MRR':>7} "
          f"{'P@1':>6} {'P@5':>6} {'R@5':>6} {'nDCG':>6} {'Dist%':>6}")
    print(f"  {'-'*100}")
    type_summary = {}
    for doc_type, group in sorted(_group(all_results, lambda r: r.doc_type).items()):
        m = _metrics(group)
        type_summary[doc_type] = m
        label = DOC_TYPE_LABELS.get(doc_type, f"Type {doc_type}")
        print(f"  {label:<{doc_col_w}} {m['n']:4d}  {m['found']:6d}  "
              f"{m['mrr']:6.3f}  {m['p@1']:5.3f}  {m['p@5']:5.3f}  {m['r@5']:5.3f}  "
              f"{m['nDCG@5']:5.3f}  {m['distractor_rate']:5.1%}")

    # By retrieval type
    print(f"\n  {'Retrieval Type':<12} {'N':>4} {'Found':>6} {'MRR':>7} "
          f"{'P@1':>6} {'P@5':>6} {'R@5':>6} {'nDCG':>6}")
    print(f"  {'-'*65}")
    ret_summary = {}
    for ret_type, group in sorted(_group(all_results, lambda r: r.retrieval_type).items()):
        m = _metrics(group)
        ret_summary[ret_type] = m
        print(f"  {ret_type:<12}  {m['n']:4d}  {m['found']:6d}  "
              f"{m['mrr']:6.3f}  {m['p@1']:5.3f}  {m['p@5']:5.3f}  {m['r@5']:5.3f}  "
              f"{m['nDCG@5']:5.3f}")

    # By difficulty
    print(f"\n  {'Difficulty':<12} {'N':>4} {'Found':>6} {'MRR':>7} "
          f"{'P@1':>6} {'nDCG':>6}")
    print(f"  {'-'*50}")
    diff_summary = {}
    for diff, group in sorted(_group(all_results, lambda r: r.difficulty).items()):
        m = _metrics(group)
        diff_summary[diff] = m
        print(f"  {diff:<12}  {m['n']:4d}  {m['found']:6d}  "
              f"{m['mrr']:6.3f}  {m['p@1']:5.3f}  {m['nDCG@5']:5.3f}")

    # By query length
    print(f"\n  {'Query Length':<12} {'N':>4} {'Found':>6} {'MRR':>7} "
          f"{'P@1':>6}")
    print(f"  {'-'*42}")
    len_summary = {}
    for ql, group in sorted(_group(all_results, lambda r: r.query_length).items()):
        m = _metrics(group)
        len_summary[ql] = m
        print(f"  {ql:<12}  {m['n']:4d}  {m['found']:6d}  "
              f"{m['mrr']:6.3f}  {m['p@1']:5.3f}")

    # Weaknesses: failed queries
    failed = [r for r in all_results if not r.rank_of_correct]
    if failed:
        print(f"\n  Missed queries ({len(failed)}):")
        for r in failed:
            dt = DOC_TYPE_LABELS.get(r.doc_type, r.doc_type)
            dist = " [DISTRACTOR]" if r.is_distractor else ""
            print(f"    [{dt} / {r.retrieval_type} / {r.difficulty}] "
                  f"{r.query[:55]}"
                  f" -- expected {r.expected_pages}{dist}")

    print(f"\n{'='*70}\n")

    return {
        "overall":       overall,
        "by_type":       type_summary,
        "by_retrieval":  ret_summary,
        "by_difficulty": diff_summary,
        "by_query_len":  len_summary,
        "failed":        [(r.query, r.expected_pages) for r in failed],
    }


def print_overall_comparison(
    combined_report: dict[str, dict],
    mode_order: list[tuple[str, str]],
) -> None:
    """Print overall metrics for all modes and pipelines in one block (after full runs)."""
    rows: list[tuple[str, dict]] = []
    for run_key, run_label in mode_order:
        if run_key not in combined_report:
            continue
        payload = combined_report[run_key]
        if isinstance(payload, dict) and any(
            k in payload for k in ("mock", "extract", "real")
        ):
            for pk in ("mock", "extract", "real"):
                sub = payload.get(pk)
                if isinstance(sub, dict) and "overall" in sub:
                    pipe_suffix = {"mock": "Mock", "extract": "Extract",
                                   "real": "Real PDF"}[pk]
                    rows.append((f"{run_label} | {pipe_suffix}", sub["overall"]))
        elif isinstance(payload, dict) and "overall" in payload:
            rows.append((run_label, payload["overall"]))
    if len(rows) <= 1:
        return
    print(f"\n{'='*70}")
    print("  OVERALL (comparison)")
    print(f"{'='*70}")
    run_w = 56
    print(f"\n  {'Run':<{run_w}} {'n':>3} {'found':>6} {'MRR':>7} "
          f"{'P@1':>6} {'P@5':>6} {'R@5':>6} {'nDCG':>6} {'Dist%':>6}")
    print(f"  {'-' * (run_w + 42)}")
    for label, o in rows:
        label = label[:run_w] if len(label) > run_w else label
        ndcg = o.get("nDCG@5", 0.0)
        dist = o.get("distractor_rate", 0.0)
        print(
            f"  {label:<{run_w}} {o['n']:3d} {o['found']:6d} "
            f"{o['mrr']:7.3f} {o['p@1']:6.3f} {o['p@5']:6.3f} {o['r@5']:6.3f} "
            f"{ndcg:6.3f} {dist:5.1%}"
        )
    print(f"\n{'='*70}\n")


def save_report(summary: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Report saved -> {path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="KG retrieval accuracy evaluation")
    parser.add_argument("--type",
                        choices=["E","F","G","H","I"],
                        help="Run only this document type")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-query detail")
    parser.add_argument(
        "--mode",
        choices=["adaptive", "both", "hyde_ce", "baseline"],
        default="adaptive",
        help="adaptive: single run using current query behavior (case-by-case HyDE/CE). "
             "Legacy aliases: both/hyde_ce/baseline map to adaptive.",
    )
    parser.add_argument("--no-generative", action="store_true",
                        help="Deprecated: forces baseline (HyDE off, CE off)")
    parser.add_argument("--top-k",         type=int, default=5,
                        help="Number of results to retrieve (default 5)")
    parser.add_argument(
        "--pipeline",
        choices=["real"],
        default="real",
        help="Real PDFs only (E-I).",
    )
    parser.add_argument("--save",          type=str, default="",
                        help="Path to save JSON report")
    args = parser.parse_args()

    eval_mode = args.mode
    if args.no_generative:
        _status("`--no-generative` is deprecated and now treated as adaptive mode.")
        eval_mode = "adaptive"
    if eval_mode in ("both", "hyde_ce", "baseline"):
        _status(f"`--mode {eval_mode}` is deprecated and now treated as adaptive mode.")
        eval_mode = "adaptive"

    # Single-mode evaluation only: HyDE / reranking are now applied per-query.
    mode_runs: list[tuple[str, bool, str]] = [
        ("adaptive", True, "Adaptive retrieval (case-by-case HyDE/CE)")
    ]

    SYNTHETIC_TYPES = {}

    # Determine which types to run
    if args.type:
        types_to_run = [args.type]
    else:
        types_to_run = list(REAL_PDF_SOURCES.keys())

    # Determine which pipeline runs to use
    pipeline_runs: list[tuple[str, str]] = [("real", "Real PDF documents (E-I)")]

    combined_report: dict[str, dict] = {}

    # Count queries
    synth_queries = 0
    real_queries = sum(
        len([q for q in REAL_QA_DATASET if q.doc_type == t])
        for t in types_to_run if t in REAL_PDF_SOURCES
    )
    total_per_pipeline_mode = real_queries
    synth_pipeline_count = 0
    real_pipeline_count  = sum(1 for pk, _ in pipeline_runs if pk == "real")
    total_queries = real_queries * real_pipeline_count * len(mode_runs)
    progress_done = [0]
    _last_progress_time = [time.time()]

    def _on_query_done() -> None:
        progress_done[0] += 1
        now = time.time()
        pct = (100.0 * progress_done[0] / total_queries) if total_queries else 100.0
        # Always update on first, last, every 5th, or every 10 seconds
        since_last = now - _last_progress_time[0]
        is_milestone = (progress_done[0] % 5 == 0
                        or progress_done[0] == total_queries
                        or since_last >= 10.0)
        if is_milestone:
            _last_progress_time[0] = now
            _status(f"Query {progress_done[0]}/{total_queries}  ({pct:.0f}%)")
        else:
            print(
                f"\r  [{_elapsed():>7s}] Query {progress_done[0]}/{total_queries}  ({pct:.0f}%)",
                end="",
                flush=True,
            )

    show_progress = (total_queries > 0) and (not args.verbose)
    if show_progress:
        global _t0
        _t0 = time.time()
        _status(f"Starting evaluation - {total_queries} total queries")

    # ── Phase 1: Build all KGs (once per pipeline+doc_type) ────────────
    # Cache: kg_cache[(pipe_key, doc_type)] = (G, index, resolved_qa_items)
    kg_cache: dict[tuple[str, str], tuple] = {}
    failed_types: set[str] = set()

    print(f"\n{'='*60}", flush=True)
    _status("PHASE 1: Building all knowledge graphs")
    print(f"{'='*60}", flush=True)

    for pipe_key, pipe_title in pipeline_runs:
        _status(f"Pipeline: {pipe_title}")

        if pipe_key in ("mock", "extract"):
            for doc_type in types_to_run:
                if doc_type not in SYNTHETIC_TYPES:
                    continue
                fail_key = f"{pipe_key}:{doc_type}"
                pages, description = SYNTHETIC_TYPES[doc_type]
                items = [q for q in QA_DATASET if q.doc_type == doc_type]
                try:
                    G, index, resolved = build_kg_for_pipeline(
                        doc_type, pages,
                        document_title=description,
                        pipeline=pipe_key,
                        qa_items=items,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    _status(f"Type {doc_type} FAILED: {e} -- skipping")
                    failed_types.add(fail_key)
                    continue
                if G is None or index is None:
                    failed_types.add(fail_key)
                    continue
                kg_cache[(pipe_key, doc_type)] = (G, index, resolved or items)

        elif pipe_key == "real":
            real_types = [t for t in types_to_run if t in REAL_PDF_SOURCES]
            if not real_types and not args.type:
                real_types = list(REAL_PDF_SOURCES.keys())

            for doc_type in real_types:
                fail_key = f"real:{doc_type}"
                pdf_path = _get_cached_pdf(doc_type, verbose=args.verbose)
                if pdf_path is None:
                    _status(f"Skipping Type {doc_type}: cached PDF not available")
                    failed_types.add(fail_key)
                    continue

                items = [q for q in REAL_QA_DATASET if q.doc_type == doc_type]
                description = DOC_TYPE_LABELS.get(doc_type, f"Type {doc_type}")
                try:
                    G, index, resolved = build_kg_for_pipeline(
                        doc_type, [],
                        document_title=description,
                        pipeline="real",
                        pdf_path=pdf_path,
                        qa_items=items,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    _status(f"Type {doc_type} FAILED: {e} -- skipping")
                    failed_types.add(fail_key)
                    continue
                if G is None or index is None:
                    failed_types.add(fail_key)
                    continue
                kg_cache[("real", doc_type)] = (G, index, resolved or items)

    built_count = len(kg_cache)
    _status(f"PHASE 1 complete: {built_count} KGs built, "
            f"{len(failed_types)} failed")

    # ── Phase 2: Run queries against cached KGs ─────────────────────────
    print(f"\n{'='*60}", flush=True)
    _status("PHASE 2: Running queries")
    print(f"{'='*60}", flush=True)

    for run_key, use_gen, run_label in mode_runs:
        mode_payload: dict[str, dict] = {}
        print(f"\n{'-'*60}", flush=True)
        _status(f"MODE: {run_label}")
        print(f"{'-'*60}", flush=True)

        for pipe_key, pipe_title in pipeline_runs:
            all_results: list[EvalResult] = []
            _status(f"Pipeline: {pipe_title}")

            if pipe_key in ("mock", "extract"):
                for doc_type in types_to_run:
                    if doc_type not in SYNTHETIC_TYPES:
                        continue
                    cache_key = (pipe_key, doc_type)
                    if cache_key not in kg_cache:
                        continue
                    G, index, items = kg_cache[cache_key]
                    description = SYNTHETIC_TYPES[doc_type][1]
                    try:
                        results = run_queries(
                            G, index, items,
                            document_title=description,
                            use_generative=use_gen,
                            top_k=args.top_k,
                            verbose=args.verbose,
                            on_query_done=_on_query_done if show_progress else None,
                        )
                    except Exception as e:
                        _status(f"Queries for Type {doc_type} FAILED: {e}")
                        continue
                    all_results.extend(results)

            elif pipe_key == "real":
                real_types = [t for t in types_to_run if t in REAL_PDF_SOURCES]
                if not real_types and not args.type:
                    real_types = list(REAL_PDF_SOURCES.keys())

                for doc_type in real_types:
                    cache_key = ("real", doc_type)
                    if cache_key not in kg_cache:
                        continue
                    G, index, items = kg_cache[cache_key]
                    description = DOC_TYPE_LABELS.get(doc_type, f"Type {doc_type}")
                    try:
                        results = run_queries(
                            G, index, items,
                            document_title=description,
                            use_generative=use_gen,
                            top_k=args.top_k,
                            verbose=args.verbose,
                            on_query_done=_on_query_done if show_progress else None,
                        )
                    except Exception as e:
                        _status(f"Queries for Type {doc_type} FAILED: {e}")
                        continue
                    all_results.extend(results)

            if all_results:
                if show_progress:
                    print()
                summary = print_report(
                    all_results,
                    title=f"{run_label} | {pipe_title}",
                    include_overall=True,
                )
                mode_payload[pipe_key] = summary
                if show_progress:
                    print()

        if mode_payload:
            combined_report[run_key] = mode_payload

    if show_progress:
        print()

    if combined_report:
        print_overall_comparison(
            combined_report,
            [(m[0], m[2]) for m in mode_runs],
        )
        save_path = args.save or str(_HERE / "test_report.json")
        save_report(combined_report, Path(save_path))

    _status(f"All done - {progress_done[0]} queries evaluated.")


if __name__ == "__main__":
    main()
