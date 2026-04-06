"""
O-Ring PDF → Knowledge Graph
=============================
Main entry point. Imports and runs each step in sequence.

Dependencies:
    pip install pdfplumber pymupdf networkx matplotlib sentence-transformers transformers pillow torch

Usage:
    python main.py your_datasheet.pdf
    python main.py --demo
    python main.py
        (no args: uses Desktop Catalog_O-Ring-Handbook_PTD5705-EN.pdf if found, else demo)
    python main.py your_datasheet.pdf --no-images
    python main.py your_datasheet.pdf --rebuild
    python main.py your_datasheet.pdf --save-graph graph.png
    python main.py your_datasheet.pdf --export-json graph.json
"""

import os

# Hugging Face / tqdm download bars often print many lines in IDEs; keep one-line UX for our own progress.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
import json
import argparse
from pathlib import Path

# Project now uses sibling folders: Program/ and App/.
PROGRAM_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = PROGRAM_DIR.parent
APP_DIR = PROJECT_DIR / "App"
for _p in (str(PROJECT_DIR), str(PROGRAM_DIR), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Program.step1_pdf_extraction    import extract_tables_with_context
from Program.step1b_image_extraction import extract_images
from Program.step1c_image_captioning import caption_images, configure_caption_model
from Program.step4_graph_construction import build_knowledge_graph
from Program.step5_query_helpers     import (
    build_index,
    query,
    format_results,
    build_effective_query,
    configure_embedder_model,
    configure_embedder_from_saved,
)
from Program.step6_visualization     import visualize_graph
from Program.step7_persistence       import save_kg, load_kg, delete_kg

# Default handbook filename (same file users often keep on Desktop)
HANDBOOK_PDF_NAME = "Catalog_O-Ring-Handbook_PTD5705-EN.pdf"


def _stdin_is_tty() -> bool:
    """
    Safe TTY check that won't crash if stdin is a custom stream object.
    """
    fn = getattr(sys.stdin, "isatty", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _ask_choice(prompt: str, valid: tuple[str, ...], default: str) -> str:
    valid_set = {v.lower() for v in valid}
    while True:
        raw = input(f"{prompt} [{'/'.join(valid)}] (default: {default}): ").strip().lower()
        if not raw:
            return default
        if raw in valid_set:
            return raw
        print(f"  Please choose one of: {', '.join(valid)}")


def _prompt_run_options(default_include_images: bool) -> tuple[bool, str, str]:
    """
    Prompt once at startup for image inclusion + model sizes.
    Returns: (include_images, caption_size, embedder_size)
    """
    print("\n=== Runtime options (build from PDF) ===")
    if not _stdin_is_tty():
        include_images = default_include_images
        caption_size = "small"
        embedder_size = "large"
        print("  Non-interactive run detected; using defaults.")
        return include_images, caption_size, embedder_size

    include_default = "y" if default_include_images else "n"
    include = _ask_choice("Include images in KG build?", ("y", "n"), include_default) == "y"
    caption_size = "small"
    if include:
        caption_size = _ask_choice(
            "Caption model size? (BLIP)",
            ("small", "large"),
            "small",
        )
    embedder_size = _ask_choice(
        "Query embedder size? (SentenceTransformer)",
        ("small", "large"),
        "large",
    )
    return include, caption_size, embedder_size


def _prompt_try_load_saved(pdf_path: str | None, *, rebuild: bool) -> bool:
    """
    Ask whether to load a saved KG + embeddings for the given PDF path.
    Returns False if there is no PDF path, ``--rebuild`` was passed, or the user declines.
    """
    if rebuild or not pdf_path:
        return False
    print("\n=== Load saved KG ===")
    if not _stdin_is_tty():
        return True
    raw = input(
        f"Load saved KG + embeddings for this PDF if present?\n  {pdf_path}\n"
        f"[Y/n]: "
    ).strip().lower()
    if raw in ("n", "no"):
        return False
    return True


def _physical_desktop_dir_win():
    """
    Actual Desktop folder on Windows (respects OneDrive / known-folder redirection).
    Unlike Path.home()/Desktop, this matches what Explorer shows as Desktop.
    """
    if sys.platform != "win32":
        return None
    import ctypes

    buf = ctypes.create_unicode_buffer(32_767)
    # CSIDL_DESKTOPDIRECTORY — file-system folder that backs the user's Desktop
    CSIDL_DESKTOPDIRECTORY = 0x10
    hr = ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, 0, buf)
    if hr != 0:
        return None
    p = Path(buf.value)
    return p if p.is_dir() else None


def _resolve_pdf_path(raw: str) -> Path:
    """Normalize user/guard input: strip whitespace, drop outer quotes, expand ~, resolve."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return Path(s).expanduser().resolve()


def _handbook_pdf_candidates() -> list[Path]:
    """Where the handbook might live if copied to Desktop (plain or OneDrive-backed)."""
    home = Path.home()
    candidates: list[Path] = []
    desk = _physical_desktop_dir_win()
    if desk is not None:
        candidates.append(desk / HANDBOOK_PDF_NAME)
    candidates.extend(
        [
            home / "Desktop" / HANDBOOK_PDF_NAME,
            home / "OneDrive" / "Desktop" / HANDBOOK_PDF_NAME,
            home / "OneDrive - Personal" / "Desktop" / HANDBOOK_PDF_NAME,
        ]
    )
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _find_handbook_on_desktop() -> Path | None:
    for p in _handbook_pdf_candidates():
        if p.is_file():
            return p.resolve()
    return None


def _extract_pdf_or_exit(pdf_path: Path, *, cli_argument: str | None) -> list:
    """
    Echo resolved path to stdout (IDE run configs often hide stderr), verify file, extract.
    cli_argument: original argv string if from CLI, else None.
    """
    pdf_path = pdf_path.resolve()
    print("\n--- PDF path diagnostics (always printed) ---")
    if cli_argument is not None:
        print(f"  CLI argument: {cli_argument!r}")
    print(f"  Resolved path: {pdf_path}")
    print(f"  exists: {pdf_path.exists()}  |  is_file: {pdf_path.is_file()}")
    print("---")

    if not pdf_path.is_file():
        print(
            "Error: PDF not found before opening.\n"
            f"  Resolved: {pdf_path}",
            flush=True,
        )
        sys.exit(1)

    try:
        extracted = extract_tables_with_context(str(pdf_path))
    except OSError as e:
        print(str(e), flush=True)
        sys.exit(1)

    print(f"  Found {len(extracted)} table(s).\n")
    return extracted


# ---------------------------------------------------------------------------
# Demo data (synthetic O-ring tables — no PDF required)
# ---------------------------------------------------------------------------

DEMO_TABLES = [
    {
        "page": 1,
        "bbox": (40, 100, 560, 300),
        "section_heading": "AS568 Standard O-Ring Dimensions",
        "caption_above": "Table 1 — Dash sizes, NBR compound, 70 Shore A",
        "table": [
            ["Dash No.", "ID (in)", "ID (mm)", "Cross-section (in)", "Cross-section (mm)", "Material"],
            ["-001*",   "0.029",   "0.74",    "0.040",              "1.02",               "NBR"],
            ["-002*",   "0.042",   "1.07",    "0.050",              "1.27",               "NBR"],
            ["-003",    "0.056",   "1.42",    "0.060",              "1.52",               "NBR"],
            ["-004",    "0.070",   "1.78",    "0.070",              "1.78",               "NBR"],
            ["-005",    "0.101",   "2.57",    "0.070",              "1.78",               "Viton"],
        ],
        "notes_below": "* Not recommended for dynamic applications. Temperature range: -40°F to +250°F.",
        "footnotes": {"*": "Not recommended for dynamic applications. Temperature range: -40°F to +250°F."},
    },
    {
        "page": 2,
        "bbox": (40, 120, 560, 320),
        "section_heading": "Material Compatibility",
        "caption_above": "Table 2 — Pressure ratings by compound",
        "table": [
            ["Material", "Max Pressure (PSI)", "Max Temp (°F)", "Min Temp (°F)", "Application"],
            ["NBR",      "3000",               "250",           "-40",           "Hydraulic"],
            ["FKM",      "3000",               "400",           "-15",           "Chemical"],
            ["EPDM",     "1500",               "300",           "-65",           "Steam"],
            ["Silicone", "500",                "450",           "-175",          "High temp"],
        ],
        "notes_below": "",
        "footnotes": {},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NLP for PDFs → Knowledge Graph")
    parser.add_argument("pdf", nargs="?", help="Path to PDF file")
    parser.add_argument("--demo",        action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--no-images",   action="store_true", help="Skip image extraction and captioning entirely")
    parser.add_argument("--no-captions", action="store_true", help="Extract and link images but skip loading the BLIP caption model")
    parser.add_argument("--delete-kg",   action="store_true", help="Delete any saved KG for the given PDF and exit")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Skip loading saved KG/embeddings and rebuild from the PDF.",
    )
    parser.add_argument("--save-graph",  metavar="PATH", help="Save graph image to file")
    parser.add_argument("--export-json", metavar="PATH", help="Export graph nodes/edges as JSON")
    args = parser.parse_args()

    # --delete-kg: remove saved data and exit (no other prompts)
    if args.delete_kg:
        target = args.pdf or str(_find_handbook_on_desktop() or "")
        if target:
            delete_kg(target)
        else:
            print("  --delete-kg requires a PDF path argument.")
        return

    # Resolve PDF path early so we can load saved KG before build prompts
    pdf_path_str: str | None = None
    if args.pdf:
        pdf_path_str = str(_resolve_pdf_path(args.pdf).resolve())
    elif not args.demo:
        handbook_early = _find_handbook_on_desktop()
        if handbook_early is not None:
            pdf_path_str = str(handbook_early)

    loaded_from_disk = False
    G = None
    index = None
    load_meta: dict | None = None

    if (
        pdf_path_str
        and not args.demo
        and _prompt_try_load_saved(pdf_path_str, rebuild=args.rebuild)
    ):
        loaded = load_kg(pdf_path_str)
        if loaded:
            G, index, load_meta = loaded
            loaded_from_disk = True
            configure_embedder_from_saved(
                load_meta.get("model_name") if load_meta else None
            )
            print("\n  Saved KG loaded — skipping questions about images, captions, and embedder size.")

    if not loaded_from_disk:
        include_images, caption_size, embedder_size = _prompt_run_options(
            default_include_images=not args.no_images
        )
        configure_embedder_model(embedder_size)
        if include_images:
            configure_caption_model(caption_size)
        elif args.no_images is False:
            args.no_images = True
    else:
        include_images = False

    # Step 1: Ingest tables (skip when graph was loaded)
    image_records: list = []
    extracted = None

    if G is not None:
        pass
    elif args.demo:
        print("\n=== DEMO MODE (synthetic O-ring data) ===\n")
        extracted = DEMO_TABLES
    elif args.pdf:
        print("\nExtracting tables from CLI path...")
        extracted = _extract_pdf_or_exit(_resolve_pdf_path(args.pdf), cli_argument=args.pdf)

        if include_images:
            print("Extracting images...")
            image_records = extract_images(pdf_path_str)  # type: ignore[arg-type]
            if image_records and not args.no_captions:
                print(f"\nGenerating image captions...")
                image_records = caption_images(image_records)
            elif image_records:
                print(f"  Skipping captions (--no-captions). {len(image_records)} image(s) linked by position only.")
            print()
    elif pdf_path_str:
        print("\nExtracting tables (no CLI path — using handbook on Desktop)...")
        extracted = _extract_pdf_or_exit(Path(pdf_path_str), cli_argument=None)

        if include_images:
            print("Extracting images...")
            image_records = extract_images(pdf_path_str)
            if image_records and not args.no_captions:
                print(f"\nGenerating image captions...")
                image_records = caption_images(image_records)
            elif image_records:
                print(f"  Skipping captions (--no-captions). {len(image_records)} image(s) linked by position only.")
            print()
    else:
        tried = "\n    ".join(str(p) for p in _handbook_pdf_candidates())
        print(
            "\n=== DEMO MODE (synthetic O-ring data) ===\n"
            f"No PDF argument and {HANDBOOK_PDF_NAME!r} not found.\n"
            "  Searched:\n    "
            f"{tried}\n"
            "  Pass the file explicitly, e.g.:\n"
            '    python main.py "C:\\Users\\YOU\\Desktop\\Catalog_O-Ring-Handbook_PTD5705-EN.pdf"\n',
            flush=True,
        )
        extracted = DEMO_TABLES

    # Steps 2–4: Parse, normalize, build graph (skip if loaded from save)
    if G is None:
        print("Building knowledge graph...")
        G = build_knowledge_graph(extracted, image_records=image_records or None)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Step 5: Build NLP index (rebuild if embeddings missing)
    print("\nBuilding NLP index...")
    if index is None or index.get("embeddings") is None:
        index = build_index(G)

    # Offer to save if this was freshly built from a real PDF
    if pdf_path_str and not args.demo and not loaded_from_disk:
        try:
            if _stdin_is_tty():
                ans = input("\nSave KG and embeddings for next run? [y/N]: ").strip().lower()
            else:
                ans = "n"
            if ans == "y":
                import step5_query_helpers as _s5
                import step1c_image_captioning as _s1c
                save_kg(pdf_path_str, G, index,
                        model_name=_s5.MODEL_NAME,
                        caption_model=_s1c.MODEL_NAME)
        except Exception as e:
            print(f"  Could not save KG: {e}")

#    example_queries = [
#        "NBR o-ring suitable for hydraulic applications",
#        "small cross section o-ring in nitrile",
#        "high temperature resistant material above 400 degrees",
#    ]
#
#    for q in example_queries:
#        results = query(G, index, q, top_k=3)
#        print(format_results(results, q))

    # Interactive query loop
    query_history = []
    print("=" * 60)
    print("Enter your own queries below (type exit/quit/q to stop):")
    while True:
        try:
            user_query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_query.lower() in ("exit", "quit", "q"):
            break
        query_history.append(user_query)
        # Same embedder as retrieval picks related priors, then current is weighted
        effective_query = build_effective_query(query_history, window=2)
        results = query(G, index, effective_query, top_k=5)
        print(format_results(results, user_query))
        

    # Optional JSON export
    if args.export_json:
        export_data = {
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes()],
            "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
        }
        with open(args.export_json, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nGraph exported to: {args.export_json}")

    # Step 6: Visualize
    #print("\nGenerating graph visualization...")
    #visualize_graph(G, output_path=args.save_graph)


if __name__ == "__main__":
    main()
