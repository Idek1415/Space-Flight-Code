O-Ring Knowledge Graph — Installation & Usage Guide
====================================================

REQUIREMENTS
------------
- Windows 10 / 11 (64-bit)
- Python 3.10 or later  →  https://www.python.org/downloads/
  (check "Add Python to PATH" during install)
- An NVIDIA GPU is recommended for fast KG building and captioning,
  but the application runs on CPU as well.


INSTALLATION
------------
1.  Copy the entire OringKG\ folder to wherever you want to keep it
    (e.g. C:\Users\You\OringKG\).

2.  Double-click launcher.bat.
    On the very first run this will:
      a) Detect your GPU and install the matching PyTorch build.
      b) Install all other Python packages (PyQt6, pdfplumber, etc.).
      c) Download NLTK WordNet data (used for negation-aware search).
      d) Generate the application icon.
    This first-run setup takes 2–10 minutes depending on your connection.
    Subsequent launches skip setup and start the app in a few seconds.

3.  (Optional) Create a desktop shortcut:
        python create_shortcut.py
    This places an OringKG shortcut on your Desktop so you can
    double-click it to launch without navigating to the folder.


USAGE
-----

Opening a PDF
  • Click File > Open PDF … (or the "Open PDF …" button).
  • A file-selection dialog opens.  Navigate to your datasheet.
  • If the file is outside the application folder you will be asked for
    permission — this is shown every time for transparency.
  • The PDF immediately appears in the right-hand viewer.
  • If a saved Knowledge Graph exists for this file you are offered the
    option to load it instantly (skipping processing).

Processing options (shown once a PDF is loaded)
  • Include images — extracts raster images and optionally captions them
    with BLIP. Disable for faster builds on text-heavy documents.
  • Caption model — Small (~900 MB, faster) or Large (~1.9 GB, better).
  • Skip captions — extracts images and links them spatially to nearby
    tables without loading the caption model.
  • Query embedder — Small (80 MB / all-MiniLM) or Large (420 MB /
    all-mpnet-base-v2, recommended for better search accuracy).

Building the Knowledge Graph
  Click "Build Knowledge Graph".  Progress is shown in the build log.
  Typical times:
    Text-only, small PDF (50 pages)   →   1–3 min
    With images                       →   add 2–5 min per 50 images
    Large handbook (400+ pages)        →   5–15 min

Querying
  • Type any natural-language question in the Query box and press Enter.
  • The system returns up to 5 results, ranked by relevance.
  • Each result shows its page number as a blue link — click it to jump
    to that page in the PDF viewer on the right.
  • Negation is understood: "not for steam", "avoid FKM", "non-hydraulic"
    will penalise matching results and surface alternatives.
  • Query history within a session is used as context.  Click
    "Clear history" to reset and start a fresh line of inquiry.

Saving & loading Knowledge Graphs
  • After building, click "Save KG".  The graph and embeddings are
    stored in OringKG\saved_kgs\<filename>_kg\ (inside the app folder).
  • On the next run, when you open the same PDF the app offers to load
    the saved version immediately.
  • Click File > View saved KGs … (or "View saved KGs …") to see all
    saved graphs, load one for a different PDF, or delete old ones.

PDF viewer controls
  • ◀ Prev / Next ▶   — navigate one page at a time.
  • Fit width          — scale the current page to fill the panel.
  • + / −              — zoom in or out manually.
  • Page links in results scroll the viewer automatically.


FOLDER STRUCTURE
----------------
OringKG\
  launcher.bat              Double-click to launch
  launcher.py               First-run setup logic
  app.py                    PyQt6 GUI application
  create_shortcut.py        Creates a Desktop shortcut
  requirements.txt          Python package list
  README.txt                This file
  console_progress.py       Shared progress helpers
  device_config.py          GPU/CPU selection
  step1_pdf_extraction.py   Table extraction + TOC detection
  step1b_image_extraction.py Image extraction (PyMuPDF)
  step1c_image_captioning.py BLIP image captioning
  step1d_block_classifier.py Table vs. prose classification
  step2_table_parsing.py    Column/row/unit parsing
  step3_normalization.py    Engineering term canonicalization
  step4_graph_construction.py NetworkX KG builder
  step5_query_helpers.py    NLP index + negation-aware search
  step6_visualization.py    Matplotlib graph visualizer
  step7_persistence.py      Save/load/delete CLI helpers
  assets\
    icon.ico                Application icon (auto-generated)
  saved_kgs\
    <name>_kg\              One folder per saved document
      graph.pkl             NetworkX graph
      index_meta.json       Node IDs, types, texts
      embeddings.pt         Sentence-transformer embeddings
      metadata.json         PDF hash, timestamps, model names


TROUBLESHOOTING
---------------
"Python not found"
  Install Python 3.10+ from python.org and ensure "Add to PATH" is
  checked.  Then re-run launcher.bat.

App fails to start with ImportError
  Run launcher.bat again — it will detect missing packages and reinstall.

PDF opens but Knowledge Graph build fails
  Check the build log for the error message.  Common causes:
    - PDF is password-protected (unsupported)
    - PDF contains only scanned images with no extractable text
      (use OCR pre-processing externally before loading)
    - Out of memory: reduce image batch size or use --cpu

GPU not detected / running on CPU
  Install the CUDA build of PyTorch manually:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
  Replace cu121 with your CUDA version (check with: nvidia-smi).

"Saved KG found but PDF has changed"
  The PDF file was modified after the KG was saved.
  Click "Build Knowledge Graph" to rebuild.

Delete a saved KG
  File > View saved KGs …, select the entry, click "Delete selected".

CONTACT / SOURCE
----------------
This application is built on open-source libraries.  See README.txt for
the complete dependency list.
