"""
Step 1c — Image Captioning
============================
Generates detailed natural language descriptions for each extracted
image by running BLIP with multiple targeted prompts and concatenating
the results into one rich description.

Prompts cover (domain-agnostic, suitable for any PDF figure or photo):
  - Overall scene or layout
  - Visible text, labels, and captions
  - Numbers, dates, and other quantitative detail
  - Main objects, people, or visual elements
  - Charts, plots, or diagrams and their labels (when present)

Model choice
------------
BLIP-base (~900 MB) runs on GPU.
Swap MODEL_NAME for the large variant if VRAM allows:
    "Salesforce/blip-image-captioning-base"   (~900 MB, default)
    "Salesforce/blip-image-captioning-large"  (~1.9 GB)

Dependencies:
    pip install transformers pillow torch
"""

import os
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from App.console_progress import status_line, status_line_done
from App.device_config import get_torch_device_str


MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = get_torch_device_str()

# Parallel disk I/O while batching GPU work (multiple images per BLIP forward)
_CAPTION_LOAD_WORKERS = min(8, max(2, (os.cpu_count() or 4)))
# Images per BLIP forward (VRAM trade-off; reduce if OOM)
_CAPTION_GPU_BATCH_SIZE = 4

_processor = None
_model     = None

CAPTION_MODELS = {
    "small": "Salesforce/blip-image-captioning-base",
    "large": "Salesforce/blip-image-captioning-large",
}


def configure_caption_model(size: str) -> str:
    """
    Configure caption model by size ("small" or "large") and clear cache so
    the new model is loaded on next call.
    """
    global MODEL_NAME, _processor, _model
    key = size.strip().lower()
    if key not in CAPTION_MODELS:
        raise ValueError(f"Unknown caption model size: {size!r}")
    MODEL_NAME = CAPTION_MODELS[key]
    _processor = None
    _model = None
    return MODEL_NAME

# Each prompt targets a different aspect of the image (BLIP conditional prefix).
# Results are concatenated into one description per image.
_CAPTION_PROMPTS = [
    "a photograph or illustration showing",
    "the text, labels, and captions visible in this image include",
    "any numbers, dates, or quantitative details shown are",
    "the main objects, people, or visual elements in this image are",
    "the type of chart, plot, or diagram shown, including any axes or legends, is",
]


def _get_model():
    """Lazy-load the BLIP model and processor onto GPU when CUDA is available."""
    global _processor, _model
    if _model is None:
        print(f"  Loading vision model ({MODEL_NAME}) on {DEVICE.upper()}...")
        if DEVICE == "cpu":
            print(
                "  (CUDA not available — install a CUDA build of PyTorch to caption on GPU.)",
                flush=True,
            )
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model = _model.to(DEVICE)
        _model.eval()
    return _processor, _model


def _load_rgb(path: str):
    """Return RGB PIL image or None on failure."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _load_images_threaded(records: list) -> list:
    """Load images in parallel from disk (I/O-bound)."""
    paths = [r["path"] for r in records]
    workers = min(_CAPTION_LOAD_WORKERS, len(paths))
    if workers <= 1:
        return [_load_rgb(p) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_load_rgb, paths))


def _caption_one_pil(processor, model, record, image: Image.Image) -> None:
    """Single-image path (fallback or batch size 1)."""
    parts = []
    for prompt in _CAPTION_PROMPTS:
        inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=1.3,
            )
        text = processor.decode(output[0], skip_special_tokens=True).strip()
        if text:
            parts.append(text)
    record["caption"] = _join_unique(parts)


def _caption_batch_pil(processor, model, records: list, images: list) -> None:
    """Run BLIP once per prompt over a batch of images (GPU-bound)."""
    n = len(records)
    parts_per = [[] for _ in range(n)]
    for prompt in _CAPTION_PROMPTS:
        inputs = processor(
            images=images,
            text=[prompt] * n,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=1.3,
            )
        for i in range(n):
            text = processor.decode(output[i], skip_special_tokens=True).strip()
            if text:
                parts_per[i].append(text)
    for i, record in enumerate(records):
        record["caption"] = _join_unique(parts_per[i])


def caption_images(image_records):
    """
    Generate detailed multi-prompt captions for a list of image dicts.

    Loads images with a thread pool and runs BLIP in GPU batches to reduce
    wall-clock time versus one image at a time.

    Updates each dict in-place with a "caption" key containing the
    concatenated description from all prompts.

    Args:
        image_records: List of image dicts from extract_images()

    Returns:
        The same list with "caption" added to each dict.
    """
    if not image_records:
        return image_records

    processor, model = _get_model()
    total = len(image_records)
    bs = max(1, _CAPTION_GPU_BATCH_SIZE)

    for start in range(0, total, bs):
        batch = image_records[start : start + bs]
        pct = round(min((start + len(batch)) / total * 100, 100))
        status_line(
            f"  Captioning... {pct}% ({start + len(batch)}/{total} images)"
        )

        images = _load_images_threaded(batch)
        if all(img is not None for img in images):
            try:
                _caption_batch_pil(processor, model, batch, images)
            except Exception as e:
                print(
                    f"\n  Warning: batch caption failed ({e}); retried per image.",
                    flush=True,
                )
                for rec, img in zip(batch, images):
                    try:
                        _caption_one_pil(processor, model, rec, img)
                    except Exception as e2:
                        rec["caption"] = ""
                        print(
                            f"\n  Warning: could not caption {rec['image_id']}: {e2}",
                            flush=True,
                        )
        else:
            for rec, img in zip(batch, images):
                try:
                    if img is None:
                        rec["caption"] = ""
                        print(
                            f"\n  Warning: could not load image {rec['image_id']}",
                            flush=True,
                        )
                    else:
                        _caption_one_pil(processor, model, rec, img)
                except Exception as e:
                    rec["caption"] = ""
                    print(
                        f"\n  Warning: could not caption {rec['image_id']}: {e}",
                        flush=True,
                    )

    status_line_done(f"  Captioning... 100% ({total}/{total} images) — done.")
    return image_records


def _join_unique(parts):
    """
    Join caption parts, dropping any part that is a near-duplicate
    of a part already included (simple word-overlap check).
    """
    kept = []
    seen_words = set()
    for part in parts:
        words = set(part.lower().split())
        # Skip if more than 70% of its words are already covered
        if kept and len(words & seen_words) / max(len(words), 1) > 0.7:
            continue
        kept.append(part)
        seen_words |= words
    return " | ".join(kept)
