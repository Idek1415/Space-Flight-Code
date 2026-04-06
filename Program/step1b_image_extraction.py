"""
Step 1b — Image Extraction
===========================
Extracts all raster images from a digital PDF using PyMuPDF (fitz).
Each image is saved to a temp folder and returned with its page number
and bounding box so it can be spatially linked to nearby tables.

Returned structure
------------------
[
    {
        "image_id":  "p3_img0",
        "page":      3,
        "bbox":      (x0, y0, x1, y1),   # position on page in points
        "path":      "/tmp/oring_imgs/p3_img0.png",
        "width":     400,
        "height":    300,
    },
    ...
]

Dependencies:
    pip install pymupdf
"""

import os
import fitz  # PyMuPDF

from App.console_progress import status_line, status_line_done


def extract_images(pdf_path, output_dir=None, min_dimension=50):
    """
    Extract all raster images from a PDF.

    Args:
        pdf_path:      Path to the PDF file.
        output_dir:    Folder to save extracted images. Defaults to a
                       temp folder beside the PDF.
        min_dimension: Skip images smaller than this in either dimension
                       (filters out decorative icons, rule lines, etc.)

    Returns:
        List of image dicts with id, page, bbox, path, width, height.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(pdf_path)),
            "oring_extracted_images"
        )
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    results = []

    for page_num in range(total_pages):
        pct = round((page_num + 1) / total_pages * 100)
        status_line(f"  Extracting images... {pct}% (page {page_num + 1}/{total_pages})")

        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            width  = base_image["width"]
            height = base_image["height"]

            # Skip tiny decorative images
            if width < min_dimension or height < min_dimension:
                continue

            ext      = base_image["ext"]
            img_data = base_image["image"]
            image_id = f"p{page_num + 1}_img{img_idx}"
            filename = f"{image_id}.{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(img_data)

            # Get bounding box of this image on the page
            bbox = _get_image_bbox(page, xref)

            results.append({
                "image_id": image_id,
                "page":     page_num + 1,
                "bbox":     bbox,
                "path":     filepath,
                "width":    width,
                "height":   height,
            })

    doc.close()
    status_line_done(f"  Extracting images... 100% ({total_pages}/{total_pages} pages) — done.")
    print(f"  Found {len(results)} image(s) → {output_dir}")
    return results


def _get_image_bbox(page, xref):
    """
    Return the bounding box (x0, y0, x1, y1) of an image on the page
    by matching its xref in the page's image placement list.
    Falls back to (0, 0, 0, 0) if not found.
    """
    for item in page.get_image_info(hashes=False):
        if item.get("xref") == xref:
            r = item.get("bbox") or item.get("rect")
            if r:
                return tuple(r)
    return (0, 0, 0, 0)
