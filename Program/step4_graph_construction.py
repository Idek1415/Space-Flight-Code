"""
Step 4 — Knowledge Graph Construction
======================================
Improvements in this version:
  - Section hierarchy: Section nodes carry a 'level' attribute (1=H1, 2=H2).
    'has_subsection' edges link parent sections to child sections when a
    lower-level heading appears within a parent's page span.
  - Entity deduplication: after graph construction, Row entities that
    normalise to the same canonical string are linked with 'alias_of' edges.
    This prevents graph fragmentation when surface forms vary ("NBR", "Nitrile",
    "NBR-70" all resolve to the same canonical entity).
  - HyPE support: if use_generative=True, step1e.generate_hype_queries is
    called after graph construction to enrich Row node text fields.
"""

import networkx as nx
from Program.step2_table_parsing import parse_oring_table
from Program.step3_normalization import normalize
from App.console_progress import status_line, status_line_done

PROXIMITY_THRESHOLD = 150


def build_knowledge_graph(extracted_tables, image_records=None,
                          use_generative=False):
    G = nx.MultiDiGraph()

    # Separate metadata from table entries
    pages_data: dict[int, dict] = {}
    table_entries: list[dict]   = []
    for entry in extracted_tables:
        if "_pages" in entry:
            pages_data = entry["_pages"]
        if "page" in entry and "table" in entry:
            table_entries.append(entry)

    G.graph["pages_data"]     = pages_data
    G.graph["use_generative"] = use_generative

    # ── Page nodes ────────────────────────────────────────────────────────
    all_pages = sorted({e["page"] for e in table_entries} | set(pages_data.keys()))
    for p in all_pages:
        prose = pages_data.get(p, {}).get("prose", "")
        G.add_node(f"page::{p}", type="Page", label=f"Page {p}",
                   page=p, prose=prose, text=prose or f"Page {p}")

    # ── Continuation edges ────────────────────────────────────────────────
    page_to_sections: dict[int, set] = {}
    for e in table_entries:
        h = e.get("section_heading") or ""
        page_to_sections.setdefault(e["page"], set()).add(h)

    sorted_pages = sorted(page_to_sections)
    for i in range(len(sorted_pages) - 1):
        p1, p2 = sorted_pages[i], sorted_pages[i + 1]
        if p2 == p1 + 1:
            shared = page_to_sections[p1] & page_to_sections[p2]
            if shared - {"", "Unknown section"}:
                G.add_edge(f"page::{p1}", f"page::{p2}",
                           relation="continues_to")

    # ── Section hierarchy ─────────────────────────────────────────────────
    # Collect all unique headings with their first appearance page
    heading_first_page: dict[str, int] = {}
    for e in sorted(table_entries, key=lambda x: x["page"]):
        h = e.get("section_heading") or "Unknown section"
        if h not in heading_first_page:
            heading_first_page[h] = e["page"]

    # Infer hierarchy: a heading that is shorter and appears before another
    # on the same page is treated as its parent.  We track a simple stack.
    heading_stack: list[str] = []   # stack of (heading, level)

    def _heading_level(text: str) -> int:
        """Heuristic: fewer words + all-caps or short → higher level."""
        words = text.split()
        if len(words) <= 2:
            return 1
        if len(words) <= 4:
            return 2
        return 3

    for h in heading_first_page:
        level = _heading_level(h)
        section_id = f"section::{h}"
        if not G.has_node(section_id):
            G.add_node(section_id, type="Section", label=h, level=level)

        # Pop stack until we find a plausible parent (lower level number)
        while heading_stack and _heading_level(heading_stack[-1]) >= level:
            heading_stack.pop()

        if heading_stack:
            parent_id = f"section::{heading_stack[-1]}"
            if G.has_node(parent_id):
                G.add_edge(parent_id, section_id, relation="has_subsection")

        heading_stack.append(h)

    # ── Tables / Rows / Cells / Triples ───────────────────────────────────
    table_bbox_lookup: dict[int, list] = {}

    for t_idx, table_data in enumerate(table_entries):
        page_num   = table_data["page"]
        table_id   = f"table_{t_idx+1}_p{page_num}"
        heading    = table_data.get("section_heading") or "Unknown section"
        caption    = table_data.get("caption_above") or table_id
        page_node  = f"page::{page_num}"
        section_id = f"section::{heading}"

        if not G.has_node(section_id):
            G.add_node(section_id, type="Section", label=heading,
                       level=_heading_level(heading))

        G.add_edge(section_id, page_node, relation="on_page")

        G.add_node(table_id, type="Table", label=caption,
                   page=page_num, section=heading,
                   bbox=str(table_data["bbox"]))
        G.add_edge(table_id, section_id, relation="belongs_to")
        G.add_edge(table_id, page_node,  relation="on_page")

        table_bbox_lookup.setdefault(page_num, []).append(
            (table_id, table_data["bbox"], section_id)
        )

        # Conditions
        condition_nodes: dict[str, str] = {}
        for marker, desc in table_data.get("footnotes", {}).items():
            cond_id = f"condition::{desc[:60]}"
            if not G.has_node(cond_id):
                G.add_node(cond_id, type="Condition", label=desc, marker=marker)
            condition_nodes[marker] = cond_id
            G.add_edge(cond_id, table_id,  relation="referenced_in")
            G.add_edge(cond_id, page_node, relation="on_page")

        matrix = parse_oring_table(table_data.get("table", []))
        if not matrix:
            continue

        # Columns
        col_node_ids: dict[int, str] = {}
        for col in matrix["columns"]:
            col_id     = f"{table_id}::col_{col['index']}"
            unit_label = f" ({col['unit']})" if col["unit"] else ""
            G.add_node(col_id, type="Column",
                       label=f"{col['header']}{unit_label}",
                       header=col["header"], unit=col["unit"] or "",
                       col_index=col["index"], page=page_num)
            G.add_edge(table_id, col_id,    relation="has_column")
            G.add_edge(col_id,   page_node, relation="on_page")
            col_node_ids[col["index"]] = col_id

        # Rows
        page_prose = pages_data.get(page_num, {}).get("prose", "")
        for row in matrix["rows"]:
            row_id = f"{table_id}::row_{row['index']}"
            subject_value = next((c["value"] for c in row["cells"] if c["value"]), "")

            cell_texts = []
            for cell in row["cells"]:
                if cell["value"]:
                    u = f" {cell['unit']}" if cell["unit"] else ""
                    cell_texts.append(f"{cell['header']}: {cell['value']}{u}")

            row_text = (
                f"{heading} — {caption} | "
                + ", ".join(cell_texts)
                + (f" | context: {page_prose[:200]}" if page_prose else "")
            )

            G.add_node(row_id, type="Row", label=f"Row {row['index'] + 1}",
                       entity=subject_value, page=page_num,
                       section=heading, table_caption=caption, text=row_text)
            G.add_edge(table_id, row_id,    relation="has_row")
            G.add_edge(row_id,   page_node, relation="on_page")

            all_raw = " ".join(c["raw"] for c in row["cells"])
            for marker, cond_id in condition_nodes.items():
                if marker in all_raw:
                    G.add_edge(row_id, cond_id, relation="subject_to")

            # Cells + Triples
            for cell in row["cells"]:
                if not cell["value"]:
                    continue
                col_node_id = col_node_ids.get(cell["col_index"])
                cell_id     = f"{row_id}::col_{cell['col_index']}"
                norm_value  = normalize(cell["value"])
                unit_label  = f" ({cell['unit']})" if cell["unit"] else ""

                G.add_node(cell_id, type="Cell",
                           label=f"{norm_value}{unit_label}",
                           value=norm_value, raw_value=cell["raw"],
                           header=cell["header"], unit=cell["unit"] or "",
                           col_index=cell["col_index"], page=page_num)
                G.add_edge(row_id,  cell_id,   relation="has_cell")
                G.add_edge(cell_id, page_node, relation="on_page")
                if col_node_id:
                    G.add_edge(cell_id, col_node_id, relation="in_column")

                if cell["col_index"] > 0 and subject_value:
                    norm_pred = normalize(cell["header"])
                    triple_id = (
                        f"triple::{normalize(subject_value)}"
                        f"::{norm_pred}::{normalize(cell['value'])}"
                    )
                    if not G.has_node(triple_id):
                        u = f" {cell['unit']}" if cell["unit"] else ""
                        G.add_node(triple_id, type="Triple",
                                   label=f"{subject_value} → {cell['header']}: "
                                         f"{cell['value']}{u}",
                                   subject=subject_value,
                                   predicate=cell["header"],
                                   object=cell["value"],
                                   unit=cell["unit"] or "", page=page_num)
                    G.add_edge(row_id,    triple_id,  relation="has_triple")
                    G.add_edge(triple_id, page_node,  relation="on_page")
                    if col_node_id:
                        G.add_edge(triple_id, col_node_id, relation="predicate_of")

    # ── Images ────────────────────────────────────────────────────────────
    if image_records:
        total = len(image_records)
        for i, img in enumerate(image_records):
            pct      = round((i + 1) / total * 100)
            status_line(f"  Adding images to graph... {pct}% ({i + 1}/{total})")
            caption  = img.get("caption", "")
            image_id = f"image::{img['image_id']}"
            page_num = img["page"]
            img_bbox = img["bbox"]
            img_text = (f"Image on page {page_num}: {caption}"
                        if caption else f"Image on page {page_num}")
            page_node = f"page::{page_num}"

            G.add_node(image_id, type="Image",
                       label=caption or img["image_id"],
                       page=page_num, bbox=str(img_bbox),
                       path=img["path"], caption=caption, text=img_text)
            G.add_edge(image_id, page_node, relation="on_page")

            if not _link_image_to_table(G, image_id, page_num,
                                        img_bbox, table_bbox_lookup):
                pass  # already connected to page_node

        status_line_done(f"  Adding images to graph... 100% ({total}/{total}) — done.")

    # ── Entity deduplication ──────────────────────────────────────────────
    _deduplicate_entities(G)

    return G


def _deduplicate_entities(G) -> None:
    """
    Find Row entity strings that normalise to the same canonical form
    and link them with 'alias_of' edges.  Prevents graph fragmentation
    when the same entity appears under different surface forms across pages.
    """
    canonical_to_nodes: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "Row":
            continue
        entity = data.get("entity", "").strip()
        if not entity:
            continue
        canonical = normalize(entity)
        canonical_to_nodes.setdefault(canonical, []).append(node_id)

    merged = 0
    for canonical, node_ids in canonical_to_nodes.items():
        if len(node_ids) < 2:
            continue
        # First node is the canonical representative; others are aliases
        canonical_node = node_ids[0]
        for alias_node in node_ids[1:]:
            if not G.has_edge(alias_node, canonical_node):
                G.add_edge(alias_node, canonical_node,
                           relation="alias_of", canonical=canonical)
                merged += 1

    if merged:
        print(f"  Entity dedup: {merged} alias edges added.", flush=True)


def _link_image_to_table(G, image_id, page_num, img_bbox, table_bbox_lookup):
    candidates = table_bbox_lookup.get(page_num, [])
    if not candidates:
        return False
    img_mid_y = (img_bbox[1] + img_bbox[3]) / 2
    best_table_id = min(candidates,
                        key=lambda c: abs(img_mid_y - (c[1][1] + c[1][3]) / 2),
                        default=None)
    if best_table_id is None:
        return False
    table_id, t_bbox, _ = best_table_id
    if abs(img_mid_y - (t_bbox[1] + t_bbox[3]) / 2) <= PROXIMITY_THRESHOLD:
        G.add_edge(image_id, table_id, relation="illustrates")
        return True
    return False
