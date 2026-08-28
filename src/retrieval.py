"""Document access for a single ConvFinQA record.

Each conversation in ConvFinQA is grounded in exactly one financial document
(pre_text + table + post_text). These documents are short (a few paragraphs
plus a small table, median 3x5 cells, max 10x19), so the agent is simply
given the whole thing directly rather than chunked/retrieved via embeddings.
"""

from src.models import Document


def table_to_markdown(table: dict[str, dict[str, float | str | int]]) -> str:
    """Render the document table as a markdown table for the agent to read directly."""
    columns: list[str] = []
    for row in table.values():
        for col in row.keys():
            if col not in columns:
                columns.append(col)

    header = "| (row) | " + " | ".join(columns) + " |"
    separator = "| --- | " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row_label, row in table.items():
        cells = [str(row.get(col, "")) for col in columns]
        rows.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def build_document_text(doc: Document) -> str:
    """Return the complete document as one block of text."""
    parts = []

    if doc.pre_text.strip():
        parts.append("## PRE-TEXT\n")
        parts.append(doc.pre_text)

    parts.append("\n\n## TABLE\n")
    parts.append(table_to_markdown(doc.table))

    if doc.post_text.strip():
        parts.append("\n\n## POST-TEXT\n")
        parts.append(doc.post_text)

    return "\n".join(parts)
