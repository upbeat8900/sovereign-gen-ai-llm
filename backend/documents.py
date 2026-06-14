import json
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, List, Optional

from fastapi import HTTPException
import pymupdf
import pymupdf4llm

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def _public_document(row: dict) -> dict:
    public = dict(row)
    metadata = public.pop("metadata_json", None)
    public["metadata"] = json.loads(metadata) if metadata else None
    return public


def load_documents(connection, conversation_id: int) -> List[dict]:
    rows = connection.execute(
        """
        SELECT * FROM documents
        WHERE conversation_id = ? AND archived_at IS NULL
        ORDER BY updated_at DESC, id DESC
        """,
        (conversation_id,),
    ).fetchall()
    return [_public_document(dict(row)) for row in rows]


def document_or_404(connection, conversation_id: int, document_id: int) -> dict:
    row = connection.execute(
        """
        SELECT * FROM documents
        WHERE id = ? AND conversation_id = ? AND archived_at IS NULL
        """,
        (document_id, conversation_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return _public_document(dict(row))


def create_document(
    connection,
    conversation_id: int,
    *,
    title: str,
    content_markdown: str,
    kind: str = "uploaded",
    source_filename: Optional[str] = None,
    source_media_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    title = (title or "Untitled document").strip() or "Untitled document"
    content = (content_markdown or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    if len(content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Document exceeds maximum size")
    cursor = connection.execute(
        """
        INSERT INTO documents (
            conversation_id, title, kind, content_markdown, source_filename,
            source_media_type, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            title,
            kind,
            content,
            source_filename,
            source_media_type,
            json.dumps(metadata) if metadata else None,
        ),
    )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    return document_or_404(connection, conversation_id, cursor.lastrowid)


def update_document(
    connection,
    conversation_id: int,
    document_id: int,
    *,
    title: Optional[str] = None,
    content_markdown: Optional[str] = None,
) -> dict:
    existing = document_or_404(connection, conversation_id, document_id)
    next_title = (title if title is not None else existing["title"]).strip() or "Untitled document"
    next_content = content_markdown if content_markdown is not None else existing["content_markdown"]
    if not (next_content or "").strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    connection.execute(
        """
        UPDATE documents
        SET title = ?, content_markdown = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND conversation_id = ?
        """,
        (next_title, next_content, document_id, conversation_id),
    )
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    return document_or_404(connection, conversation_id, document_id)


def archive_document(connection, conversation_id: int, document_id: int) -> None:
    document_or_404(connection, conversation_id, document_id)
    connection.execute(
        """
        UPDATE documents
        SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND conversation_id = ?
        """,
        (document_id, conversation_id),
    )


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^\w.\- ]+", "_", (filename or "document").strip())
    return cleaned[:180] or "document"


def _converter_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _page_image_metadata(document: pymupdf.Document) -> List[dict[str, Any]]:
    images: List[dict[str, Any]] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            rects = []
            for rect in page.get_image_rects(xref):
                rects.append(
                    {
                        "x0": round(rect.x0, 2),
                        "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2),
                        "y1": round(rect.y1, 2),
                    }
                )
            images.append(
                {
                    "page": page_index + 1,
                    "index": image_index,
                    "xref": xref,
                    "width": image[2],
                    "height": image[3],
                    "bits_per_component": image[4],
                    "colorspace": image[5],
                    "name": image[7] if len(image) > 7 else None,
                    "filter": image[8] if len(image) > 8 else None,
                    "rects": rects,
                }
            )
    return images


def _pdf_metadata(document: pymupdf.Document) -> dict[str, Any]:
    images = _page_image_metadata(document)
    return {
        "converter": {
            "name": "pymupdf4llm",
            "version": _converter_version("pymupdf4llm"),
            "options": {
                "ignore_images": True,
                "write_images": False,
                "embed_images": False,
            },
        },
        "images": {
            "count": len(images),
            "handling": "metadata_only_no_interpretation",
            "items": images,
        },
    }


def _convert_pdf_to_markdown(filename: str, raw_bytes: bytes) -> tuple[str, dict[str, Any]]:
    try:
        document = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read PDF upload") from exc
    try:
        metadata = _pdf_metadata(document)
        markdown = pymupdf4llm.to_markdown(
            document,
            ignore_images=True,
            write_images=False,
            embed_images=False,
            filename=sanitize_filename(filename),
            show_progress=False,
        ).strip()
        if not markdown:
            image_count = metadata["images"]["count"]
            markdown = (
                f"# {sanitize_filename(filename)}\n\n"
                "No extractable text was found in this PDF.\n\n"
                f"Image metadata was stored for {image_count} image(s); image content was not interpreted."
            )
        return markdown, metadata
    finally:
        document.close()


def normalize_upload_to_markdown(filename: str, media_type: str, raw_bytes: bytes) -> tuple[str, str, Optional[dict]]:
    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Upload exceeds maximum size")
    lowered = (filename or "").lower()
    media = (media_type or "").lower()
    if media in {"text/markdown", "text/x-markdown"} or lowered.endswith(".md"):
        return raw_bytes.decode("utf-8", errors="replace"), "text/markdown", None
    if media in {"text/plain"} or lowered.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="replace")
        return f"# {sanitize_filename(filename)}\n\n{text}", "text/plain", None
    if media in {"application/pdf"} or lowered.endswith(".pdf"):
        markdown, metadata = _convert_pdf_to_markdown(filename, raw_bytes)
        return markdown, "application/pdf", metadata
    raise HTTPException(
        status_code=400,
        detail="Unsupported upload format. Use Markdown (.md), plain text (.txt), or PDF (.pdf).",
    )


def search_documents(documents: List[dict], query: Optional[str] = None) -> List[dict]:
    if not query or not query.strip():
        return documents
    needle = query.strip().lower()
    results = []
    for document in documents:
        haystack = f"{document.get('title', '')}\n{document.get('content_markdown', '')}".lower()
        if needle in haystack:
            results.append(document)
    return results


def document_list_summary(documents: List[dict]) -> str:
    if not documents:
        return "(none)"
    lines = []
    for document in documents:
        kind = document.get("kind") or "uploaded"
        lines.append(f"- [{document['id']}] {document['title']} ({kind})")
    return "\n".join(lines)
