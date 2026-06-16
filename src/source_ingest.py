from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.aimooc_schema import SourceItem, SourceManifest


EXTENSION_TYPES = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".docx": "docx",
    ".md": "md",
    ".txt": "txt",
    ".html": "html",
    ".htm": "html",
}


def infer_source_type(filename: str) -> str:
    return EXTENSION_TYPES.get(Path(filename).suffix.lower(), "unknown")


def build_source_item(upload_meta: dict[str, Any], role: str = "reference", priority: int = 3, notes: str = "") -> SourceItem:
    filename = str(upload_meta.get("file_name") or upload_meta.get("filename") or "source.bin")
    source_path = str(upload_meta.get("saved_path") or upload_meta.get("path") or "")
    title = Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename
    return SourceItem(
        source_id=str(upload_meta.get("id") or upload_meta.get("source_id")),
        filename=filename,
        path=source_path,
        source_type=infer_source_type(filename),
        role=role,
        priority=priority,
        title=title,
        notes=notes,
    )


def build_source_manifest(project_id: str, source_items: list[SourceItem]) -> SourceManifest:
    ranked = sorted(source_items, key=lambda item: (item.priority, item.role != "primary", item.filename))
    return SourceManifest(project_id=project_id, sources=ranked, created_at=time.time())


def validate_source_manifest(manifest: SourceManifest) -> list[str]:
    issues: list[str] = []
    if not manifest.sources:
        issues.append("source_manifest has no sources")
    primary_count = sum(1 for item in manifest.sources if item.role == "primary")
    if primary_count == 0:
        issues.append("source_manifest has no primary source")
    for item in manifest.sources:
        if item.source_type != "url" and item.path and not Path(item.path).exists():
            issues.append(f"missing source file: {item.source_id}")
    return issues
