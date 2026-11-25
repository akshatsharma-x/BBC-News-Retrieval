"""Reset document metadata (author, last-modified-by, generator tags) before publishing.

Word and PowerPoint stamp the local Office user name into files they save; this
replaces it with neutral project metadata.

Usage:  python report/clean_metadata.py report.docx slides.pptx
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

TITLE = {"docx": "BBC News Retrieval System — Project Report",
         "pptx": "BBC News Retrieval System — Presentation"}


def clean(path: Path) -> None:
    kind = path.suffix.lstrip(".").lower()
    if kind == "docx":
        from docx import Document
        doc = Document(str(path))
        props = doc.core_properties
    elif kind == "pptx":
        from pptx import Presentation
        doc = Presentation(str(path))
        props = doc.core_properties
    else:
        raise SystemExit(f"unsupported file: {path}")
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    props.title = TITLE[kind]
    props.subject = "Information Retrieval — Project 12: BBC News Retrieval"
    props.author = ""
    props.last_modified_by = ""
    props.comments = ""
    props.keywords = "information retrieval; BBC News; BM25; TF-IDF; Sentence Transformer; RRF"
    props.created = props.modified = now
    doc.save(str(path))
    print("cleaned", path.name)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        clean(Path(arg))
