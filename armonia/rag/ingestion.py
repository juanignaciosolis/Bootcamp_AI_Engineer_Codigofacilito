import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    content: str
    metadata: dict
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def load_txt(path: str) -> Document:
    """Carga un archivo .txt como Document."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        content=content,
        metadata={"source": path, "type": "txt"},
    )


def load_pdf(path: str) -> Document:
    """Carga un archivo .pdf como Document usando pypdf."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(pages)
    return Document(
        content=content,
        metadata={"source": path, "type": "pdf", "pages": len(reader.pages)},
    )


def load_markdown(path: str) -> Document:
    """Carga un archivo .md como Document, removiendo frontmatter YAML si existe."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Remover frontmatter YAML (entre --- al inicio del archivo)
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)
    return Document(
        content=content,
        metadata={"source": path, "type": "md"},
    )


LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".md": load_markdown,
}


def load_document(path: str) -> Document:
    """Carga un documento según su extensión."""
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Extensión no soportada: {ext} (archivo: {path})")
    return loader(path)


def load_directory(dir_path: str) -> list[Document]:
    """Carga todos los documentos soportados de un directorio."""
    documents: list[Document] = []
    for filename in sorted(os.listdir(dir_path)):
        ext = Path(filename).suffix.lower()
        if ext in LOADERS:
            full_path = os.path.join(dir_path, filename)
            documents.append(load_document(full_path))
    return documents


import re
from typing import List


def infer_metadata(section_text: str) -> dict:
    text = section_text.lower()

    # tipo general
    if "faq" in text:
        tipo = "faq"
    elif "concept" in text or "tecnolog" in text:
        tipo = "concepto"
    elif "combo" in text or "pago" in text:
        tipo = "comercial"
    else:
        tipo = "general"

    # tema más específico
    if "turno" in text:
        tema = "turnos"
    elif "pago" in text or "cuota" in text:
        tema = "pagos"
    elif "cuidado" in text or "previo" in text:
        tema = "cuidados"
    elif "contraindic" in text:
        tema = "contraindicaciones"
    elif "radiofrecuencia" in text:
        tema = "radiofrecuencia"
    elif "ipl" in text:
        tema = "ipl"
    elif "dermapen" in text:
        tema = "dermapen"
    else:
        tema = "general"

    # keywords simples
    keywords = []
    for word in ["turno", "pago", "cuotas", "tratamiento", "piel", "sesiones"]:
        if word in text:
            keywords.append(word)

    return {
        "tipo": tipo,
        "tema": tema,
        "keywords": keywords,
    }


def chunk_by_sections(
    doc,
    max_chunk_size: int = 800,
) -> List["Chunk"]:
    """
    Chunking por secciones + metadata enriquecida
    """

    text = doc.content

    # Detectar secciones (títulos en mayúsculas + ===)
    section_pattern = re.compile(r"(?:^|\n)([A-ZÁÉÍÓÚÑ\s]{5,}\n=+\n)", re.MULTILINE)
    splits = section_pattern.split(text)

    chunks = []
    chunk_index = 0

    sections = []
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        sections.append((title, content))

    for title, content in sections:
        section_full = f"{title}\n{content}"

        base_metadata = infer_metadata(section_full)

        if len(section_full) <= max_chunk_size:
            chunks.append(
                Chunk(
                    content=section_full.strip(),
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "section": title.strip(),
                        **base_metadata,
                    },
                )
            )
            chunk_index += 1
        else:
            paragraphs = content.split("\n\n")
            current = ""

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                if len(current) + len(p) + 2 > max_chunk_size:
                    chunks.append(
                        Chunk(
                            content=f"{title}\n{current}".strip(),
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_index,
                                "section": title.strip(),
                                **base_metadata,
                            },
                        )
                    )
                    chunk_index += 1
                    current = p
                else:
                    current += "\n\n" + p if current else p

            if current:
                chunks.append(
                    Chunk(
                        content=f"{title}\n{current}".strip(),
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "section": title.strip(),
                            **base_metadata,
                        },
                    )
                )
                chunk_index += 1

    return chunks
