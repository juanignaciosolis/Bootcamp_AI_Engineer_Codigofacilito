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

    # priorizar título
    title = text.split("\n")[0]

    if "faq" in title or "?" in text:
        tipo = "faq"

    elif "contraindic" in title:
        tipo = "contraindicaciones"

    elif "beneficio" in title:
        tipo = "beneficios"

    elif "pago" in title or "cuota" in text:
        tipo = "pagos"

    elif "combo" in title or "promo" in title:
        tipo = "combos"

    elif "concept" in title or "tecnolog" in title:
        tipo = "concepto"

    elif "cuidado" in title:
        tipo = "cuidados"

    else:
        tipo = "general"

    return {"tipo": tipo}


def chunk_by_sections(
    doc,
    max_chunk_size: int = 800,
) -> List["Chunk"]:

    import re

    text = doc.content

    # 🔹 limpieza inicial
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")

    sections = []
    current_title = None
    current_content = []

    def is_section_title(line: str, next_line: str):
        return bool(line.strip()) and set(next_line.strip()) == {"="}

    # 🔹 detectar secciones reales (solo con =====)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if is_section_title(line, next_line):
            if current_title:
                content = "\n".join(current_content)
                content = re.sub(r"\n{3,}", "\n\n", content).strip()
                sections.append((current_title, content))

            current_title = line
            current_content = []
            i += 2  # saltar =====
            continue

        if set(line) == {"="}:
            i += 1
            continue

        current_content.append(lines[i])
        i += 1

    # última sección
    if current_title:
        content = "\n".join(current_content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        sections.append((current_title, content))

    # 🔹 chunking
    chunks = []
    chunk_index = 0

    for title, content in sections:

        # =========================================================
        # 🔥 CASO ESPECIAL: CONCEPTOS Y TECNOLOGIAS
        # =========================================================
        if "CONCEPTOS Y TECNOLOGIAS" in title.upper():

            # split por tratamientos (líneas mayúsculas)
            parts = re.split(r"\n(?=[A-ZÁÉÍÓÚÑ\s\/\(\)]+(?:\n|$))", content)

            for part in parts:
                part = part.strip()

                # evitar ruido (bloques muy chicos)
                if not part or len(part) < 20:
                    continue

                chunk_text = f"{title}\n{part}"

                chunk_text = re.sub(r"\n{3,}", "\n\n", chunk_text)
                chunk_text = re.sub(r"[ \t]+", " ", chunk_text).strip()

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "section": title.strip(),
                            "tipo": "concepto",  # 🔥 forzado correcto
                        },
                    )
                )
                chunk_index += 1

            continue  # 🔥 NO seguir con lógica normal

        # =========================================================
        # 🔹 LÓGICA NORMAL
        # =========================================================

        section_full = f"{title}\n{content}"

        section_full = re.sub(r"\n?=+\n?", "\n", section_full)
        section_full = re.sub(r"\n{3,}", "\n\n", section_full)
        section_full = re.sub(r"[ \t]+", " ", section_full).strip()

        base_metadata = infer_metadata(section_full)

        if len(section_full) <= max_chunk_size:
            chunks.append(
                Chunk(
                    content=section_full,
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
                p = re.sub(r"[ \t]+", " ", p)

                if not p:
                    continue

                if len(current) + len(p) + 2 > max_chunk_size:
                    chunk_text = f"{title}\n{current}".strip()

                    chunks.append(
                        Chunk(
                            content=chunk_text,
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
                chunk_text = f"{title}\n{current}".strip()

                chunks.append(
                    Chunk(
                        content=chunk_text,
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