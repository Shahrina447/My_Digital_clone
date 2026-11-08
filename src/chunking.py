"""
chunking.py - Document loading and chunking pipeline for Digital Clone RAG.

Assignment Requirements (Criterion 1):
- Split documents into chunks of about 200 to 300 words with small overlap.
- Keep source file name and chunk ID with each chunk as metadata.
- Ensure top retrieved chunks can fit cleanly within an LLM prompt context window.
"""

from __future__ import annotations
import glob
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = 250,
    overlap: int = 50,
    min_chunk_size: int = 60,
) -> List[DocumentChunk]:
    """
    Split text into overlapping word chunks of 200-300 words.

    Args:
        text: Raw document text.
        source_filename: Name of the source file (e.g., 'projects.md').
        chunk_size: Target words per chunk (default: 250, between 200 and 300).
        overlap: Overlapping word count between consecutive chunks (default: 50).
        min_chunk_size: Minimum words threshold; trailing words below this
                        are merged into the preceding chunk.

    Returns:
        List of DocumentChunk instances with source and chunk_id metadata.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    words = cleaned_text.split()
    total_words = len(words)

    # Base name for chunk ID prefix (e.g., 'projects' from 'projects.md')
    base_name = Path(source_filename).stem.replace(" ", "_")

    # If document is small enough to fit within one chunk
    if total_words <= chunk_size:
        return [
            DocumentChunk(
                chunk_id=f"{base_name}_chunk_0",
                source=source_filename,
                text=" ".join(words),
                word_count=total_words,
            )
        ]

    chunks: List[DocumentChunk] = []
    step = chunk_size - overlap
    chunk_index = 0

    for start_idx in range(0, total_words, step):
        end_idx = min(start_idx + chunk_size, total_words)
        chunk_words = words[start_idx:end_idx]

        # If this is a small trailing slice and we already have chunks,
        # merge it into the previous chunk to avoid tiny fragments.
        if len(chunk_words) < min_chunk_size and chunks:
            prev_chunk = chunks[-1]
            merged_words = words[start_idx - step : total_words]
            chunks[-1] = DocumentChunk(
                chunk_id=prev_chunk.chunk_id,
                source=prev_chunk.source,
                text=" ".join(merged_words),
                word_count=len(merged_words),
            )
            break

        chunks.append(
            DocumentChunk(
                chunk_id=f"{base_name}_chunk_{chunk_index}",
                source=source_filename,
                text=" ".join(chunk_words),
                word_count=len(chunk_words),
            )
        )
        chunk_index += 1

        if end_idx == total_words:
            break

    return chunks


def load_and_chunk_documents(
    data_dir: str = "data",
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Load all Markdown (.md) and text (.txt) documents from data_dir and
    chunk them according to the assignment requirements.

    Returns:
        List of all DocumentChunk objects across the knowledge base.
    """
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Knowledge base directory '{data_dir}' not found.")

    supported_extensions = ("*.md", "*.txt")
    files: List[Path] = []
    for ext in supported_extensions:
        files.extend(data_path.glob(ext))

    files = sorted(files)
    if not files:
        raise ValueError(f"No .md or .txt files found in '{data_dir}'.")

    all_chunks: List[DocumentChunk] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_chunks = chunk_text(
            text=content,
            source_filename=file_path.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Document Chunking Pipeline...")
    print("=" * 60)

    chunks = load_and_chunk_documents(data_dir="data", chunk_size=250, overlap=50)

    word_counts = [c.word_count for c in chunks]
    doc_sources = sorted(list(set(c.source for c in chunks)))

    print(f"Total documents processed: {len(doc_sources)}")
    print(f"Total chunks created:     {len(chunks)}")
    print(f"Average chunk size:       {sum(word_counts) / len(word_counts):.1f} words")
    print(f"Min chunk size:           {min(word_counts)} words")
    print(f"Max chunk size:           {max(word_counts)} words")
    print("\nDocuments indexed:")
    for src in doc_sources:
        src_chunks = [c for c in chunks if c.source == src]
        print(f"  - {src:<28} : {len(src_chunks)} chunk(s)")

    print("\nSample Chunk Metadata Preview:")
    sample = chunks[0]
    print(f"  Chunk ID:   {sample.chunk_id}")
    print(f"  Source:     {sample.source}")
    print(f"  Word Count: {sample.word_count}")
    print(f"  Preview:    {sample.text[:120]}...\n")
