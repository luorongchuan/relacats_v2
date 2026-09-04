"""Small artifact helpers shared by the two GPU evaluation stages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_jsonl


# Keep the original CaTS evaluation tasks available.  The shell launcher uses
# the paper/Table-2 subset by default, but exposing the full upstream list here
# means ``DATASETS=...`` can reproduce the other CaTS curves without changing
# Python code.  Relational intervention remains disabled for every task at
# test time; numeric tasks are evaluated with the original CaTS handlers.
SUPPORTED_EVAL_DATASETS = (
    "object_counting",
    "gsm8k",
    "math_qa",
    "arc_challenge",
    "arc_easy",
    "svamp",
    "sciq",
    "commonsense_qa",
    "winogrande",
    "openbookqa",
    "reclor",
    "logiqa",
)


def require_local_model(model: str, allow_remote_model: bool = False) -> str:
    """Validate the default, auditable local-model contract."""

    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if allow_remote_model:
        return model
    raise FileNotFoundError(
        f"Model must be a local directory, got {model!r}. "
        "Pass --allow-remote-model only when a Hugging Face download is intentional."
    )


def scalar_json(value: Any) -> Any:
    """Convert common scalar types returned by dataset handlers to JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_chunk(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, records)


def complete_chunk(
    path: Path,
    expected_sample_ids: Sequence[str],
    required_fields: Sequence[str] = (),
) -> bool:
    """Return True only for an intact chunk with exactly the expected samples."""

    if not path.is_file():
        return False
    try:
        records = list(read_jsonl(path))
    except (OSError, ValueError):
        return False
    actual = [str(record.get("sample_id", "")) for record in records]
    return actual == list(expected_sample_ids) and all(
        field in record for record in records for field in required_fields
    )


def merge_chunks(chunk_paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    """Atomically concatenate chunks without loading response text into memory."""

    def records() -> Iterator[dict[str, Any]]:
        for chunk_path in chunk_paths:
            yield from read_jsonl(chunk_path)

    atomic_write_jsonl(output_path, records())
    questions: set[str] = set()
    samples = 0
    digest = hashlib.sha256()
    for record in read_jsonl(output_path):
        question_id = str(record.get("question_id", ""))
        sample_id = str(record.get("sample_id", ""))
        questions.add(question_id)
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
        samples += 1
    return {
        "questions": len(questions),
        "samples": samples,
        "sample_id_sha256": digest.hexdigest(),
    }


def source_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        signatures.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signatures


def response_sources(path: str | Path) -> list[Path]:
    """Resolve response JSONL input while avoiding merged/chunk duplication."""

    root = Path(path).expanduser()
    if root.is_file():
        return [root.resolve()]
    if not root.is_dir():
        raise FileNotFoundError(root)

    # A single artifact directory: prefer its immutable chunks.  This permits
    # confidence calculation to resume even if the final merged file was not
    # written because generation was interrupted just after the last chunk.
    direct_chunks = sorted((root / "chunks").glob("chunk-*.jsonl"))
    if direct_chunks:
        return [item.resolve() for item in direct_chunks]

    # A parent containing multiple response shards.
    metadata_files = sorted(root.rglob("response_metadata.json"))
    sources: list[Path] = []
    for metadata_path in metadata_files:
        chunk_paths = sorted((metadata_path.parent / "chunks").glob("chunk-*.jsonl"))
        if chunk_paths:
            sources.extend(item.resolve() for item in chunk_paths)
        elif (metadata_path.parent / "responses.jsonl").is_file():
            sources.append((metadata_path.parent / "responses.jsonl").resolve())
    if sources:
        return sources

    merged = sorted(root.rglob("responses.jsonl"))
    if merged:
        return [item.resolve() for item in merged]
    raise FileNotFoundError(f"No response JSONL artifacts found below {root}")


def write_manifest(path: Path, artifact_type: str, stats: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": artifact_type,
            **stats,
        },
    )
