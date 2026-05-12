import argparse
import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

from nitbot.EMB import NVIDIAEmbeddings


DEFAULT_DATASET = "dataset_updated.json"
DEFAULT_OUTPUT_DIR = "faiss_index"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS_PER_CHUNK = 800
TOKEN_CHAR_RATIO = 4


def log(message: str):
    print(f"[build_professor_faiss] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from professor outputs in dataset_updated.json."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the fine-tuning dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where index.faiss and chunks.json will be written.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS_PER_CHUNK,
        help="Maximum estimated tokens per chunk.",
    )
    return parser.parse_args()


def estimate_tokens(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / TOKEN_CHAR_RATIO))


def normalize_output(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+\.", ".", text)
    return text


def extract_professor_name(record: dict) -> str:
    candidates = [record.get("instruction", ""), record.get("output", "")]
    patterns = [
        r"(Prof\.?\s+[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,5})",
        r"(Professor\s+[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,5})",
    ]

    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if match:
                return match.group(1).replace("Professor ", "Prof. ").strip()

    fallback = record.get("instruction", "").strip()[:80]
    return fallback or "Unknown Professor"


def load_dataset(dataset_path: Path):
    log(f"Loading dataset from {dataset_path}")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of records.")
    log(f"Loaded {len(data)} records")
    return data


def group_outputs_by_professor(records):
    grouped = defaultdict(OrderedDict)

    for record in records:
        output = normalize_output(record.get("output", ""))
        if not output:
            continue

        professor_name = extract_professor_name(record)
        grouped[professor_name][output] = None

    log(f"Grouped outputs into {len(grouped)} professor buckets")
    return grouped


def split_professor_chunks(professor_name: str, outputs, max_tokens: int):
    chunks = []
    current_parts = []
    current_token_estimate = estimate_tokens(f"Professor: {professor_name}\n")

    for output in outputs:
        candidate = output if output.endswith(".") else f"{output}."
        candidate_tokens = estimate_tokens(candidate)

        if current_parts and current_token_estimate + candidate_tokens > max_tokens:
            chunks.append(build_chunk_record(professor_name, current_parts, len(chunks) + 1))
            current_parts = []
            current_token_estimate = estimate_tokens(f"Professor: {professor_name}\n")

        current_parts.append(candidate)
        current_token_estimate += candidate_tokens

    if current_parts:
        chunks.append(build_chunk_record(professor_name, current_parts, len(chunks) + 1))

    return chunks


def build_chunk_record(professor_name: str, outputs, part_number: int):
    text = f"Professor: {professor_name}\n" + "\n".join(outputs)
    return {
        "source": "dataset_updated.json",
        "professor": professor_name,
        "part": part_number,
        "text": text,
        "token_estimate": estimate_tokens(text),
    }


def build_chunk_records(grouped_outputs, max_tokens: int):
    all_chunks = []
    professor_names = sorted(grouped_outputs)
    total_professors = len(professor_names)

    for index, professor_name in enumerate(professor_names, start=1):
        outputs = list(grouped_outputs[professor_name].keys())
        professor_chunks = split_professor_chunks(professor_name, outputs, max_tokens)
        all_chunks.extend(professor_chunks)

        if index == 1 or index % 100 == 0 or index == total_professors:
            log(
                f"Chunked {index}/{total_professors} professors; total chunks so far: {len(all_chunks)}"
            )
    return all_chunks


def create_embeddings(chunks):
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing. Set it in your environment or .env file.")

    embeddings = NVIDIAEmbeddings(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        model_name=DEFAULT_EMBED_MODEL,
    )
    texts = [chunk["text"] for chunk in chunks]
    log(f"Creating embeddings for {len(texts)} chunks")
    return np.array(embeddings.embed_documents(texts), dtype="float32")


def write_index(chunks, vectors, output_dir: Path):
    log(f"Writing FAISS index to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    (output_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    records = load_dataset(dataset_path)
    grouped = group_outputs_by_professor(records)
    log(f"Building chunks with max token estimate {args.max_tokens}")
    chunks = build_chunk_records(grouped, args.max_tokens)

    if not chunks:
        raise RuntimeError("No output text was found to index.")

    log(f"Prepared {len(chunks)} chunks for indexing")
    vectors = create_embeddings(chunks)
    write_index(chunks, vectors, output_dir)

    log(f"Processed records: {len(records)}")
    log(f"Professors grouped: {len(grouped)}")
    log(f"Chunks written: {len(chunks)}")
    log(f"Index path: {output_dir / 'index.faiss'}")
    log(f"Chunk metadata path: {output_dir / 'chunks.json'}")


if __name__ == "__main__":
    main()
