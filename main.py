#!/usr/bin/env python3
import sys, argparse, time, json, re
from pathlib import Path

def log(msg): print(msg, file=sys.stderr, flush=True)

def parse_requirements(raw: str) -> list:
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return dedupe(parsed)
    except json.JSONDecodeError:
        pass
    log("Could not parse model output as JSON — check raw output")
    return [{"requirement": raw, "page": None, "section": None}]


def dedupe(reqs: list) -> list:
    seen, out = set(), []
    for r in reqs:
        key = r.get("requirement", "")[:60].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf_path")
    p.add_argument("--api-key", "-k", default=None)
    p.add_argument("--output",  "-o", default=None)
    args = p.parse_args()

    pdf = Path(args.pdf_path)
    if not pdf.exists(): sys.exit(f"ERROR: File not found: {pdf}")

    output = args.output or f"{pdf.stem}_eligibility.json"
    t0 = time.time()

    log(f"[1/5] Extracting text from {pdf.name}")
    from pdf_extractor import extract_pages
    blocks = extract_pages(str(pdf))
    if not blocks: sys.exit("ERROR: No text extracted. PDF may be scanned — run OCR first.")

    log("[2/5] Chunking")
    from chunker import build_chunks
    chunks = build_chunks(blocks)

    log("[3/5] Embedding (BGE-large)")
    from vector_store import VectorStore
    store = VectorStore()
    store.build(chunks)

    log("[4/5] Retrieving relevant chunks")
    from retriever import retrieve, format_context
    relevant = retrieve(store)
    context = format_context(relevant)
    log(f"→ ~{len(context)//4:,} tokens")

    log("[5/5] Calling Ollama")
    from llm_caller import get_client, call, MODEL
    client = get_client(api_key=args.api_key)
    raw = call(client, context)

    requirements = parse_requirements(raw)

    elapsed = time.time() - t0
    result = {
        "document": pdf.name,
        "model": MODEL,
        "total_requirements": len(requirements),
        "requirements": requirements,
        "metadata": {
            "pages_processed": max(b.page_num for b in blocks),
            "chunks_retrieved": len(relevant),
            "tokens_sent": len(context) // 4,
            "time_seconds": round(elapsed, 1),
        }
    }

    output_str = json.dumps(result, indent=2, ensure_ascii=False)
    Path(output).write_text(output_str, encoding="utf-8")
    log(f"\nDone in {elapsed:.1f}s → {output}")
    print(output_str)

if __name__ == "__main__":
    main()