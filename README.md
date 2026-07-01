# Tender-Eligibility-Extractor

Extracts every bidder eligibility requirement from a government tender PDF using a local RAG pipeline. Runs fully offline.

---

## How it works

1. Extracts text from the PDF (with OCR fallback for scanned pages)
2. Chunks and embeds text using `BAAI/bge-large-en-v1.5`
3. Retrieves the most relevant chunks via FAISS vector search
4. Sends context to a local LLM via Ollama
5. Outputs a structured JSON file of all eligibility requirements

---

## Requirements

### System dependencies

**Ollama** — runs the LLM locally
- Download and install from https://ollama.com
- Pull the model:
  ```bash
  ollama pull qwen3:14b-instruct
  ```
- Start the server:
  ```bash
  ollama serve
  ```

**Tesseract OCR** — only needed for scanned PDFs
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- After installing on Windows, update the path in `pdf_extractor.py`:
  ```python
  TESSERACT_PATH = r'C:\path\to\tesseract.exe'
  ```

---

### Python dependencies

Requires Python 3.10+

```bash
pip install -r requirements.txt
```

```
pymupdf>=1.23.0
sentence-transformers>=3.0.0
faiss-cpu>=1.7.4
numpy>=1.24.0
tqdm>=4.66.0
pytesseract>=0.3.10
Pillow>=10.0.0
```

---

## Usage

```bash
python main.py path/to/tender.pdf
```

Output is saved as `<tender_name>_eligibility.json` in the same directory.

**Optional flags:**
```bash
python main.py path/to/tender.pdf --output results.json
```

---

## Output format

```json
{
  "document": "tender.pdf",
  "model": "qwen3:14b-instruct",
  "total_requirements": 37,
  "requirements": [
    {
      "requirement": "Average Annual Financial Turnover during the last 3 financial years should be at least Rs. 63.30 Crore.",
      "page": 4,
      "section": "Financial"
    }
  ],
  "metadata": {
    "pages_processed": 73,
    "chunks_retrieved": 50,
    "tokens_sent": 5500,
    "time_seconds": 55.2
  }
}
```

---

## File structure

```
RAG/
├── main.py            # Entry point
├── pdf_extractor.py   # PDF text extraction + OCR fallback
├── chunker.py         # Text chunking with eligibility boosting
├── vector_store.py    # BGE embeddings + FAISS index
├── retriever.py       # Multi-query retrieval
├── llm_caller.py      # Ollama API call
└── requirements.txt
```
**Slow generation** — check `ollama ps` to confirm the model is running. If PROCESSOR shows `CPU/GPU` split instead of `100% GPU`, your GPU has insufficient VRAM for the model. Either free VRAM from other processes or switch to `qwen3:8b-instruct`.
