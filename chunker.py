import re
from dataclasses import dataclass
from pdf_extractor import PageBlock, is_eligibility_heading

@dataclass
class Chunk:
    chunk_id: int
    heading: str
    text: str
    page_start: int
    page_end: int
    is_eligibility_section: bool
    boost_score: float

ELIGIBILITY_KEYWORDS = [
    r"\bturnover\b", r"\bnet worth\b", r"\bexperience\b", r"\bcompletion certificate\b",
    r"\bwork order\b", r"\bcertif", r"\biso\b", r"\bcmm(i)?\b", r"\blicen[cs]e\b",
    r"\bregistration\b", r"\bat least\b", r"\bminimum\b", r"\bshall have\b",
    r"\bmust have\b", r"\bblacklist\b", r"\bdebarr\b", r"\blitigation\b",
    r"\bsolvency\b", r"\bjoint venture\b", r"\bconsortium\b", r"\bkey personnel\b",
    r"\bmanpower\b", r"\bcrore\b", r"\blakh\b", r"\brs\.\s*\d", r"\bpan\b",
    r"\bgst\b", r"\baffidavit\b", r"\bundertaking\b", r"\bsimilar work\b",
]

SKIP_TEXT = [
    r"payment\s+schedule", r"billing\s+(cycle|period)", r"invoice",
    r"service\s+level", r"\bsla\b", r"penalty\s+clause", r"liquidated\s+damage",
]

def has_eligibility_keyword(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in ELIGIBILITY_KEYWORDS)

def should_skip(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in SKIP_TEXT)

def build_chunks(blocks: list[PageBlock], max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    chunks, chunk_id = [], 0
    cur_texts, cur_pages = [], []
    cur_heading = ""
    in_elig = False

    def flush():
        nonlocal chunk_id
        if not cur_texts:
            return
        text = " ".join(cur_texts)
        boost = 1.5 if in_elig else (1.25 if has_eligibility_keyword(text) else 1.0)
        chunks.append(Chunk(chunk_id, cur_heading, text,min(cur_pages), max(cur_pages), in_elig, boost))
        chunk_id += 1

    for b in blocks:
        if b.is_heading:
            flush()
            cur_texts, cur_pages = [], []
            cur_heading = b.text
            in_elig = is_eligibility_heading(b.text)

        cur_texts.append(b.text)
        cur_pages.append(b.page_num)

        if sum(len(t) for t in cur_texts) > max_chars:
            flush()
            overlap_text = " ".join(cur_texts)[-overlap:]
            cur_texts = [overlap_text]
            cur_pages = [cur_pages[-1]]

    flush()
    print(f"→ {len(chunks)} chunks ",file=__import__('sys').stderr)
    return chunks