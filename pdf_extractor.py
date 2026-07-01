import re, sys
import fitz
from dataclasses import dataclass

#set full path to tesseract.exe
TESSERACT_PATH = None

@dataclass
class PageBlock:
    page_num: int
    heading: str
    text: str
    is_heading: bool

ELIGIBILITY_HEADINGS = [
    r"eligibilit", r"pre[\s\-]?qualif", r"pq[c\b]", r"qualification",
    r"technical\s+(qualification|requirement|criteria)", r"financial\s+(qualification|reqrement|criteria)",
    r"experience\s+(requirement|criteria)", r"certif", r"bidder\s+requirement",
    r"minimum\s+(requirement|criteria)", r"mandatory\s+(requirement|criteria)",
    r"consortium", r"joint\s+venture", r"manpower", r"key\s+personnel",
    r"net\s+worth", r"turnover", r"blacklist", r"debarr",
    r"legal\s+(requirement|compliance)", r"document\s+required",
    r"documents?\s+to\s+be\s+submitted", r"bid\s+(condition|requirement|eligibilit)",
]

SKIP_HEADINGS = [
    r"scope\s+of\s+work", r"scope\s+of\s+service", r"payment\s+(term|schedule|condition)",
    r"service\s+level\s+agreement", r"\bsla\b", r"penalty\s+clause",
    r"liquidated\s+damage", r"warranty", r"maintenance", r"technical\s+specification",
    r"functional\s+requirement", r"general\s+(term|condition)", r"contract\s+(term|condition)",
    r"delivery\s+(term|schedule)",
]

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if re.match(r"^(section\s+)?\d+(\.\d+)*[\s\.\)]+\S", line, re.I):
        return True
    if line.isupper() and 3 <= len(line) <= 100:
        return True
    words = line.split()
    if len(words) <= 8 and not line.endswith((".", ",", ";", ":")):
        if sum(1 for w in words if w[0].isupper()) / len(words) >= 0.6:
            return True
    return False

def matches(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def is_eligibility_heading(h: str) -> bool:
    return matches(h, ELIGIBILITY_HEADINGS)

def is_skip_heading(h: str) -> bool:
    return matches(h, SKIP_HEADINGS)

def ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io
        if TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except ImportError:
        print("pytesseract not installed — skipping OCR for scanned page.\n"
              "Install: pip install pytesseract pillow\n"
              "And Tesseract: https://github.com/UB-Mannheim/tesseract/wiki",
              file=sys.stderr)
        return ""
    except Exception as e:
        print(f"OCR failed on page {page.number + 1}: {e}", file=sys.stderr)
        return ""

def extract_pages(pdf_path: str) -> list[PageBlock]:
    blocks = []
    doc = fitz.open(pdf_path)
    total = doc.page_count
    print(f"→ {total} pages", file=sys.stderr)

    scanned_count = 0
    current_heading = ""

    for page in doc:
        text = page.get_text()
        if not text.strip():
            scanned_count += 1
            text = ocr_page(page)

        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if is_heading(line):
                current_heading = line
            blocks.append(PageBlock(page.number + 1, current_heading, line, is_heading(line)))

    if scanned_count > 0:
        print(f"→ {scanned_count} scanned pages processed via OCR", file=sys.stderr)

    doc.close()
    return blocks