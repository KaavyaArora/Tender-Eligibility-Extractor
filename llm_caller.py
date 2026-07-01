import sys
import urllib.request
import json

MODEL = "qwen3:14b-instruct"

PROMPT = """/no_think
You are reading excerpts from a government tender document.
Your job is to find and list EVERY condition a bidder must satisfy to be eligible.

TENDER EXCERPTS:
{context}

INSTRUCTIONS:
- Read every line carefully.
- Extract every requirement, condition, or criteria a bidder must meet.
- Look for phrases: must, shall, should, minimum, at least, required, mandatory, eligible, not less than, registered, certified, experienced.
- Include financial requirements (turnover, net worth).
- Include experience requirements (projects, years, value).
- Include legal requirements (PAN, GST, registration, blacklisting).
- Include document requirements (what must be submitted).
- Include team/personnel requirements.
- Include certification requirements.
- Include consortium/JV conditions.
- Do NOT include scope of work, deliverables, payment terms, SLA, penalties.
- Do NOT include vague references without specific values or conditions.
- Do NOT include process statements, submission instructions, or consequences of non-compliance.
- Preserve exact numbers, rupee values, years, percentages.

OUTPUT FORMAT — READ THIS CAREFULLY:
You must return a raw JSON array where EACH ELEMENT is a separate object.
The array must have ONE object per requirement — not one object containing all requirements.

CORRECT format (multiple objects in the array):
[
  {"requirement": "First condition here.", "page": 3, "section": "Eligibility"},
  {"requirement": "Second condition here.", "page": 4, "section": "Financial"},
  {"requirement": "Third condition here.", "page": 5, "section": null}
]

WRONG — do NOT do this (putting JSON inside a string):
[{"requirement": "[{...}, {...}]", "page": null, "section": null}]

WRONG — do NOT do this (markdown fences):
```json
[...]
```

Rules:
- Start your response with [ and end with ]
- Each object has exactly three fields: "requirement" (string), "page" (integer or null), "section" (string or null)
- The "requirement" field must be a plain string — never a JSON array, never escaped JSON
- No trailing commas
- No comments
- No text before [ or after ]"""


def get_client(api_key=None):
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        sys.exit("ERROR: Ollama is not running.\n"
                 "Start it: open Ollama app or run 'ollama serve'")
    return None


def call(client, context: str) -> str:
    prompt = PROMPT.replace("{context}", context)
    tokens_est = len(prompt) // 4
    print(f"→ Sending ~{tokens_est:,} tokens to Ollama ({MODEL})", file=sys.stderr)

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 16384,
        }
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            raw = json.loads(resp.read())["response"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return raw.strip()
    except Exception as e:
        sys.exit(f"ERROR: Ollama call failed: {e}\n"
                 f"Make sure model is pulled: ollama pull {MODEL}")