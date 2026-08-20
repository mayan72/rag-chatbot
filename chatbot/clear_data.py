import json
import re
from pathlib import Path

INPUT_FILE = "output.json"
OUTPUT_FILE = "vector_chunks.json"


IGNORE_VALUES = {
    "DoclingDocument",
    "application/pdf",
    "application/octet-stream",
    "TOPLEFT",
    "BOTTOMLEFT",
    "TOPRIGHT",
    "BOTTOMRIGHT",
    "unspecified",
    "picture",
    "table",
    "text",
    "group",
    "body",
    "page_footer",
    "key_value_area",
    "furniture",
    "*root*",
    "rot_0",
}


class ChunkExtractor:

    def __init__(self):
        self.lines = []
        self.seen = set()

    def add(self, text):

        if not isinstance(text, str):
            return

        text = text.strip()

        if len(text) < 2:
            return

        if text in IGNORE_VALUES:
            return

        if text.startswith("#/"):
            return

        if re.fullmatch(r"texts/\d+", text):
            return

        if text in self.seen:
            return

        self.seen.add(text)

        self.lines.append(text)

    def walk(self, obj):

        if isinstance(obj, dict):

            for value in obj.values():
                self.walk(value)

        elif isinstance(obj, list):

            for item in obj:
                self.walk(item)

        elif isinstance(obj, str):

            self.add(obj)


def create_chunks(lines):

    chunks = []

    current = []
    section = "general"

    keywords = {
        "account summary": "account_summary",
        "deposits": "deposits",
        "atm withdrawals": "atm_withdrawals",
        "checks paid": "checks_paid",
        "statement date": "statement_information",
        "commerce bank": "bank_information",
        "primary account": "account_information",
    }

    for line in lines:

        lower = line.lower()

        new_section = None

        for k, v in keywords.items():

            if k in lower:
                new_section = v
                break

        if new_section:

            if current:
                chunks.append(
                    {
                        "section": section,
                        "text": "\n".join(current)
                    }
                )

            current = [line]
            section = new_section

        else:
            current.append(line)

    if current:
        chunks.append(
            {
                "section": section,
                "text": "\n".join(current)
            }
        )

    return chunks


def main():

    with open(INPUT_FILE, encoding="utf8") as f:
        data = json.load(f)

    extractor = ChunkExtractor()

    extractor.walk(data)

    chunks = create_chunks(extractor.lines)

    final = []

    for i, chunk in enumerate(chunks):

        final.append(
            {
                "id": f"chunk_{i+1}",
                "section": chunk["section"],
                "document_type": "bank_statement",
                "text": chunk["text"]
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    print(f"\nCreated {len(final)} chunks")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()