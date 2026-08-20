import json
from pathlib import Path

from docling.document_converter import DocumentConverter


def main():
    image_path = Path("bank_docs.jpg")

    if not image_path.exists():
        raise FileNotFoundError(f"{image_path} not found.")

    print(f"Processing: {image_path.name}")

    converter = DocumentConverter()

    result = converter.convert(str(image_path))

    document = result.document

    # -----------------------------
    # Export as JSON
    # -----------------------------
    data = document.export_to_dict()

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✓ Saved output.json")

    # -----------------------------
    # Export as Markdown
    # -----------------------------
    markdown = document.export_to_markdown()

    with open("output.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    print("✓ Saved output.md")

    # -----------------------------
    # Print document structure
    # -----------------------------
    print("\nDocument Structure:\n")

    if isinstance(data, dict):
        for key in data.keys():
            print(f"- {key}")

    print("\nDone.")


if __name__ == "__main__":
    main()