import json

with open("docs/extracted/all_curriculum_text.json", encoding="utf-8") as f:
    d = json.load(f)

for k, pages in d.items():
    print("=" * 60)
    print(f"PDF: {k}")
    full_text = "\n".join([p["text"].strip() for p in pages])
    print(f"Total length: {len(full_text)} characters")
    print(full_text[:1200])
    print("...\n")
