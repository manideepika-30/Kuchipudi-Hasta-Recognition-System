"""
Extract authentic curriculum text, shlokas, and images from original Kuchipudi PDFs.
"""
import os
import glob
import json
import pypdf

PDF_DIR = "docs/pdfs"
OUTPUT_DIR = "docs/extracted"
IMG_OUTPUT_DIR = "frontend/public/assets/curriculum"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

data = {}
image_count = 0

for pdf in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
    filename = os.path.basename(pdf)
    clean_name = filename.replace(".pdf", "").replace(" ", "_")
    reader = pypdf.PdfReader(pdf)
    pages_list = []
    
    print(f"Processing: {filename} ({len(reader.pages)} pages)")
    for p_idx, page in enumerate(reader.pages):
        page_num = p_idx + 1
        text = page.extract_text() or ""
        pages_list.append({"page": page_num, "text": text})
        
        # Extract images
        for img_idx, img in enumerate(page.images):
            ext = img.name.split(".")[-1]
            img_name = f"{clean_name}_p{page_num}_{img_idx}.{ext}"
            img_path = os.path.join(IMG_OUTPUT_DIR, img_name)
            try:
                with open(img_path, "wb") as f:
                    f.write(img.data)
                image_count += 1
            except Exception as e:
                print(f"  Warning saving image {img_name}: {e}")

    data[filename] = pages_list

out_json = os.path.join(OUTPUT_DIR, "all_curriculum_text.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n[DONE] Successfully extracted {len(data)} PDFs to {out_json}")
print(f"[DONE] Extracted {image_count} authentic images into {IMG_OUTPUT_DIR}")
