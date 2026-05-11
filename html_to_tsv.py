import argparse
import base64
import csv
import re
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup, FeatureNotFound

KNOWN_HEADERS = {
    "sample_id", "image", "a", "b", "c", "d", "e",
    "answer", "gender", "country", "question type", "specialty",
}

OUTPUT_COLUMNS = [
    "index", "image", "question", "A", "B", "C", "D",
    "answer", "correct_option", "question_type", "gender",
    "country", "specialty", "language", "sample_id",
    "image_filename", "source_file", "source_sheet",
]


def clean_leading(text: str):
    return re.sub(r'^[\s\u00A0\.\,\-\–\—\:;…]+', '', text).strip()


def normalize_correct_option(text: str):
    text = text.strip()
    m = re.match(r'^([A-Ea-e])[\)\.\-]?$', text)
    if m:
        letter = m.group(1).upper()
        return letter if letter in "ABCD" else ""
    return ""


def encode_b64(data: bytes):
    return base64.b64encode(data).decode("utf-8")


def find_question_col(headers: list):
    """Return index of the column that contains the question (not a known header)."""
    for i, h in enumerate(headers):
        if h.strip().lower() not in KNOWN_HEADERS:
            return i
    return None


def build_image_map(*zip_files: zipfile.ZipFile) -> dict:
    """Return a dict mapping lowercase filename -> b64 string from one or more zips."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_map = {}
    for zf in zip_files:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() in IMAGE_EXTS:
                image_map[p.name.lower()] = encode_b64(zf.read(name))
    return image_map


def parse_spreadsheet_html(html_content: str, image_map: dict,
                            language: str, source_file: str, source_sheet: str):
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html_content, "html.parser")

    rows = soup.find_all("tr")
    if not rows:
        return []

    # Find header row (first <tr> whose <td> cells look like column headers)
    header_row_idx = None
    headers = []
    for i, row in enumerate(rows):
        cells = row.find_all("td")
        if len(cells) >= 5:
            texts = [c.get_text(" ", strip=True) for c in cells]
            low = [t.lower() for t in texts]
            if "sample_id" in low or "answer" in low:
                headers = texts
                header_row_idx = i
                break

    if header_row_idx is None:
        return []

    col = {h.strip().lower(): idx for idx, h in enumerate(headers)}
    question_col_idx = find_question_col(headers)
    img_col_idx = col.get("image")

    def get(cells, key):
        idx = col.get(key.lower())
        if idx is None or idx >= len(cells):
            return ""
        return cells[idx].get_text(" ", strip=True).strip()

    rows_out = []
    for row in rows[header_row_idx + 1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        if all(c.get_text(" ", strip=True).strip() == "" for c in cells):
            continue

        sample_id = get(cells, "sample_id")
        question = (
            cells[question_col_idx].get_text(" ", strip=True).strip()
            if question_col_idx is not None and question_col_idx < len(cells)
            else ""
        )
        opt_a = get(cells, "a")
        opt_b = get(cells, "b")
        opt_c = get(cells, "c")
        opt_d = get(cells, "d")
        raw_answer = get(cells, "answer")
        gender = get(cells, "gender")
        country = get(cells, "country")
        question_type = get(cells, "question type")
        specialty = get(cells, "specialty")

        correct_option = normalize_correct_option(raw_answer)
        opt_map = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
        answer_text = clean_leading(opt_map.get(correct_option, raw_answer))

        img_filename = ""
        img_b64 = ""
        if img_col_idx is not None and img_col_idx < len(cells):
            img_filename = cells[img_col_idx].get_text(" ", strip=True).strip()
            img_b64 = image_map.get(img_filename.lower(), "")

        if not question:
            continue

        rows_out.append({
            "index": len(rows_out) + 1,
            "image": img_b64,
            "question": question,
            "A": opt_a,
            "B": opt_b,
            "C": opt_c,
            "D": opt_d,
            "answer": answer_text,
            "correct_option": correct_option,
            "question_type": question_type,
            "gender": gender,
            "country": country,
            "specialty": specialty,
            "language": language,
            "sample_id": sample_id,
            "image_filename": img_filename,
            "source_file": source_file,
            "source_sheet": source_sheet,
        })

    return rows_out


def write_tsv(rows, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def process_zip(zip_path: Path, out_path: Path = None, images_zip_path: Path = None):
    zip_name = zip_path.name
    language = zip_name.split("_")[0]

    if out_path is None:
        stem = zip_name
        for suffix in zip_path.suffixes:
            stem = stem[: -len(suffix)]
        out_path = zip_path.parent / f"{stem}.tsv"

    all_rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        html_files = [n for n in zf.namelist() if n.lower().endswith(".html")]
        if not html_files:
            raise ValueError(f"No HTML files found in {zip_path}")

        images_zf = zipfile.ZipFile(images_zip_path, "r") if images_zip_path else None
        try:
            img_map = build_image_map(zf, *([images_zf] if images_zf else []))
            print(f"Loaded {len(img_map)} images")

            for html_name in html_files:
                source_sheet = Path(html_name).stem
                html_content = zf.read(html_name).decode("utf-8")
                print(f"Parsing {html_name}...")
                rows = parse_spreadsheet_html(
                    html_content, img_map, language, zip_name, source_sheet,
                )
                offset = len(all_rows)
                for r in rows:
                    r["index"] = offset + r["index"]
                all_rows.extend(rows)
        finally:
            if images_zf:
                images_zf.close()

    write_tsv(all_rows, out_path)
    print(f"Wrote {len(all_rows)} rows → {out_path.resolve()}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Convert AfriMedQA zip (xlsx HTML export) to TSV for VLM evaluation."
    )
    p.add_argument("zip", type=Path, help="Path to the .zip file (e.g. TWI_TEST.xlsx.zip)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output TSV path (default: <stem>.tsv next to zip)")
    p.add_argument("--images", type=Path, default=None, metavar="IMAGES_ZIP",
                   help="Separate zip file containing image files")
    args = p.parse_args()

    process_zip(args.zip, args.out, images_zip_path=args.images)


if __name__ == "__main__":
    main()
