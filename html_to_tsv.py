import argparse
import base64
import csv
import re
from pathlib import Path
from bs4 import BeautifulSoup, FeatureNotFound

DEFAULT_HTML = "All_Pics_Questions/All_Pics_Questions.html"
DEFAULT_OUT = "AfrimedQA.tsv"

def clean_question(text: str):
    return re.sub(r'^\s*\d+\s*[).]\s*', '', text).strip()

def clean_leading(text: str):
    return re.sub(r'^[\s\u00A0\.\,\-\–\—\:;…]+', '', text).strip()

def strip_option(text: str):
    return re.sub(r'^[A-D]\s*[\)\.\-]\s*', '', text).strip()

def split_answer_cell(text: str):
    text = text.strip()
    if not text:
        return '', ''
    m = re.match(r'ans\s*[:\-]\s*([A-D])\s*[\)\.\-]?\s*(.*)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), clean_leading(m.group(2))
    m = re.match(r'^([A-D])\s*[\)\.\-]\s*(.+)$', text)
    if m:
        return m.group(1).upper(), clean_leading(m.group(2))
    m = re.match(r'^([A-D])[\)\.\-]?$', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), ''
    m = re.match(r'\(?\s*ans\s*[:\-]\s*(.*?)\s*\)?$', text, flags=re.IGNORECASE)
    if m:
        return '', clean_leading(m.group(1))
    return '', clean_leading(text)

def encode_b64(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def parse_html_to_rows(html_path: Path):
    soup = None
    html_text = html_path.read_text(encoding="utf-8")
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html_text, "html.parser")

    img_root = html_path.parent
    rows_out = []

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        lis = tds[0].find_all("li")
        if not lis:
            continue
        question = clean_question(lis[0].get_text(" ", strip=True))
        raw_options = [li.get_text(" ", strip=True) for li in lis[1:]]

        options = [strip_option(o) for o in raw_options]
        while len(options) < 4:
            options.append("")

        img_b64 = ""
        img_tag = tr.find("img")
        if img_tag and img_tag.get("src"):
            img_path = img_root / img_tag["src"]
            if img_path.exists():
                img_b64 = encode_b64(img_path)

        answer_cell_text = " ".join(tds[-1].stripped_strings)
        correct_letter, correct_answer = split_answer_cell(answer_cell_text)

        if any(options):
            if not correct_letter and correct_answer:
                for idx, opt in enumerate(options):
                    if correct_answer.lower() in opt.lower():
                        correct_letter = "ABCD"[idx]
                        break
            if correct_letter and not correct_answer:
                idx = "ABCD".find(correct_letter)
                if 0 <= idx < 4:
                    correct_answer = options[idx]

        if any(options):
            question_type = "MCQ"
            answer_field = clean_leading(correct_answer)
        else:
            question_type = "SAQ"
            answer_field = clean_leading(correct_answer or answer_cell_text)

        rows_out.append({
            "index": len(rows_out) + 1,
            "image": img_b64,
            "question": question,
            "A": options[0],
            "B": options[1],
            "C": options[2],
            "D": options[3],
            "answer": answer_field,
            "correct_option": correct_letter,
            "question_type": question_type,
            "split": "test",
        })

    return rows_out

def write_tsv(rows, out_path: Path):
    fieldnames = [
        "index", "image", "question", "A", "B", "C", "D",
        "answer", "correct_option", "question_type", "split"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="Convert AfriMedQA HTML files to TSV format for VLM toolkit evaluation.")
    p.add_argument("--html", type=Path, default=Path(DEFAULT_HTML), help="Path to Google Docs HTML export")
    p.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="Output TSV path")
    args = p.parse_args()

    rows = parse_html_to_rows(args.html)
    write_tsv(rows, args.out)
    print(f"Wrote {len(rows)} rows → {args.out.resolve()}")

if __name__ == "__main__":
    main()
