import csv, json, re, sys
from pathlib import Path

ROOT = Path(".")
SCHEMA = ROOT / "gapfinder/src/gapfinder/extraction/schema.py"
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/runs/phase3_full_corpus_v4_gemini31_low/extractions.jsonl"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "extractions_posthoc.jsonl"
ANN = ROOT / "experiments/annotation/extraction_sample_annotated.csv"
CORPUS = ROOT / "corpus_main_v4.jsonl"
OUT = ROOT / "repro"
OUT.mkdir(exist_ok=True)

src = SCHEMA.read_text(encoding="utf-8").replace("\r", "")
start = src.index("GENERIC_CLAIM_PHRASES = frozenset(")
end = src.index("class EvidenceValue")
code = src[start:end]
code = re.sub(r"class SpecificityValidationError.*?\n\n\n", "", code, flags=re.S)
ns = {"re": re}
exec(code, ns)
is_generic = ns["is_generic_claim_value"]

nulled = papers = total = 0
with RUN.open(encoding="utf-8") as src_f, (OUT / OUT_NAME).open("w", encoding="utf-8") as out_f:
    for line in src_f:
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        hit = False
        for field in ("task", "method", "dataset"):
            item = rec["claims"].get(field)
            if item and item.get("value") and is_generic(field, item["value"]):
                item["value"] = None
                item["evidence_span"] = None
                nulled += 1
                hit = True
        papers += hit
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"records: {total}; values nulled: {nulled}; records affected: {papers}")

v4_ids = {json.loads(l)["id"] for l in CORPUS.open(encoding="utf-8") if l.strip()}
with ANN.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r["paper_id"] in v4_ids]
    fields = reader.fieldnames
with (OUT / "annotated_v4.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"annotation rows kept (in v4): {len(rows)}")