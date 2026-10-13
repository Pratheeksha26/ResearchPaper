import json, collections as C
o = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
a = o("corpus_main_v4.jsonl")
b = o("repro/corpus_live/corpus_main_v4.jsonl")
A = {r["id"] for r in a}
B = {r["id"] for r in b}
print("analysed:", len(A), "| live rebuild:", len(B), "| shared:", len(A & B), f"({100*len(A & B)/len(A):.1f}%)")
tot = C.Counter(r["year"] for r in a)
shr = C.Counter(r["year"] for r in a if r["id"] in B)
for y in sorted(tot):
    print(y, shr[y], "of", tot[y], f"({100*shr[y]/tot[y]:.0f}%)")
e15 = sum(shr[y] for y in (2015, 2016, 2017)); t15 = sum(tot[y] for y in (2015, 2016, 2017))
print("2015-2017:", e15, "of", t15, f"({100*e15/t15:.0f}%)")
