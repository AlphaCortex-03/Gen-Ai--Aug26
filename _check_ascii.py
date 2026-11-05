import pathlib
import sys

bad_files = []
offenders = 0
for p in pathlib.Path(".").rglob("*.py"):
    if ".git" in p.parts or ".cache" in p.parts:
        continue
    txt = p.read_text(encoding="utf-8-sig")  # utf-8-sig strips the BOM if present
    if "\ufeff" in txt:
        bad_files.append((str(p), "has BOM inside file"))
    for i, line in enumerate(txt.splitlines(), 1):
        chars = [c for c in line if ord(c) > 127]
        if chars:
            print(f"{str(p)}:{i}: " + ascii(line.strip())[:140])
            offenders += 1
print("non-ascii scan done")
sys.exit(1 if offenders or bad_files else 0)
