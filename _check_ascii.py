import pathlib

bad_files = []
for p in pathlib.Path(".").rglob("*.py"):
    if ".cache" in str(p):
        continue
    txt = p.read_text(encoding="utf-8-sig")
    if "\ufeff" in txt:
        bad_files.append((str(p), "has BOM inside file"))
    for i, line in enumerate(txt.splitlines(), 1):
        chars = [c for c in line if ord(c) > 127]
        if chars:
            print(f"{str(p)}:{i}: " + ascii(line.strip())[:140])
print("non-ascii scan done")
