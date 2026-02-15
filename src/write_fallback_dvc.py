from pathlib import Path
import hashlib

repo_root = Path(__file__).resolve().parents[1]
data_root = repo_root / "data"

targets = [
    data_root / "bronze" / "bronze_all.csv",
    data_root / "silver" / "silver_all.csv",
    data_root / "gold" / "gold.csv",
]

for t in targets:
    if t.exists():
        h = hashlib.md5()
        with t.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        checksum = h.hexdigest()
        dvc_file = Path(str(t) + ".dvc")
        doc = f"outs:\n- md5: {checksum}\n  path: {t.name}\n"
        try:
            with dvc_file.open("w", encoding="utf-8") as f:
                f.write(doc)
            print(f"Wrote fallback .dvc for {t}: {dvc_file}")
        except Exception as e:
            print("Failed to write .dvc for", t, e)
    else:
        print("Target missing, skipping .dvc:", t)

print("fallback .dvc write complete")
