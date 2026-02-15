from pathlib import Path
import pandas as pd
import glob
import sys

repo_root = Path(__file__).resolve().parents[1]
root = repo_root / 'data'

print('--- DIRS ---')
for d in ['bronze','silver','gold','incoming','_staging_batches']:
    p = root / d
    if p.exists():
        files = sorted([f.name for f in p.iterdir()])
        print(f"{p}: {files}")
    else:
        print(f"{p}: MISSING")

print('\n--- MANIFEST ---')
man = root / 'bronze' / 'manifest.csv'
if man.exists():
    print(man.read_text(encoding='utf-8'))
else:
    print('manifest.csv not found')

print('\n--- DVC FILES ---')
dvcs = list(glob.glob('**/*.dvc', recursive=True))
if dvcs:
    for f in dvcs:
        print(f)
        print(Path(f).read_text(encoding='utf-8'))
else:
    print('No .dvc files found')

print('\n--- COUNTS ---')
for name,fn in [('bronze_all', root / 'bronze' / 'bronze_all.csv'),
               ('silver_all', root / 'silver' / 'silver_all.csv'),
               ('rejected', root / 'silver' / 'rejected_rows.csv'),
               ('gold', root / 'gold' / 'gold.csv')]:
    if fn.exists():
        try:
            df = pd.read_csv(fn)
            print(f"{fn}: {len(df)} rows")
        except Exception as e:
            print(f"{fn}: exists but read error: {e}")
    else:
        print(f"{fn}: MISSING")

# show first 5 lines of each master file if present
print('\n--- SAMPLES ---')
for fn in [root / 'bronze' / 'bronze_all.csv', root / 'silver' / 'silver_all.csv', root / 'silver' / 'rejected_rows.csv', root / 'gold' / 'gold.csv']:
    if fn.exists():
        print(f"\n{fn} sample:")
        with fn.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i>=6:
                    break
                print(line.rstrip())
    else:
        print(f"\n{fn}: MISSING")

print('\nEVIDENCE_COLLECTION_DONE')
