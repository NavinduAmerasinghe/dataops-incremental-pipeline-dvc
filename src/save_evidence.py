from pathlib import Path
import glob
import shutil

repo = Path(__file__).resolve().parents[1]
out = repo / 'evidence'
out.mkdir(exist_ok=True)

# dvc status fallback
dvcs = sorted(glob.glob(str(repo / '**' / '*.dvc'), recursive=True))
with (out / 'dvc_status.txt').open('w', encoding='utf-8') as f:
    if not dvcs:
        f.write('No .dvc files found')
    else:
        for p in dvcs:
            f.write(p + '\n')
            f.write(Path(p).read_text(encoding='utf-8'))
            f.write('\n---\n')

# copy manifest
manifest = repo / 'data' / 'bronze' / 'manifest.csv'
if manifest.exists():
    shutil.copy(manifest, out / 'manifest.csv')

# write folder tree
with (out / 'folder_tree.txt').open('w', encoding='utf-8') as f:
    for p in sorted((repo / 'data').rglob('*')):
        rel = p.relative_to(repo)
        f.write(str(rel) + ('/' if p.is_dir() else '') + '\n')

print('saved evidence to', out)
