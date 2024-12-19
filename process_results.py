from pathlib import Path
import shutil

a = Path("/scratch-shared/scur2850/inpars")
tgt = Path("results")
results = a.rglob("results.json")

for r in results:
    print(r)
    new_name = tgt / f"{r.parts[4]}.json"
    shutil.copy(r, new_name)
    
