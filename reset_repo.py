from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
for folder in ["repository", "logs", "uploads"]:
    path = ROOT / folder
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
print("✅ Repository, logs, and uploads cleared. Fresh start!")
