from pathlib import Path

_dir = Path(__file__).parent

# makes the subdirectory a sub-package and loads the .txt files as variables
for _f in sorted(_dir.glob("*.txt")):
    globals()[_f.stem] = _f.read_text()

