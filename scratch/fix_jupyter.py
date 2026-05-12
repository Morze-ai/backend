import json

import Path


def main():
    with Path.open("notebooks/seaData.ipynb") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "import sys\n" in cell["source"]:
            cell["source"].insert(0, "%autoreload 2\n")
            cell["source"].insert(0, "%load_ext autoreload\n")
            break

    with Path.open("notebooks/seaData.ipynb", "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    main()
