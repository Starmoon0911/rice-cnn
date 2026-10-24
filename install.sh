#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--device cpu|xpu] [--skip-dataset] [--skip-models]

Create/reuse .venv, install dependencies, download the Kaggle rice leaf dataset,
and cache all ImageNet weights used by src/pretrained_models.py.

Options:
  --device cpu|xpu  PyTorch wheel backend (default: cpu; xpu for Intel GPUs)
  --skip-dataset    Keep the dataset step disabled
  --skip-models     Keep the pretrained weight download step disabled
  -h, --help        Show this help

Environment:
  PYTHON            Python executable used to create .venv (default: python3)
  KAGGLE_API_TOKEN  Optional Kaggle token if the download requires authentication
  TORCH_HOME        Optional PyTorch cache location (also set it when training)

Requires Python 3.10+ and Bash (Linux, macOS, WSL, or Git Bash on Windows).
HELP
}

device=cpu
skip_dataset=0
skip_models=0
while (($#)); do
    case "$1" in
        --device)
            if (($# < 2)); then
                echo 'Error: --device requires cpu or xpu.' >&2
                exit 2
            fi
            device="$2"
            shift 2
            ;;
        --skip-dataset) skip_dataset=1; shift ;;
        --skip-models) skip_models=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$device" in
    cpu|xpu) ;;
    *) echo 'Error: --device must be cpu or xpu.' >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "Installation failed. Fix the error above and rerun bash install.sh with the same options." >&2' ERR

if [[ ! -d .venv ]]; then
    bootstrap_python="${PYTHON:-python3}"
    if ! command -v "$bootstrap_python" >/dev/null 2>&1; then
        echo "Error: $bootstrap_python not found. Install Python 3.10+ or set PYTHON." >&2
        exit 1
    fi
    "$bootstrap_python" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
    "$bootstrap_python" -m venv .venv
fi
if [[ -x .venv/bin/python ]]; then
    venv_python=.venv/bin/python
elif [[ -f .venv/Scripts/python.exe ]]; then
    venv_python=.venv/Scripts/python.exe
else
    echo 'Error: .venv has no Python interpreter for this OS. Use a compatible virtual environment.' >&2
    exit 1
fi
"$venv_python" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
"$venv_python" -m pip install --upgrade pip
torch_args=()
# macOS CPU wheels are distributed through PyPI, rather than the CPU index.
if [[ "$device" == xpu ]]; then
    torch_args=(--index-url https://download.pytorch.org/whl/xpu)
elif [[ "$(uname -s)" != Darwin ]]; then
    torch_args=(--index-url https://download.pytorch.org/whl/cpu)
fi
# Reinstall together so pip also switches an existing CPU/XPU wheel build.
"$venv_python" -m pip install --upgrade --force-reinstall torch torchvision "${torch_args[@]}"
"$venv_python" -m pip install -r requirements-install.txt

"$venv_python" - "$skip_dataset" "$skip_models" <<'PY'
from pathlib import Path
import shutil
import sys

from PIL import Image

classes = ('Bacterial leaf blight', 'Brown spot', 'Leaf smut')
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
target = Path('dataset/rice_leaf_diseases')


def validate_dataset(root):
    counts = {}
    for name in classes:
        folder = root / name
        images = [p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in extensions] if folder.is_dir() else []
        if not images:
            raise ValueError(f'Missing images in {folder}')
        for path in images:
            with Image.open(path) as image:
                image.verify()
        counts[name] = len(images)
    return counts


if sys.argv[1] != '1':
    if target.exists():
        # Never overwrite a local dataset, including a partial/broken download.
        try:
            counts = validate_dataset(target)
        except (ValueError, OSError) as error:
            raise SystemExit(f'{error}. Repair or move {target} and rerun.') from error
        print(f'Reusing dataset: {counts}', flush=True)
    else:
        import kagglehub

        try:
            source = Path(kagglehub.dataset_download('vbookshelf/rice-leaf-diseases'))
        except Exception as error:
            raise SystemExit(
                'Kaggle download failed. Check your network; if authentication is '
                'required, set KAGGLE_API_TOKEN or configure ~/.kaggle/kaggle.json. '
                f'Details: {error}'
            ) from error
        # Kaggle archives may put the class folders inside an extra directory.
        roots = [p.parent for p in source.rglob(classes[0]) if p.is_dir()]
        if len(roots) != 1:
            raise SystemExit(f'Expected one rice dataset root in {source}, found {len(roots)}.')
        source = roots[0]
        counts = validate_dataset(source)
        target.mkdir(parents=True)
        for name in classes:
            shutil.copytree(source / name, target / name)
        print(f'Dataset installed at {target}: {counts}', flush=True)

if sys.argv[2] != '1':
    from src.pretrained_models import MODEL_NAMES, weights_for

    seen = set()
    for name in MODEL_NAMES:
        weights = weights_for(name)
        if weights is None or weights.url in seen:
            continue
        seen.add(weights.url)
        print(f'Caching {name}: {weights}', flush=True)
        weights.get_state_dict(progress=True, check_hash=True)
    print(f'Cached {len(seen)} unique ImageNet weight files.', flush=True)

import torch
import torchvision
print(f'PyTorch {torch.__version__}; torchvision {torchvision.__version__}')
PY

printf '\nSetup complete. Train with:\n  %s src/compare_pretrained.py --device %s\n' "$venv_python" "$device"
