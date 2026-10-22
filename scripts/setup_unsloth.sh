#!/usr/bin/env bash
set -x
cd "$(dirname "$0")/.."
PY=/usr/bin/python3.12
VENV=.venv-unsloth

$PY -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel setuptools

"$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install unsloth

"$VENV/bin/python" - <<'PYEOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "cc:", torch.cuda.get_device_capability(0))
try:
    import unsloth
    print("unsloth:", getattr(unsloth, "__version__", "?"))
except Exception as e:
    print("unsloth import FAIL:", e)
for mod in ["transformers", "peft", "trl", "bitsandbytes", "xformers", "triton", "datasets", "accelerate"]:
    try:
        m = __import__(mod)
        print(mod, ":", getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, ": MISSING")
PYEOF
echo "SETUP DONE"
