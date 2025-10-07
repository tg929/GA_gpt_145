from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import QED
except ImportError:
    Chem = None
    QED = None

print("Test script running", flush=True)
