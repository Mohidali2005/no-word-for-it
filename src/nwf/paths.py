import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data"
RAW = DATA/"raw"
# the authors code sits here as a sparse clone at the pinned commit
AUTHORS = RAW/"vnw_repo"
AUTHORS_SHAPES = AUTHORS/"2D_shape_recognition"
# kaggle mounts the stimuli read only and the tests use small copies so these folders can be moved
STIMULI = Path(os.environ.get("NWF_STIMULI",DATA/"stimuli"))
RESULTS = Path(os.environ.get("NWF_RESULTS",ROOT/"results"))
ANALYSIS = ROOT/"analysis"
# tests point this at tiny random models built from the real configs
LOCAL_MODELS = os.environ.get("NWF_LOCAL_MODELS")
# per item vision outputs for the precision check are large so they stay outside the saved results
SCRATCH = Path(os.environ.get("NWF_SCRATCH",ROOT/"cache"))
CAPTIONS = Path(os.environ.get("NWF_CAPTIONS",ROOT/"assets"/"captions.txt"))
STIMULI_MANIFEST = ROOT/"assets"/"stimuli_manifest.json"
