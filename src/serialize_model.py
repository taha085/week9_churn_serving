"""Create joblib and skops serializations of the existing Week 7 pipeline.

No model fitting occurs here. skops may flag third-party XGBoost objects as
untrusted; inspect get_untrusted_types and only trust types from this artifact
when its origin and integrity are verified.
"""
from pathlib import Path
import json
import joblib
import skops.io as sio

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "churn_pipeline.joblib"
SKOPS_PATH = ROOT / "models" / "churn_pipeline.skops"

def main():
    pipeline = joblib.load(SOURCE)  # trusted, locally generated Week 7 artifact only
    sio.dump(pipeline, SKOPS_PATH)
    untrusted = sio.get_untrusted_types(file=SKOPS_PATH)
    (ROOT / "models" / "skops_untrusted_types.json").write_text(
        json.dumps(untrusted, indent=2)
    )
    print(f"Created {SKOPS_PATH}")
    print("Review models/skops_untrusted_types.json before loading.")
    print("Do not blindly trust unknown types; XGBoost may require explicit trust.")

if __name__ == "__main__":
    main()
