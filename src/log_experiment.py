"""Log the frozen Week 7 experiment to a local MLflow file store."""
from pathlib import Path
import json
import joblib
import mlflow

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline.joblib"
METRICS = ROOT / "artifacts" / "week7_model_metrics.json"
THRESHOLD = 0.3251224160194397

def main():
    report = json.loads(METRICS.read_text())
    pipeline = joblib.load(MODEL)  # trusted local artifact only
    classifier = pipeline.named_steps["classifier"]
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("telco-churn-week7-final")
    with mlflow.start_run(run_name="xgboost-scale-pos-weight-frozen"):
        mlflow.log_params({
            "model": "XGBClassifier",
            "imbalance_strategy": report["selected_strategy"],
            "threshold": THRESHOLD,
            "random_state": 42,
            "cv_folds": report["cv"]["folds"],
            "cv_scoring": report["cv"]["scoring"],
            "n_estimators": classifier.n_estimators,
            "max_depth": classifier.max_depth,
            "learning_rate": float(classifier.learning_rate),
            "min_child_weight": classifier.min_child_weight,
            "subsample": float(classifier.subsample),
            "colsample_bytree": float(classifier.colsample_bytree),
            "scale_pos_weight": float(classifier.scale_pos_weight),
            "reg_lambda": float(classifier.reg_lambda),
        })
        metrics = report["holdout_metrics_at_selected_threshold"]
        mlflow.log_metrics({
            "holdout_precision": metrics["precision"],
            "holdout_recall": metrics["recall"],
            "holdout_f1": metrics["f1"],
            "holdout_roc_auc": metrics["roc_auc"],
            "holdout_pr_auc": metrics["pr_auc_average_precision"],
            "holdout_accuracy": metrics["accuracy"],
        })
        mlflow.log_artifact(str(MODEL), artifact_path="model_artifact")
        mlflow.log_artifact(str(METRICS), artifact_path="evaluation")
        mlflow.set_tag("training_repeated", "false")
        mlflow.set_tag("source", "Week 7 frozen pipeline")
    print(f"MLflow run logged locally at {ROOT / 'mlruns'}")

if __name__ == "__main__":
    main()
