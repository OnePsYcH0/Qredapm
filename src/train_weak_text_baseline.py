from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dataset import load_jsonl
from metrics import compute_binary_metrics


DEFAULT_TRAIN_FILE = "data/train_0.json"
DEFAULT_TEST_FILE = "data/test_0.json"
DEFAULT_OUTPUT_DIR = "outputs/weak_text_baseline"


def stringify_text_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def build_text(row: dict) -> str:
    disease_names = stringify_text_field(row.get("disease_names"))
    drug_names = stringify_text_field(row.get("drug_names"))
    visit_sn = stringify_text_field(row.get("visit_sn"))
    text = " ".join(part for part in [disease_names, drug_names, visit_sn] if part).strip()
    return text or "无文本记录"


def save_predictions_csv(path: Path, y_true, y_score, y_pred) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_score", "y_pred"])
        for index, (truth, score, pred) in enumerate(zip(y_true, y_score, y_pred)):
            writer.writerow([index, int(truth), f"{float(score):.10f}", int(pred)])


def update_comparison_csv(project_root: Path) -> None:
    mapping = {
        "weak_structured_lr": project_root / "weak_baseline_outputs" / "metrics.json",
        "weak_text_tfidf": project_root / "weak_text_outputs" / "metrics.json",
        "mlp_structured": project_root / "baseline_outputs" / "metrics.json",
        "bert_text": project_root / "text_baseline_outputs" / "metrics.json",
        "redapm": project_root / "redapm_outputs" / "metrics.json",
    }

    lines = ["model,roc_auc,pr_auc,accuracy,precision,recall,f1"]
    for model_name, path in mapping.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["test_metrics"]
        lines.append(
            ",".join(
                [
                    model_name,
                    str(metrics["roc_auc"]),
                    str(metrics["pr_auc"]),
                    str(metrics["accuracy"]),
                    str(metrics["precision"]),
                    str(metrics["recall"]),
                    str(metrics["f1_score"]),
                ]
            )
        )

    content = "\n".join(lines)
    (project_root / "model_comparison.csv").write_text(content, encoding="utf-8")
    (project_root / "weak_text_outputs" / "model_comparison.csv").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练弱文本 baseline：TF-IDF + LogisticRegression")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)

    x_train = [build_text(row) for row in train_rows]
    x_test = [build_text(row) for row in test_rows]
    y_train = [int(row["y2"]) for row in train_rows]
    y_test = [int(row["y2"]) for row in test_rows]

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=args.max_features)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_score = pipeline.predict_proba(x_test)[:, 1]
    y_pred = [1 if score >= 0.5 else 0 for score in y_score.tolist()]
    metrics = compute_binary_metrics(y_test, y_score.tolist())

    metrics_payload = {
        "model_name": "TFIDF+LogisticRegression",
        "label_column": "y2",
        "text_fields": ["disease_names", "drug_names", "visit_sn"],
        "drug_names_present_in_data": any("drug_names" in row for row in train_rows),
        "text_template": "disease_names + ' ' + drug_names + ' ' + visit_sn",
        "tfidf_max_features": args.max_features,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_metrics": metrics,
        "notes": "弱文本基线：TF-IDF vectorizer(max_features=10000) + LogisticRegression，不使用 BERT，不使用结构化特征。",
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions_csv(output_dir / "predictions.csv", y_test, y_score.tolist(), y_pred)

    update_comparison_csv(project_root)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
