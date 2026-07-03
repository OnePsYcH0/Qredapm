from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import LABEL_COLUMN, infer_feature_columns, load_jsonl, rows_to_tensors
from metrics import compute_binary_metrics


DEFAULT_TRAIN_FILE = "data/train_0.json"
DEFAULT_TEST_FILE = "data/test_0.json"
DEFAULT_OUTPUT_DIR = "outputs/baseline"


def save_predictions_csv(path: Path, y_true, y_score, y_pred) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_score", "y_pred"])
        for index, (truth, score, pred) in enumerate(zip(y_true, y_score, y_pred)):
            writer.writerow([index, int(truth), f"{float(score):.10f}", int(pred)])


def main() -> None:
    parser = argparse.ArgumentParser(description="璁粌涓€涓洿寮便€佹洿浼犵粺鐨勯€昏緫鍥炲綊 baseline銆?)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)
    feature_columns = infer_feature_columns(train_rows)

    x_train, y_train = rows_to_tensors(train_rows, feature_columns, label_column=LABEL_COLUMN)
    x_test, y_test = rows_to_tensors(test_rows, feature_columns, label_column=LABEL_COLUMN)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(x_train.numpy(), y_train.numpy())

    y_score = pipeline.predict_proba(x_test.numpy())[:, 1]
    metrics = compute_binary_metrics(y_test.tolist(), y_score.tolist())
    y_pred = [1 if score >= 0.5 else 0 for score in y_score.tolist()]

    metrics_payload = {
        "model_name": "LogisticRegression",
        "label_column": LABEL_COLUMN,
        "feature_count": len(feature_columns),
        "feature_columns_path": str(output_dir / "feature_columns.json"),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_metrics": metrics,
        "notes": "寮卞熀绾匡細浠呬娇鐢ㄧ粨鏋勫寲鐗瑰緛锛屾ā鍨嬩负鏍囧噯鍖?+ 閫昏緫鍥炲綊锛屼笉浣跨敤鏂囨湰銆佷笉浣跨敤 BERT銆佷笉浣跨敤娣卞眰绁炵粡缃戠粶銆?,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions_csv(output_dir / "predictions.csv", y_test.tolist(), y_score.tolist(), y_pred)

    print(json.dumps(metrics_payload["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

