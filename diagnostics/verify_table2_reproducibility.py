"""Adversarial verification for Project 5 (Table II evidence and rerun contract)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


MODELS = ["GAT", "GCN", "GraphSAGE", "multi_GAT", "TransformerConv", "Res_SAGE", "QWGCN"]
EXPECTED_ARCHIVE = {
    ("GAT", "heads=1, params=650"): (0.6280, 0.3174, 0.6826),
    ("GCN", "params=518"): (0.6289, 0.3293, 0.6707),
    ("GraphSAGE", "params=838"): (0.5951, 0.3473, 0.6527),
    ("multi_GAT", "heads=8, params=12326"): (0.6280, 0.3174, 0.6826),
    ("multi_GAT", "heads=2, params=1550"): (0.6196, 0.3174, 0.6826),
    ("TransformerConv", "heads=1, params=1676"): (0.6170, 0.3473, 0.6527),
    ("Res_SAGE", "params=1224"): (0.6069, 0.3473, 0.6527),
    ("QWGCN", "params=1884"): (0.6436, 0.3293, 0.6707),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def almost_equal(left: float, right: float, tolerance: float = 5e-8) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def metric_from_predictions(path: Path) -> tuple[float, float, float]:
    rows = read_csv(path)
    correct = sum(row["correct"].lower() == "true" for row in rows)
    accuracy = correct / len(rows)
    error_rate = 1.0 - accuracy
    mse = sum(
        (int(row["label"]) - float(row["probability_class_1"])) ** 2 for row in rows
    ) / len(rows)
    return accuracy, error_rate, mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--paper-tex", type=Path, required=True)
    parser.add_argument("--before-tex", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--smoke-a", type=Path, required=True)
    parser.add_argument("--smoke-b", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    audit_json = json.loads((args.audit_output / "table2_audit.json").read_text(encoding="utf-8"))
    archived_rows = read_csv(args.audit_output / "archived_final_tests.csv")
    comparison_rows = read_csv(args.audit_output / "paper_vs_archive_metrics.csv")
    reproducibility_rows = read_csv(args.audit_output / "reproducibility_matrix.csv")

    check("paper rows", audit_json["counts"]["paper_rows"] == 7, str(audit_json["counts"]))
    check("archive text-log count", audit_json["archive"]["text_log_count"] == 9, str(audit_json["archive"]))
    check("valid final-test count", audit_json["counts"]["valid_final_tests"] == 8, str(audit_json["counts"]))
    check("failed-log count", audit_json["counts"]["failed_logs"] == 1, str(audit_json["counts"]))
    check("no documented seeds", audit_json["counts"]["documented_seed_logs"] == 0, str(audit_json["counts"]))
    check(
        "mean/SD verdict blocked",
        audit_json["verdicts"]["table2_mean_sd_reproducible"] is False,
        str(audit_json["verdicts"]),
    )
    check(
        "archive is unsafe as manuscript replacement",
        audit_json["verdicts"]["safe_to_copy_archived_values_into_manuscript"] is False,
        str(audit_json["verdicts"]),
    )
    check(
        "archived ranking is mixed",
        audit_json["verdicts"]["qwgcn_best_f1_in_archived_snapshot"] is True
        and audit_json["verdicts"]["qwgcn_best_mse_in_archived_snapshot"] is False
        and audit_json["verdicts"]["qwgcn_best_accuracy_in_archived_snapshot"] is False,
        str(audit_json["verdicts"]),
    )
    check(
        "binary MSE/accuracy identity diagnosed",
        audit_json["verdicts"]["archived_mse_equals_one_minus_accuracy"] is True
        and audit_json["verdicts"]["paper_mse_accuracy_means_are_complementary"] is False
        and audit_json["verdicts"]["paper_mse_accuracy_sds_are_consistent"] is False,
        str(audit_json["verdicts"]),
    )
    identity_audit = audit_json["mse_accuracy_identity_audit"]
    check(
        "paper identity failures localized",
        identity_audit["paper_mean_identity_failures"] == ["TransformerConv"]
        and [row["model"] for row in identity_audit["paper_sd_mismatches"]]
        == ["GCN", "GraphSAGE", "TransformerConv"],
        str(identity_audit),
    )

    valid_rows = [row for row in archived_rows if row["has_final_test"].lower() == "true"]
    observed_archive = {
        (row["model"], row["variant"]): (float(row["f1"]), float(row["mse"]), float(row["acc"]))
        for row in valid_rows
    }
    exact_archive = set(observed_archive) == set(EXPECTED_ARCHIVE) and all(
        all(almost_equal(actual, expected) for actual, expected in zip(observed_archive[key], expected_values))
        for key, expected_values in EXPECTED_ARCHIVE.items()
    )
    archive_detail = [
        {"model": key[0], "variant": key[1], "metrics": values}
        for key, values in sorted(observed_archive.items())
    ]
    check("archived metrics exact", exact_archive, json.dumps(archive_detail, indent=2))
    multi_variants = {row["variant"] for row in valid_rows if row["model"] == "multi_GAT"}
    check(
        "MultiGAT architectures not pooled",
        multi_variants == {"heads=8, params=12326", "heads=2, params=1550"},
        str(sorted(multi_variants)),
    )
    check("24 metric comparisons", len(comparison_rows) == 24, f"rows={len(comparison_rows)}")
    check(
        "all seven rows blocked",
        len(reproducibility_rows) == 7
        and all(row["mean_sd_reproducible"].lower() == "false" for row in reproducibility_rows),
        str([(row["model"], row["mean_sd_reproducible"]) for row in reproducibility_rows]),
    )
    check(
        "audit excludes QWGNN archive",
        all("train_data\\QWGCN" in row["log_path"] and "train_data\\QWGNN" not in row["log_path"] for row in archived_rows),
        "Every audited log must come from train_data/QWGCN.",
    )

    paper_hash = sha256_file(args.paper_tex)
    before_hash = sha256_file(args.before_tex)
    check(
        "unsupported manuscript numbers were not overwritten",
        paper_hash == before_hash == audit_json["manuscript"]["sha256"],
        f"live={paper_hash}, before={before_hash}, audited={audit_json['manuscript']['sha256']}",
    )

    supplied_root = args.project_root.resolve()
    code_root = supplied_root / "project_ai" if (supplied_root / "project_ai").is_dir() else supplied_root
    runner_path = code_root / "experiments/table2/run_table2.py"
    audit_script_path = code_root / "experiments/table2/audit_table2_evidence.py"
    runner_text = runner_path.read_text(encoding="utf-8")
    audit_text = audit_script_path.read_text(encoding="utf-8")
    check(
        "runner bypasses deferred small-graph splitter",
        "from dataset.dataset_split" not in runner_text
        and "import dataset_split" not in runner_text
        and not re.search(r"(?<!def )dataset_split\(", runner_text),
        "The Table II runner must implement only the full-PROTEINS split.",
    )
    check(
        "best checkpoint restored before test",
        bool(re.search(r"model\.load_state_dict\(best_state\).*?test_metrics\s*=\s*evaluate", runner_text, flags=re.DOTALL)),
        "Static contract: restore best_state before test evaluation.",
    )
    check(
        "sample SD aggregation",
        "values.std(ddof=1)" in runner_text,
        "Aggregation must use sample SD across independent seeds.",
    )
    check(
        "new rerun uses probability MSE",
        "mean_squared_error(truth, probability_class_1)" in runner_text
        and '"error_rate": 1.0 - accuracy' in runner_text,
        "MSE must measure probability error; hard-label error rate is stored separately.",
    )
    check(
        "audit rejects checkpoint pseudo-replication",
        "never aggregates checkpoints" in audit_text,
        "Checkpoints from one trajectory must not be treated as independent runs.",
    )

    manifests = [
        json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
        for root in (args.smoke_a, args.smoke_b)
    ]
    split_manifests = [
        json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
        for root in (args.smoke_a, args.smoke_b)
    ]
    check(
        "smoke protocol hashes match",
        manifests[0]["protocol_hash"] == manifests[1]["protocol_hash"],
        f"{manifests[0]['protocol_hash']} vs {manifests[1]['protocol_hash']}",
    )
    split_counts = split_manifests[0]["partitions"]
    check(
        "fixed split sizes",
        split_counts["train"]["count"] == 779
        and split_counts["validation"]["count"] == 167
        and split_counts["test"]["count"] == 167,
        str(split_counts),
    )
    check(
        "repeat split manifests identical",
        split_manifests[0] == split_manifests[1],
        f"hashes={sha256_file(args.smoke_a / 'split_manifest.json')}, {sha256_file(args.smoke_b / 'split_manifest.json')}",
    )

    result_checks = []
    prediction_hash_checks = []
    recomputation_checks = []
    for model in MODELS:
        path_a = args.smoke_a / "runs" / model / "seed_42" / "result.json"
        path_b = args.smoke_b / "runs" / model / "seed_42" / "result.json"
        result_a = json.loads(path_a.read_text(encoding="utf-8"))
        result_b = json.loads(path_b.read_text(encoding="utf-8"))
        result_checks.append(
            result_a["status"] == "complete"
            and result_b["status"] == "complete"
            and result_a["test"]["sample_count"] == 167
            and result_b["test"]["sample_count"] == 167
            and result_a["protocol_hash"] == manifests[0]["protocol_hash"]
            and result_b["protocol_hash"] == manifests[1]["protocol_hash"]
        )
        predictions_a = args.smoke_a / "runs" / model / "seed_42" / "test_predictions.csv"
        predictions_b = args.smoke_b / "runs" / model / "seed_42" / "test_predictions.csv"
        prediction_hash_checks.append(sha256_file(predictions_a) == sha256_file(predictions_b))
        accuracy, error_rate, mse = metric_from_predictions(predictions_a)
        recomputation_checks.append(
            almost_equal(accuracy, result_a["test"]["accuracy"])
            and almost_equal(mse, result_a["test"]["mse"])
            and almost_equal(error_rate, result_a["test"]["error_rate"])
        )
        result_checks.append(
            (args.smoke_a / "runs" / model / "seed_42" / "best.pt").exists()
            and (args.smoke_a / "runs" / model / "seed_42" / "last.pt").exists()
        )

    check("all seven smoke runs complete", all(result_checks), str(result_checks))
    check("repeat predictions byte-identical", all(prediction_hash_checks), str(prediction_hash_checks))
    check("test metrics recompute from predictions", all(recomputation_checks), str(recomputation_checks))
    check(
        "one-seed smoke SD is explicitly null",
        all(
            row[field] in ("", None)
            for row in read_csv(args.smoke_a / "table2_aggregate.csv")
            for field in (
                "f1_sample_sd",
                "mse_sample_sd",
                "mae_sample_sd",
                "accuracy_sample_sd",
                "error_rate_sample_sd",
            )
        ),
        "Smoke results must not invent standard deviations.",
    )

    passed = sum(item["passed"] for item in checks)
    verification = {
        "schema_version": 1,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
    }
    args.verification_output.mkdir(parents=True, exist_ok=True)
    (args.verification_output / "table2_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Table II project verification",
        "",
        f"Result: **{'PASS' if verification['all_passed'] else 'FAIL'} — {passed}/{len(checks)} checks passed.**",
        "",
    ]
    for item in checks:
        lines.append(f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['detail']}")
    (args.verification_output / "table2_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'PASS' if verification['all_passed'] else 'FAIL'}: {passed}/{len(checks)} checks passed")
    if not verification["all_passed"]:
        for item in checks:
            if not item["passed"]:
                print(f"FAILED: {item['name']} — {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
