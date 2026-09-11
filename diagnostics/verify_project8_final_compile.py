"""Adversarial checks for the Project 8 manuscript build artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


BLOCKING_LOG_PATTERNS = {
    "undefined_citation": r"Citation .* undefined|There were undefined citations",
    "undefined_reference": r"Reference .* undefined|There were undefined references",
    "missing_character": r"Missing character",
    "overfull_box": r"Overfull \\hbox|Overfull \\vbox",
    "latex_error": r"LaTeX Error|Undefined control sequence|Emergency stop|Fatal error",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=11)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tex_path = args.tex.resolve()
    pdf_path = args.pdf.resolve()
    log_path = args.log.resolve()
    pages_dir = args.pages_dir.resolve()

    tex = tex_path.read_text(encoding="utf-8")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    reader = PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    rendered_pages = sorted(pages_dir.glob("page-*.png"))

    blocking_hits = {
        name: bool(re.search(pattern, log, flags=re.IGNORECASE))
        for name, pattern in BLOCKING_LOG_PATTERNS.items()
    }

    checks = {
        "pdf_page_count_is_expected": len(reader.pages) == args.expected_pages,
        "all_pdf_pages_rendered": len(rendered_pages) == len(reader.pages),
        "rendered_pages_are_nonempty": all(path.stat().st_size > 0 for path in rendered_pages),
        "latex_build_has_no_blocking_findings": not any(blocking_hits.values()),
        "source_has_no_placeholder_citation": "\\cite{xxx}" not in tex,
        "source_has_no_unrelated_chinese_placeholder": not re.search(
            r"不规则采样|图上的连续时间|前期工作", tex
        ),
        "source_restores_ctqw_section": "\\section{CTQW on a Graph}" in tex,
        "source_hamiltonian_reduces_to_A": (
            re.search(r"H\s*=\s*\\gamma\s*A", tex) is not None
            and re.search(r"\\gamma\s*=\s*1", tex) is not None
        ),
        "source_exact_propagator_has_negative_sign": (
            re.search(r"G\(t\)\s*=\s*e\^\{-iHt\}\s*=\s*e\^\{-iAt\}", tex)
            is not None
        ),
        "source_second_order_term_has_negative_first_order": (
            "I - i t \\tilde{A} - \\frac{t^2}{2} \\tilde{A}^2 = G_2(t)" in tex
        ),
        "pdf_contains_ctqw_section": "CTQW ON A GRAPH" in pdf_text,
        "pdf_contains_table_ii": (
            re.search(r"T\s*ABLE\s+II\.", pdf_text, flags=re.IGNORECASE) is not None
        ),
        "pdf_has_no_placeholder_text": not re.search(
            r"xxx|不规则采样|图上的连续", pdf_text, flags=re.IGNORECASE
        ),
        "pdf_has_no_unresolved_markers": "??" not in pdf_text and "[?]" not in pdf_text,
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "blocking_log_hits": blocking_hits,
        "metrics": {
            "pdf_pages": len(reader.pages),
            "rendered_pages": len(rendered_pages),
            "pdf_bytes": pdf_path.stat().st_size,
        },
        "hashes": {
            "tex_sha256": sha256(tex_path),
            "pdf_sha256": sha256(pdf_path),
            "log_sha256": sha256(log_path),
        },
        "paths": {
            "tex": str(tex_path),
            "pdf": str(pdf_path),
            "log": str(log_path),
            "pages_dir": str(pages_dir),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{report['status']}: {report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] == "FAIL":
        failed = [name for name, passed in checks.items() if not passed]
        print("failed=" + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
