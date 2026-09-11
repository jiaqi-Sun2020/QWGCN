# QWGCN correction release (2026-09)

This document separates the historical repository baseline from the reviewed
physics/runtime correction. Git history is the canonical version boundary; the
repository does not duplicate the complete source tree into `old/` and `new/`
directories.

## Version boundary

| Version | Git reference | Meaning |
|---|---|---|
| Historical baseline | `qwgcn-original-202601` at `06614d6f0019df1dcd2e813ebe26764379eded59` | Preserves the previously published repository state, including the legacy positive first-order sign in `model/QWGCN4.py`. |
| Corrected release | `qwgcn-corrected-202609` | Applies the reviewed QWGCN physics/runtime corrections and adds reproducible audit code and review artifacts. |
| Default branch | `main` | Left unchanged during this release so the original state is not overwritten implicitly. |

## Corrected contract

The corrected fast QWGCN uses the symmetrically normalized adjacency with
self-loops, denoted by `A_tilde`, and the order-2 polynomial

```text
G2(t) = I - i t A_tilde - t^2 A_tilde^2 / 2.
```

The manuscript protocol fixes `t = 1`, diffusion strength `= 1`, and default
Taylor order `= 2`. Evolution weights are normalized independently for each
graph in a batch. Legacy checkpoint keys for the formerly trainable dynamics
scalars are ignored so they cannot silently change the corrected protocol.

## Original-versus-corrected summary

| Surface | Historical baseline | Corrected release |
|---|---|---|
| First-order term | `+ i t A_tilde X` | `- i t A_tilde X` |
| Second-order term | Negative through `(i t)^2 / 2` | Negative through `(-i t)^2 / 2`; unchanged mathematically |
| Evolution defaults | Layer order could be ignored; `t` was trainable and initialized to `0.5` | Requested order reaches every layer; fixed `t=1`, strength `=1`, default order `2` |
| Batch normalization of node weights | One sum across unrelated graphs | One independent sum per graph |
| Single-layer construction | Hidden/output dimensions could disagree | One-layer construction maps input directly to output |

## File index

### Corrected implementation

| Path | Role |
|---|---|
| [`model/QWGCN4.py`](../model/QWGCN4.py) | Corrected Taylor evolution, fixed protocol parameters, per-graph normalization, and one-layer construction. |
| [`model/model_get.py`](../model/model_get.py) | Passes evolution order, time, and diffusion configuration into QWGCN. |

### Verification and reproduction code

| Path | Role |
|---|---|
| [`diagnostics/verify_runtime_integrity.py`](../diagnostics/verify_runtime_integrity.py) | Portable deterministic runtime-contract test; no training required. |
| [`diagnostics/verify_evolution_sign.py`](../diagnostics/verify_evolution_sign.py) | First-principles and adversarial sign audit; requires the companion manuscript workspace. |
| [`diagnostics/verify_fig4_operator_error.py`](../diagnostics/verify_fig4_operator_error.py) | Fig. 4 complex-operator error checks. |
| [`diagnostics/verify_fig6_runtime_benchmark.py`](../diagnostics/verify_fig6_runtime_benchmark.py) | Fig. 6 timing-data and benchmark-contract checks. |
| [`diagnostics/verify_table2_reproducibility.py`](../diagnostics/verify_table2_reproducibility.py) | Table II provenance and rerun-contract checks. |
| [`diagnostics/verify_fig1_edge_count.py`](../diagnostics/verify_fig1_edge_count.py) | Fig. 1 adjacency/edge-count verification. |
| [`diagnostics/verify_fig2_text_consistency.py`](../diagnostics/verify_fig2_text_consistency.py) | Fig. 2, manuscript, and runtime consistency checks. |
| [`diagnostics/verify_project8_final_compile.py`](../diagnostics/verify_project8_final_compile.py) | Final manuscript PDF/build QA checks. |
| [`experiments/table2/run_table2.py`](../experiments/table2/run_table2.py) | Manifest-locked corrected-QWGCN Table II runner. |
| [`experiments/table2/audit_table2_evidence.py`](../experiments/table2/audit_table2_evidence.py) | Archived Table II evidence audit. |

### Review artifacts

| Path | Content |
|---|---|
| [`docs/review/QWGCN_Project_1-8_Review.pptx`](review/QWGCN_Project_1-8_Review.pptx) | Ten-slide review deck covering Projects 1–8. |
| [`docs/review/project1_evolution_sign_before_after.png`](review/project1_evolution_sign_before_after.png) | Evolution-sign before/after evidence. |
| [`docs/review/project2_runtime_integrity_before_after.png`](review/project2_runtime_integrity_before_after.png) | Runtime-integrity comparison. |
| [`docs/review/project3_fig4_operator_error_before_after.png`](review/project3_fig4_operator_error_before_after.png) | Fig. 4 operator-error comparison. |
| [`docs/review/project4_fig6_runtime_benchmark.png`](review/project4_fig6_runtime_benchmark.png) | Fig. 6 dense reference-kernel timing plot. |
| [`docs/review/project5_table2_before_after.png`](review/project5_table2_before_after.png) | Table II reproducibility evidence. |
| [`docs/review/project5_performance_comparison.png`](review/project5_performance_comparison.png) | Manuscript statistics versus archived final-test records. |
| [`docs/review/project6_fig1_edge_count_before_after.png`](review/project6_fig1_edge_count_before_after.png) | Fig. 1 edge-count correction. |
| [`docs/review/project7_fig2_text_consistency_before_after.png`](review/project7_fig2_text_consistency_before_after.png) | Fig. 2 text/runtime audit. |
| [`docs/review/project8_final_compile_before_after.png`](review/project8_final_compile_before_after.png) | Final manuscript compilation QA. |

## Verification

Run the portable corrected-runtime check from the repository root:

```powershell
python -B .\diagnostics\verify_runtime_integrity.py
```

The remaining diagnostics expose `--help` or document their required manuscript,
archive, or generated-evidence inputs. Those workspace-integrated checks should
not be reported as portable repository-only tests.

## Explicitly deferred items

- The nonunitary fidelity metric is not redefined or regenerated in this release.
- The small-graph dataset split is not changed; the known oversampling-before-split
  leakage remains deferred.
- Table II baseline models are not retrained. Only corrected QWGCN rerun evidence
  is retained, so this release does not claim a newly controlled seven-model ranking.
