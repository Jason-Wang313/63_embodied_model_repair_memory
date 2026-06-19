from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


SPLITS = [
    "nominal",
    "low_friction_shift",
    "high_friction_shift",
    "heavy_object_shift",
    "light_object_shift",
    "actuation_bias_shift",
    "obstacle_shift",
    "nonstationary_deployment_shift",
    "combined_shift",
]

METHODS = [
    ("nominal_mpc", "Nominal"),
    ("robust_worst_case_mpc", "Robust"),
    ("global_average_repair", "Global"),
    ("knn_repair_memory_v4", "kNN v4"),
    ("online_ridge_residual", "Ridge"),
    ("corm_repair_memory_v5", "CORM"),
    ("oracle_hidden_deployment", "Oracle"),
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(text: str) -> str:
    return text.replace("_", r"\_")


def pct(value: str) -> str:
    return f"{100.0 * float(value):.1f}"


def num(value: str) -> str:
    return f"{float(value):.3f}"


def metric(metrics: list[dict[str, str]], split: str, method: str, key: str) -> str:
    for row in metrics:
        if row["split"] == split and row["method"] == method:
            return row[key]
    raise KeyError((split, method, key))


def table_success(metrics: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen CORM v5 success rates. Entries are percentages over 256 paired episodes per split.}",
        r"\label{tab:v5-success}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Split & " + " & ".join(label for _, label in METHODS) + r" \\",
        r"\midrule",
    ]
    for split in SPLITS:
        vals = [pct(metric(metrics, split, method, "success_rate")) for method, _ in METHODS]
        lines.append(f"{esc(split)} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_regret(metrics: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen CORM v5 energy regret relative to the hidden-deployment oracle. Lower is better.}",
        r"\label{tab:v5-regret}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Split & " + " & ".join(label for method, label in METHODS if method != "oracle_hidden_deployment") + r" \\",
        r"\midrule",
    ]
    for split in SPLITS:
        vals = [num(metric(metrics, split, method, "energy_regret_mean")) for method, _ in METHODS if method != "oracle_hidden_deployment"]
        lines.append(f"{esc(split)} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_safety(metrics: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen CORM v5 violation rates. Entries are percentages; lower is better.}",
        r"\label{tab:v5-safety}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Split & " + " & ".join(label for _, label in METHODS) + r" \\",
        r"\midrule",
    ]
    for split in SPLITS:
        vals = [pct(metric(metrics, split, method, "violation_rate")) for method, _ in METHODS]
        lines.append(f"{esc(split)} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_aggregate(aggregate: list[dict[str, str]]) -> str:
    order = [
        "random_candidate",
        "nominal_mpc",
        "robust_worst_case_mpc",
        "last_repair_memory",
        "global_average_repair",
        "knn_repair_memory_v4",
        "online_ridge_residual",
        "corm_repair_memory_v5",
        "corm_no_safety",
        "corm_no_uncertainty",
        "oracle_hidden_deployment",
    ]
    by_method = {row["method"]: row for row in aggregate}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aggregate frozen metrics across all nine main splits.}",
        r"\label{tab:v5-aggregate}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Success & Regret & Violation & Trust & Shield \\",
        r"\midrule",
    ]
    for method in order:
        row = by_method[method]
        lines.append(
            f"{esc(method)} & {pct(row['success_rate'])} & {num(row['energy_regret_mean'])} & "
            f"{pct(row['violation_rate'])} & {num(row['decision_trust_mean'])} & "
            f"{pct(row['shield_activation_rate'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_pairwise(pairwise: list[dict[str, str]]) -> str:
    keep = {
        "nominal_mpc",
        "robust_worst_case_mpc",
        "knn_repair_memory_v4",
        "online_ridge_residual",
        "oracle_hidden_deployment",
    }
    rows = [row for row in pairwise if row["split"] in {"combined_shift", "nonstationary_deployment_shift"} and row["baseline"] in keep]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired stress-test deltas for CORM v5. Positive success and regret-improvement favor CORM; negative violation deltas mean CORM is safer.}",
        r"\label{tab:v5-pairwise-stress}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Split & Baseline & Success $\Delta$ & Regret improv. & Violation $\Delta$ & $p_{\mathrm{regret}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['split'])} & {esc(row['baseline'])} & {num(row['success_delta_mean'])} & "
            f"{num(row['regret_improvement_mean'])} & {num(row['violation_delta_mean'])} & "
            f"{num(row['regret_signflip_p'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_ablation(ablation: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen ablations on combined and nonstationary shifts.}",
        r"\label{tab:v5-ablation}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Split & Method & Success & Regret & Violation & Trust & Shield \\",
        r"\midrule",
    ]
    for row in ablation:
        lines.append(
            f"{esc(row['split'])} & {esc(row['method'])} & {pct(row['success_rate'])} & "
            f"{num(row['energy_regret_mean'])} & {pct(row['violation_rate'])} & "
            f"{num(row['decision_trust_mean'])} & {pct(row['shield_activation_rate'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_metric_tables(metrics: list[dict[str, str]]) -> str:
    lines: list[str] = ["% Auto-generated detailed metric tables"]
    for split in SPLITS:
        rows = [row for row in metrics if row["split"] == split]
        lines.extend(
            [
                r"\begin{table}[p]",
                r"\centering",
                rf"\caption{{Detailed frozen metrics for {esc(split)}.}}",
                rf"\label{{tab:detail-{split.replace('_', '-')}}}",
                r"\small",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Method & Success & Regret & Distance & Violation & Trust & Pred. error \\",
                r"\midrule",
            ]
        )
        for row in rows:
            lines.append(
                f"{esc(row['method'])} & {pct(row['success_rate'])} & {num(row['energy_regret_mean'])} & "
                f"{num(row['final_distance_mean'])} & {pct(row['violation_rate'])} & "
                f"{num(row['decision_trust_mean'])} & {num(row['repair_prediction_error_mean'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_calibration_tables(calibration: list[dict[str, str]]) -> str:
    rows = [row for row in calibration if row["method"] == "corm_repair_memory_v5"]
    lines = [
        "% Auto-generated CORM calibration table",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{CORM v5 trust calibration by split and stream phase.}",
        r"\label{tab:corm-calibration}",
        r"\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Split & Phase & Method & Trust & Uncertainty & Pred. error & Shield \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['split'])} & {esc(row['phase'])} & CORM & {num(row['decision_trust_mean'])} & "
            f"{num(row['decision_uncertainty_mean'])} & {num(row['repair_prediction_error_mean'])} & "
            f"{pct(row['shield_activation_rate'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_pairwise_tables(pairwise: list[dict[str, str]]) -> str:
    lines: list[str] = ["% Auto-generated detailed pairwise statistics"]
    for split in SPLITS:
        rows = [row for row in pairwise if row["split"] == split]
        lines.extend(
            [
                r"\begin{table}[p]",
                r"\centering",
                rf"\caption{{Full paired CORM deltas for {esc(split)}. Positive success and regret-improvement favor CORM; negative violation deltas favor CORM.}}",
                rf"\label{{tab:pair-{split.replace('_', '-')}}}",
                r"\small",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Baseline & Success $\Delta$ & 95\% CI & Regret improv. & Violation $\Delta$ & $p_{\mathrm{regret}}$ \\",
                r"\midrule",
            ]
        )
        for row in rows:
            ci = f"[{num(row['success_delta_boot_lo'])}, {num(row['success_delta_boot_hi'])}]"
            lines.append(
                f"{esc(row['baseline'])} & {num(row['success_delta_mean'])} & {ci} & "
                f"{num(row['regret_improvement_mean'])} & {num(row['violation_delta_mean'])} & "
                f"{num(row['regret_signflip_p'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_learning_tables(learning: list[dict[str, str]]) -> str:
    lines: list[str] = ["% Auto-generated phase learning diagnostics"]
    for split in SPLITS:
        rows = [row for row in learning if row["split"] == split]
        lines.extend(
            [
                r"\begin{table}[p]",
                r"\centering",
                rf"\caption{{Early/middle/late learning diagnostics for {esc(split)}.}}",
                rf"\label{{tab:learn-{split.replace('_', '-')}}}",
                r"\scriptsize",
                r"\begin{tabular}{llrrrrr}",
                r"\toprule",
                r"Phase & Method & Success & Regret & Violation & Trust & Pred. error \\",
                r"\midrule",
            ]
        )
        for row in rows:
            lines.append(
                f"{esc(row['phase'])} & {esc(row['method'])} & {pct(row['success_rate'])} & "
                f"{num(row['energy_regret_mean'])} & {pct(row['violation_rate'])} & "
                f"{num(row['decision_trust_mean'])} & {num(row['repair_prediction_error_mean'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    metrics = read_csv("repair_memory_metrics.csv")
    aggregate = read_csv("repair_memory_aggregate.csv")
    pairwise = read_csv("repair_memory_pairwise.csv")
    ablation = read_csv("repair_memory_ablation.csv")
    calibration = read_csv("repair_memory_calibration.csv")
    learning = read_csv("repair_memory_learning_curve.csv")
    content = "\n".join(
        [
            "% Auto-generated by scripts/render_latex_tables.py",
            table_aggregate(aggregate),
            table_success(metrics),
            table_regret(metrics),
            table_safety(metrics),
            table_pairwise(pairwise),
            table_ablation(ablation),
        ]
    )
    (PAPER / "results_tables.tex").write_text(content, encoding="utf-8")
    appendix_content = "\n\n".join(
        [
            appendix_metric_tables(metrics),
            appendix_pairwise_tables(pairwise),
            appendix_learning_tables(learning),
            appendix_calibration_tables(calibration),
        ]
    )
    (PAPER / "appendix_results_tables.tex").write_text(appendix_content, encoding="utf-8")
    print(PAPER / "results_tables.tex")
    print(PAPER / "appendix_results_tables.tex")


if __name__ == "__main__":
    main()
