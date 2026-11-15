#!/usr/bin/env python3
"""Batch validation for the ten affine-analysis targets beside azimuth.

The test is intentionally about fp_affine_error.py itself: each source is
analysed, its JSON report is checked for structural/numeric consistency, and
its Perci-Tuner order file is checked.  Model-quality concerns are reported as
warnings because a conservative or incomplete model is different from a tool
execution failure.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# azimuth is deliberately absent: it was validated separately.  Keeping this
# list explicit makes this test continue to mean "the other ten programs" if
# more directories are later added under affine_percituner_test/1e-4.
PROGRAMS = {
    "bessel_j0": "bessel_j0/bessel.c",
    "bessel_j1": "bessel_j1/bessel_j1.c",
    "ellint": "ellint/ellint.c",
    "funarc": "funarc/funarc.c",
    "hartman": "hartman/hartman.c",
    "instantaneous_current": "instantaneous_current/instantaneous_current.c",
    "odometry": "odometry/odometry.c",
    "pendulum": "pendulum/pendulum.c",
    "simpsons": "simpsons/simpsons.c",
    "transport": "transport/transport.c",
}

REQUIRED_BASELINE_FIELDS = (
    "price_center",
    "price_lower",
    "price_upper",
    "price_radius",
    "target_coeff",
    "roundoff_radius",
    "random_radius",
    "nonaffine_remainder_radius",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description="Validate fp_affine_error.py on ten 1e-4 benchmark programs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=root / "affine_percituner_test" / "1e-4",
        help="directory containing the ten program directories",
    )
    parser.add_argument(
        "--tool",
        type=Path,
        default=root / "fp_affine_error.py",
        help="fp_affine_error.py path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "affine-test-results" / stamp,
        help="new directory for JSON reports, order files, and summary.json",
    )
    parser.add_argument(
        "--program",
        action="append",
        choices=tuple(PROGRAMS),
        help="validate one named program; repeat as needed (default: all ten)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="per-program analysis timeout in seconds",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0e-4,
        help="advisory limit for maximum one-variable local_relative",
    )
    parser.add_argument(
        "--order-direction",
        choices=("asc", "desc"),
        default="asc",
        help="order-file direction; asc tries low-sensitivity variables first",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="return failure if any model-quality warning is produced",
    )
    return parser.parse_args()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_order_file(path: Path, direction: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return ["variable_order.txt was not created"]

    entries = []
    sensitivities = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split("#", 1)[0].strip()
        if key:
            entries.append(key)
        marker = "sensitivity="
        if marker not in stripped:
            errors.append(f"order entry has no sensitivity: {stripped}")
            continue
        try:
            sensitivities.append(float(stripped.split(marker, 1)[1].split()[0]))
        except ValueError:
            errors.append(f"invalid order sensitivity: {stripped}")

    if not entries:
        errors.append("variable_order.txt contains no entries")
    if len(entries) != len(set(entries)):
        errors.append("variable_order.txt contains duplicate keys")
    if any(not math.isfinite(value) or value < 0.0 for value in sensitivities):
        errors.append("variable_order.txt contains invalid sensitivities")

    pairs = zip(sensitivities, sensitivities[1:])
    if direction == "asc" and any(left > right for left, right in pairs):
        errors.append("variable_order.txt is not ascending")
    elif direction == "desc" and any(left < right for left, right in pairs):
        errors.append("variable_order.txt is not descending")
    return errors


def validate_report(
    report: Any, order_path: Path, direction: str, threshold: float
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    if not isinstance(report, dict):
        return ["top-level JSON value is not an object"], warnings, metrics

    fp_vars = report.get("floating_variables")
    rows = report.get("sensitivity")
    baseline = report.get("baseline")
    if not isinstance(fp_vars, list) or not fp_vars:
        errors.append("floating_variables is missing or empty")
    if not isinstance(rows, list):
        errors.append("sensitivity is not a list")
        rows = []
    if not isinstance(baseline, dict):
        errors.append("baseline is not an object")
        baseline = {}

    if isinstance(fp_vars, list) and isinstance(rows, list) and len(rows) != len(fp_vars):
        errors.append(f"sensitivity row count {len(rows)} != floating variable count {len(fp_vars)}")

    for field in REQUIRED_BASELINE_FIELDS:
        value = baseline.get(field)
        if not finite_number(value):
            errors.append(f"baseline.{field} is missing or non-finite")
    if all(finite_number(baseline.get(field)) for field in REQUIRED_BASELINE_FIELDS):
        lower = float(baseline["price_lower"])
        center = float(baseline["price_center"])
        upper = float(baseline["price_upper"])
        radius = float(baseline["price_radius"])
        if not lower <= center <= upper:
            errors.append("baseline center is outside [price_lower, price_upper]")
        if radius < 0.0:
            errors.append("baseline radius is negative")
        expected_radius = max(center - lower, upper - center)
        # Subtracting two endpoint values near ``center`` loses about one ULP.
        endpoint_ulp_tolerance = (
            2.0 * sys.float_info.epsilon * max(abs(center), 1.0e-300)
        )
        if not math.isclose(
            radius,
            expected_radius,
            rel_tol=1.0e-9,
            abs_tol=endpoint_ulp_tolerance,
        ):
            errors.append("baseline radius is inconsistent with its bounds")

        scale = max(abs(center), 1.0e-300)
        relative_radius = radius / scale
        metrics.update(
            baseline_center=center,
            baseline_radius=radius,
            baseline_relative_radius=relative_radius,
            random_radius=float(baseline["random_radius"]),
        )
        if float(baseline["random_radius"]) > 0.0:
            warnings.append("baseline contains unknown/input affine uncertainty")
        if relative_radius > 0.5:
            warnings.append(f"baseline relative radius is large ({relative_radius:.3e})")
        if radius == 0.0 and fp_vars:
            warnings.append("baseline radius is zero despite floating-point variables")

    valid_sensitivities = []
    local_relatives = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"sensitivity row {index} is not an object")
            continue
        if "error" in row:
            errors.append(f"{row.get('name', index)} analysis error: {row['error']}")
            continue
        sensitivity = row.get("sensitivity_radius")
        local_relative = row.get("local_relative")
        if not finite_number(sensitivity) or float(sensitivity) < 0.0:
            errors.append(f"{row.get('name', index)} has invalid sensitivity_radius")
        else:
            valid_sensitivities.append(float(sensitivity))
        if not finite_number(local_relative) or float(local_relative) < 0.0:
            errors.append(f"{row.get('name', index)} has invalid local_relative")
        else:
            local_relatives.append(float(local_relative))

    if any(left < right for left, right in zip(valid_sensitivities, valid_sensitivities[1:])):
        errors.append("JSON sensitivity rows are not descending")
    if local_relatives:
        maximum = max(local_relatives)
        metrics["max_local_relative"] = maximum
        if maximum > threshold:
            warnings.append(
                f"maximum one-variable local_relative {maximum:.3e} exceeds {threshold:.3e}"
            )

    errors.extend(check_order_file(order_path, direction))
    metrics["model"] = report.get("model", "unknown")
    metrics["variable_count"] = len(fp_vars) if isinstance(fp_vars, list) else 0
    return errors, warnings, metrics


def run_one(
    name: str,
    source: Path,
    tool: Path,
    output_root: Path,
    timeout: float,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    program_dir = output_root / name
    program_dir.mkdir(parents=True, exist_ok=False)
    report_path = program_dir / "report.json"
    stderr_path = program_dir / "stderr.log"
    order_path = program_dir / "variable_order.txt"

    command = [
        sys.executable,
        str(tool),
        str(source),
        "--json",
        "--order-out",
        str(order_path),
        "--order-direction",
        direction,
    ]
    (program_dir / "command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    started = datetime.now()
    try:
        completed = subprocess.run(
            command,
            cwd=tool.parent,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        report_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "program": name,
            "source": str(source),
            "status": "FAIL",
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
            "errors": [f"analysis exceeded {timeout:g}s timeout"],
            "warnings": [],
        }

    report_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    if completed.returncode != 0:
        errors.append(f"fp_affine_error.py exited with code {completed.returncode}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"stdout is not valid JSON: {exc}")
    else:
        report_errors, warnings, metrics = validate_report(
            report, order_path, direction, threshold
        )
        errors.extend(report_errors)

    return {
        "program": name,
        "source": str(source),
        "status": "PASS" if not errors else "FAIL",
        "elapsed_seconds": (datetime.now() - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
        **metrics,
    }


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    tool = args.tool.resolve()
    output_dir = args.output_dir.resolve()
    selected = args.program or list(PROGRAMS)

    if args.timeout <= 0.0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    if not finite_number(args.threshold) or args.threshold <= 0.0:
        print("error: --threshold must be a positive finite number", file=sys.stderr)
        return 2
    if not tool.is_file():
        print(f"error: affine tool not found: {tool}", file=sys.stderr)
        return 2
    if output_dir.exists():
        print(f"error: output directory already exists: {output_dir}", file=sys.stderr)
        return 2

    sources = {name: base_dir / PROGRAMS[name] for name in selected}
    missing = [str(source) for source in sources.values() if not source.is_file()]
    if missing:
        print("error: missing source files:", file=sys.stderr)
        for source in missing:
            print(f"  {source}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True)
    results = []
    print(f"Affine tool: {tool}")
    print(f"Programs:    {len(selected)}")
    print(f"Results:     {output_dir}")
    print()

    for name, source in sources.items():
        print(f"[RUN ] {name}", flush=True)
        result = run_one(
            name,
            source,
            tool,
            output_dir,
            args.timeout,
            args.order_direction,
            args.threshold,
        )
        results.append(result)
        warning_suffix = f", warnings={len(result['warnings'])}" if result["warnings"] else ""
        print(
            f"[{result['status']:<4}] {name}: vars={result.get('variable_count', 0)}, "
            f"elapsed={result['elapsed_seconds']:.2f}s{warning_suffix}",
            flush=True,
        )
        for error in result["errors"]:
            print(f"       ERROR: {error}")
        for warning in result["warnings"]:
            print(f"       WARN:  {warning}")

    passed = sum(result["status"] == "PASS" for result in results)
    warning_count = sum(len(result["warnings"]) for result in results)
    summary = {
        "tool": str(tool),
        "base_dir": str(base_dir),
        "order_direction": args.order_direction,
        "advisory_threshold": args.threshold,
        "program_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "warning_count": warning_count,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Summary: {passed}/{len(results)} passed, "
        f"{len(results) - passed} failed, {warning_count} warnings"
    )
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    if passed != len(results):
        return 1
    if args.strict_warnings and warning_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
