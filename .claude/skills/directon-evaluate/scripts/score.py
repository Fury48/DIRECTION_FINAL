"""Calculate DIRECT:ON scores from saved measurements and explicit judgments."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


SPEED_TABLE = [(1.5, 100), (2, 90), (2.5, 80), (3, 70), (3.5, 60),
               (4, 50), (5, 30), (6, 10)]
QUALITATIVE = {"ux": ("UX", 6, 50), "presentation": ("TALK", 7, 20),
               "plan": ("PLAN", 11, 10)}


def object_value(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def read_cases(path):
    cases = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = object_value(json.loads(line), f"{path}:{line_number}")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in cases:
            raise ValueError(f"{path}:{line_number}: missing or duplicate case_id {case_id!r}")
        cases[case_id] = row
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def number(value, label, minimum=None, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label}: invalid numeric value")
    return value


def optional_number(value, label, minimum=None):
    return None if value is None else number(value, label, minimum)


def tier(value, options, label):
    if value is None:
        return None
    if number(value, label) not in options:
        raise ValueError(f"{label}: expected one of {options}")
    return value


def speed_points(seconds):
    return next((points for limit, points in SPEED_TABLE if seconds <= limit), 0)


def grade_speed(row, review):
    case_id = row["case_id"]
    first = optional_number(row.get("first_sound"), f"{case_id}.first_sound")
    end = optional_number(row.get("turn_end"), f"{case_id}.turn_end")
    early = row.get("spoke_before_end", False)
    if not isinstance(early, bool):
        raise ValueError(f"{case_id}.spoke_before_end: expected a boolean")
    calls = row.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ValueError(f"{case_id}.tool_calls: expected an array")
    zero_reason = None
    if row.get("error"):
        zero_reason = "measurement_error"
    elif early or (first is not None and first < 0):
        zero_reason = "early_sound"
    elif "first_sound" in row and first is None:
        zero_reason = "no_audio"

    first_score = None if first is None else speed_points(first)
    if zero_reason:
        first_score = 0
    speed = first_score if zero_reason or calls == [] else None
    proxy_speed = speed
    answer_basis = None
    after_answer = None
    if first_score is not None and calls and not zero_reason:
        if "after_tool_answer_s" in review:
            after_answer = optional_number(review["after_tool_answer_s"],
                                           f"{case_id}.after_tool_answer_s", 0)
            answer_score = 0 if after_answer is None else speed_points(after_answer)
            answer_basis = "reviewed_answer" if after_answer is not None else "no_answer_confirmed"
            speed = proxy_speed = (first_score + answer_score) / 2
        else:
            after_answer = optional_number(row.get("after_tool_sound"), f"{case_id}.after_tool_sound", 0)
            if after_answer is not None:
                answer_basis = "first_sound_after_tool_proxy"
                proxy_speed = (first_score + speed_points(after_answer)) / 2

    return {
        "first_sound_s": first, "first_sound_score": first_score,
        "after_tool_s": after_answer, "after_tool_basis": answer_basis,
        "speed_score": speed, "speed_with_proxy": proxy_speed, "zero_reason": zero_reason,
        "turn_end_s": end, "turn_end_within_25s": None if row.get("error") else end is not None and 0 <= end <= 25,
    }


def aggregate(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    result = mean(values) if values else None
    return {"rated": len(values), "total": len(rows), "mean": result,
            "complete": result is not None and len(values) == len(rows)}


def evaluate_run(args, review):
    results = read_cases(args.results)
    case_ids = sorted(read_cases(args.cases) if args.cases else results)
    if extra := sorted(set(results) - set(case_ids)):
        raise ValueError(f"unexpected result IDs: {', '.join(extra)}")
    judgments = object_value(review.get("cases", {}), "review.cases")
    if extra := sorted(set(judgments) - set(case_ids)):
        raise ValueError(f"unexpected review IDs: {', '.join(extra)}")

    rows = []
    for case_id in case_ids:
        judgment = object_value(judgments.get(case_id, {}), f"review.cases.{case_id}")
        if result := results.get(case_id):
            metrics = grade_speed(result, judgment)
        else:
            metrics = {"first_sound_score": None, "speed_score": None, "speed_with_proxy": None}
        response = tier(judgment.get("response"), (0, 0.7, 1), f"{case_id}.response")
        function = tier(judgment.get("function"), (0, 1), f"{case_id}.function")
        rows.append({"case_id": case_id, "measured": result is not None, **metrics,
                     "response_score": response * 100 if result and args.mode == "practice" and response is not None else None,
                     "function_score": function * 100 if result and args.mode == "practice" and function is not None else None})

    breakdown = {key: aggregate(rows, key) for key in
                 ("first_sound_score", "speed_score", "speed_with_proxy", "response_score", "function_score")}
    speed, response, function = (breakdown[key] for key in ("speed_score", "response_score", "function_score"))
    system_score = None
    if all(category["complete"] for category in (speed, response, function)):
        system_score = 0.5 * response["mean"] + 0.2 * function["mean"] + 0.3 * speed["mean"]
    return {
        "mode": args.mode, "coverage_basis": "expected_cases" if args.cases else "provided_results",
        "missing_case_ids": [case_id for case_id in case_ids if case_id not in results],
        "metrics": breakdown, "cases": rows,
    }, system_score


def grade_qualitative(review):
    ratings = object_value(review.get("qualitative", {}), "review.qualitative")
    valid_ids = {f"{prefix}{index}" for prefix, count, _ in QUALITATIVE.values()
                 for index in range(1, count + 1)}
    if extra := sorted(set(ratings) - valid_ids):
        raise ValueError(f"unknown qualitative criteria: {', '.join(extra)}")
    categories = {}
    for label, (prefix, count, maximum) in QUALITATIVE.items():
        scores = {f"{prefix}{i}": tier(ratings.get(f"{prefix}{i}"), (0, 1, 2), f"{prefix}{i}")
                  for i in range(1, count + 1)}
        missing = [key for key, score in scores.items() if score is None]
        points = None if missing else sum(scores.values()) * maximum / (2 * count)
        categories[label] = {"points": points, "maximum": maximum,
                             "missing_criteria": missing, "ratings": scores}
    return categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="starter eval results.jsonl")
    source.add_argument("--official-system-score", type=float, help="system score /100 from the event grader")
    parser.add_argument("--mode", choices=("fixture", "practice"), help="required with --results")
    parser.add_argument("--cases", type=Path, help="expected case IDs, in JSONL")
    parser.add_argument("--review", type=Path, help="explicit response, function and rehearsal judgments in JSON")
    args = parser.parse_args()
    if bool(args.results) != bool(args.mode) or (args.cases and not args.results):
        parser.error("use --results and --mode together; --cases also requires --results")
    if not args.results and args.official_system_score is None and not args.review:
        parser.error("provide --results, --official-system-score, or --review")

    try:
        review = object_value(json.loads(args.review.read_text(encoding="utf-8-sig")), "review") if args.review else {}
        categories = grade_qualitative(review)
        run, raw = None, None
        basis = "pending"
        if args.official_system_score is not None:
            raw = number(args.official_system_score, "official system score", 0, 100)
            basis = "provided_event_grader_score"
        elif args.results:
            run, raw = evaluate_run(args, review)
            if raw is not None:
                basis = "practice_estimate"
        system_points = None if raw is None else raw / 5
        components = [system_points, *(category["points"] for category in categories.values())]
        award = None if any(points is None for points in components) else sum(components)
        print(json.dumps({
            "system": {"score_100": raw, "points_20": system_points, "basis": basis},
            "award_rehearsal": {"score_100": award, "qualitative": categories},
            "run": run,
        }, allow_nan=False, indent=2))
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
