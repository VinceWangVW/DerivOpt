"""Configuration-level and paired-seed metric summaries."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


CONFIGURATION_KEYS = ("family", "backbone", "budget_ratio", "retain_frac")


def _fingerprint(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _protocol_check(records, configuration_keys, metric):
    """Reject incompatible known provenance and expose absent legacy metadata.

    Source fingerprints plus evaluated trajectory IDs, or an explicit
    test_identity, establish a matched test set independently of random seeds.
    Rows without test-set metadata retain an explicit unknown status.
    """
    protocol = {}
    for name in ("library", "split_seed", "metric_protocol", "horizon_protocol", "horizon_units"):
        values = {_fingerprint(row.get(name)) for row in records}
        if len(values) != 1:
            raise ValueError(f"Cannot mix different or known/unknown {name} values in one report")
        protocol[name] = records[0].get(name)
    if metric in ("detail_horizon", "detail_horizon_normalized", "detail_horizon_steps"):
        if protocol["horizon_protocol"] is None or protocol["horizon_units"] is None:
            raise ValueError("Horizon reports require explicit horizon_protocol and horizon_units; "
                             "re-evaluate saved predictions when metric provenance is unknown")
    protocol["metric_protocol_known"] = all(protocol[name] is not None
                                            for name in ("metric_protocol", "horizon_protocol", "horizon_units"))
    runs = defaultdict(list)
    for row in records:
        key = (row["method"], tuple(row[name] for name in configuration_keys), row["seed"])
        runs[key].append(row)
    by_family = defaultdict(list)
    for rows in runs.values():
        fixed = {}
        for name in ("dataset_identity", "test_identity", "test_trajectory_ids"):
            values = {_fingerprint(row.get(name)) for row in rows}
            if len(values) != 1:
                raise ValueError(f"Run rows disagree about {name}; test identities cannot be pooled")
            fixed[name] = rows[0].get(name)
        present = [row.get("trajectory_id") is not None for row in rows]
        if any(present) and not all(present):
            raise ValueError("Run contains known and unknown evaluated trajectory identities")
        # Preserve multiplicity, since duplicate sample rows change the weights.
        evaluated = sorted(Counter(_fingerprint(row["trajectory_id"]) for row in rows).items()) if all(present) else None
        window_values = [_fingerprint({
            "trajectory_id": row.get("trajectory_id"),
            "rollout_steps": row.get("rollout_steps"),
            "time_window": row.get("evaluation_time_window"),
        }) for row in rows]
        # Without sample IDs, legacy aggregates cannot establish matched sample
        # multiplicities. Still reject explicitly different horizon definitions.
        windows = sorted(Counter(window_values).items()) if all(present) else sorted(set(window_values))
        declared = fixed["test_trajectory_ids"]
        if declared is not None:
            if not isinstance(declared, (list, tuple)) or not declared:
                raise ValueError("test_trajectory_ids must be a nonempty sequence")
            declared = sorted(_fingerprint(item) for item in declared)
        identity = (_fingerprint(fixed["test_identity"]), _fingerprint(declared),
                    _fingerprint(evaluated), _fingerprint(windows))
        source = _fingerprint(fixed["dataset_identity"])
        known_test = fixed["test_identity"] is not None or declared is not None or evaluated is not None
        verified = (fixed["test_identity"] is not None
                    or (fixed["dataset_identity"] is not None and known_test))
        status = "verified" if verified else ("trajectory_ids_only" if known_test else "unknown")
        by_family[str(rows[0].get("family", "unspecified"))].append((source, identity, status))
    statuses = {}
    for family, observations in by_family.items():
        if len({(source, identity) for source, identity, _ in observations}) != 1:
            raise ValueError(f"Cannot mix different or known/unknown test identities for family {family}")
        statuses[family] = observations[0][2]
    protocol["test_identity_status_by_family"] = statuses
    protocol["test_identity_verified"] = all(status == "verified" for status in statuses.values())
    return protocol


def _run_means(records: Iterable[Mapping[str, Any]], metric: str,
               configuration_keys: Sequence[str]) -> dict[tuple, float]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        required = (*configuration_keys, "method", "seed", metric)
        if any(key not in row for key in required):
            raise ValueError(f"Report row needs {required}")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError("Nonfinite raw metrics must be reported explicitly; no silent dropping or imputation")
        key = (row["method"], tuple(row[name] for name in configuration_keys), row["seed"])
        groups[key].append(value)
    if not groups:
        raise ValueError("Cannot summarize an empty set of records")
    return {key: statistics.mean(values) for key, values in groups.items()}


def configuration_macro(records: Iterable[Mapping[str, Any]], metric: str, *,
                        configuration_keys: Sequence[str] = CONFIGURATION_KEYS) -> dict[str, dict[str, Any]]:
    """Average samples within run, seeds within configuration, then configurations.

    Every observed configuration receives equal weight, regardless of trajectory
    count or seed count. The observed cell list and seed counts are returned for
    coverage checking; missing cells are excluded from the aggregate.
    """
    records = list(records)
    runs = _run_means(records, metric, configuration_keys)
    protocol = _protocol_check(records, configuration_keys, metric)
    cells: dict[tuple, list[float]] = defaultdict(list)
    for (method, configuration, seed), value in runs.items():
        cells[(method, configuration)].append(value)
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (method, configuration), values in cells.items():
        methods[method].append({"configuration": dict(zip(configuration_keys, configuration)),
                                "mean": statistics.mean(values), "seed_count": len(values)})
    return {method: {"mean": statistics.mean(cell["mean"] for cell in values),
                     "configuration_count": len(values), "configurations": values,
                     "protocol": protocol}
            for method, values in methods.items()}


def paired_seed_summary(records: Iterable[Mapping[str, Any]], metric: str, method_a: str,
                        method_b: str, *, configuration_keys: Sequence[str] = CONFIGURATION_KEYS) -> dict[str, Any]:
    """Macro-average each paired seed first, then report mean and sample SD (n-1).

    Both methods must have identical configuration/seed coverage; every cell must
    use the same seed set. Unpaired records fail explicitly instead of being dropped.
    A single seed has undefined sample SD, represented by None, not zero.
    Delta is A-B; direction of improvement depends on the chosen metric.
    """
    if method_a == method_b:
        raise ValueError("A paired comparison requires two different methods")
    selected = [row for row in records if row.get("method") in (method_a, method_b)]
    runs = _run_means(selected, metric, configuration_keys)
    protocol = _protocol_check(selected, configuration_keys, metric)
    first = {(configuration, seed): value for (method, configuration, seed), value in runs.items() if method == method_a}
    second = {(configuration, seed): value for (method, configuration, seed), value in runs.items() if method == method_b}
    if not first or first.keys() != second.keys():
        raise ValueError("Paired methods must have identical nonempty configuration/seed coverage")
    config_seeds: dict[tuple, set] = defaultdict(set)
    for configuration, seed in first:
        config_seeds[configuration].add(seed)
    seed_sets = list(config_seeds.values())
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Every configuration must use the same paired seed set")
    rows = []
    for seed in sorted(seed_sets[0], key=str):
        a = statistics.mean(first[(configuration, seed)] for configuration in config_seeds)
        b = statistics.mean(second[(configuration, seed)] for configuration in config_seeds)
        rows.append({"seed": seed, "a": a, "b": b, "delta": a - b})
    result: dict[str, Any] = {"method_a": method_a, "method_b": method_b, "metric": metric,
                              "seed_count": len(rows), "configuration_count": len(config_seeds),
                              "per_seed": rows, "delta_definition": "A-B", "protocol": protocol}
    for name in ("a", "b", "delta"):
        values = [row[name] for row in rows]
        result[f"mean_{name}"] = statistics.mean(values)
        result[f"sample_sd_{name}"] = statistics.stdev(values) if len(values) > 1 else None
    return result
