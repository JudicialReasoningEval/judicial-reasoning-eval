"""Weight-sensitivity analysis for the declared composite weights.

The equal weights in Q = w1*M1 + w2*M2 + w3*M3 and the 50/50 split inside
M2 (struct_sim vs presence-F1) are declared, not derived — does the model
RANKING depend on them?

Part A — Q simplex sweep: every (w1, w2, w3) on the unit simplex at step
1/GRID_N. For each weight vector the models are re-ranked by Q(w); we
report how often the top-1 / top-3 change, the distinct rankings seen,
and Kendall tau of every ranking against the equal-weight one.

Part B — M2 alpha sweep: M2(a) = a*struct_sim + (1-a)*presence_f1 for
a in [0, 1]; we track the M2 ranking and the induced equal-weight Q
ranking. (M2 per case is the plain mean of the two components, so the
model-level sweep on component means is exact.)

Output: results/_sensitivity.json + printed summary.

Usage:
    python -m metrics.sensitivity
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

GRID_N = 20          # simplex step 0.05 -> 231 weight vectors
ALPHA_STEPS = 101    # M2 alpha in 0.00 .. 1.00


def _load_models() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        mj = d / "metrics.json"
        if not d.is_dir() or not mj.exists():
            continue
        m = json.loads(mj.read_text())
        m1, m2, m3 = m.get("M1_normative"), m.get("M2_structural"), m.get("M3_semantic")
        if not (m1 and m2 and m3):
            continue
        out[d.name] = {
            "m1": m1["weighted_f1"],
            "m2": m2["m2"]["mean"],
            "m2_struct": m2["struct_sim_mean"],
            "m2_presence": m2["presence_f1_mean"],
            "m3": m3["m3"]["mean"],
        }
    return out


def _rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda k: -scores[k])


def _kendall_tau(a: list[str], b: list[str]) -> float:
    pos_a = {m: i for i, m in enumerate(a)}
    pos_b = {m: i for i, m in enumerate(b)}
    pairs = list(combinations(a, 2))
    disc = sum(1 for x, y in pairs
               if (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) < 0)
    return 1 - 2 * disc / len(pairs)


def main() -> None:
    models = _load_models()
    names = list(models)
    print(f"[sensitivity] {len(names)} models with complete M1/M2/M3")

    def q(v: dict[str, float], w1: float, w2: float, w3: float) -> float:
        return w1 * v["m1"] + w2 * v["m2"] + w3 * v["m3"]

    base_rank = _rank({n: q(v, 1 / 3, 1 / 3, 1 / 3) for n, v in models.items()})

    # Part A — full simplex sweep
    top1: dict[str, int] = {}
    top3: dict[tuple[str, ...], int] = {}
    rankings: dict[tuple[str, ...], int] = {}
    taus: list[float] = []
    n_pts = 0
    for i in range(GRID_N + 1):
        for j in range(GRID_N + 1 - i):
            k = GRID_N - i - j
            w1, w2, w3 = i / GRID_N, j / GRID_N, k / GRID_N
            r = _rank({n: q(v, w1, w2, w3) for n, v in models.items()})
            n_pts += 1
            top1[r[0]] = top1.get(r[0], 0) + 1
            t3 = tuple(sorted(r[:3]))
            top3[t3] = top3.get(t3, 0) + 1
            rankings[tuple(r)] = rankings.get(tuple(r), 0) + 1
            taus.append(_kendall_tau(r, base_rank))

    # Part B — M2 alpha sweep (ranking by M2 itself, and by equal-weight Q)
    m2_rankings: dict[tuple[str, ...], int] = {}
    q_top1_alpha: dict[str, int] = {}
    for s in range(ALPHA_STEPS):
        a = s / (ALPHA_STEPS - 1)
        m2a = {n: a * v["m2_struct"] + (1 - a) * v["m2_presence"] for n, v in models.items()}
        m2_rankings[tuple(_rank(m2a))] = m2_rankings.get(tuple(_rank(m2a)), 0) + 1
        qa = {n: (v["m1"] + m2a[n] + v["m3"]) / 3 for n, v in models.items()}
        r = _rank(qa)
        q_top1_alpha[r[0]] = q_top1_alpha.get(r[0], 0) + 1

    payload = {
        "models": names,
        "equal_weight_ranking": base_rank,
        "q_simplex": {
            "grid_step": 1 / GRID_N,
            "n_points": n_pts,
            "top1_share": {m: c / n_pts for m, c in sorted(top1.items(), key=lambda x: -x[1])},
            "top3_sets": {" | ".join(k): c / n_pts
                          for k, c in sorted(top3.items(), key=lambda x: -x[1])},
            "n_distinct_rankings": len(rankings),
            "kendall_tau_vs_equal": {
                "mean": sum(taus) / len(taus),
                "min": min(taus),
            },
        },
        "m2_alpha": {
            "n_steps": ALPHA_STEPS,
            "n_distinct_m2_rankings": len(m2_rankings),
            "q_top1_share": {m: c / ALPHA_STEPS
                             for m, c in sorted(q_top1_alpha.items(), key=lambda x: -x[1])},
        },
    }
    out = RESULTS_DIR / "_sensitivity.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[sensitivity] Q simplex ({n_pts} weight vectors, step {1/GRID_N}):")
    for m, share in payload["q_simplex"]["top1_share"].items():
        print(f"  top-1 {m}: {share:.1%}")
    print(f"  distinct full rankings: {len(rankings)}")
    kt = payload["q_simplex"]["kendall_tau_vs_equal"]
    print(f"  Kendall tau vs equal weights: mean {kt['mean']:.3f}, min {kt['min']:.3f}")
    print(f"[sensitivity] M2 alpha sweep: {len(m2_rankings)} distinct M2 rankings; "
          f"Q top-1 share: {payload['m2_alpha']['q_top1_share']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
