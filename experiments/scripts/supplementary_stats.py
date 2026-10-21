"""
Supplementary statistics for the CEC and engineering comparisons: seed-paired Wilcoxon signed-rank
tests, rank-sum tests with a tie tolerance, Holm adjustment over a whole suite, the Friedman test on
median errors, and the comparison of the variant without the rejected-trial chain with the
competitors. Only the Python standard library is used.

Inputs
  results_summary/<set>/run_log.txt      final value f of every run (7 significant digits), one line per run
  results_summary/<set>/<suite>/summary.csv   per function and algorithm: mean, median, ... (full precision)
  outputs/tables/*.csv                  F* of the CEC functions (A1, A2), unpaired p-values (B2, C2),
                                        tests over the functions (D1, D3) used to validate this script
Outputs (outputs/tables_supplementary/)
  R0_validation.csv          this script against the scipy values stored in the generated tables
  R1_paired_competitors.csv  seed-paired Wilcoxon signed-rank tests, RTI-DE against each competitor
  R1u_unpaired_tolerance.csv rank-sum tests on the run-log values (tie tolerance), RTI-DE against each competitor
  R2_paired_variants.csv     seed-paired tests, RTI-DE against each ablation / sensitivity variant
  R3_holm_global.csv         the unpaired rank-sum p-values of B2/C2 with Holm adjustment over all
                             functions and competitors of a suite
  R4_friedman_medians.csv    Friedman test and Nemenyi critical difference on the medians
  R5_nochain_vs_competitors.csv   the variant without the chain against the six competitors
  R6_summary.md              human-readable summary of all of the above
  R7_robustness_competitors.tex  LaTeX table: alternative test procedures, RTI-DE against the competitors
  R8_paired_variants.tex         LaTeX table: seed-paired tests, RTI-DE against the variants

Conventions (as in scripts/make_tables.py): error = f - F*, errors below 1e-8 count as 0; alpha = 0.05;
"+" = RTI-DE (or the row's algorithm) significantly better (lower error), "-" = significantly worse.
Because the run log stores f with 7 significant digits, two final values that agree to 7 digits are
equal here; this removes floating-point ties but also differences below that resolution.

Tests
  signed-rank: two-sided, zero differences dropped (Wilcoxon's method); exact null distribution of the
               remaining differences when there are no ties and n <= 50 (as in Table D3), otherwise the
               normal approximation with tie correction and without continuity correction.
  rank-sum:    Mann-Whitney U, two-sided, normal approximation with tie and continuity correction
               (scipy.stats.mannwhitneyu with method "asymptotic"; the generated tables use n = 30).
  Friedman:    chi-square approximation with average ranks for ties; Nemenyi CD = q * sqrt(k(k+1)/(6N)),
               q = 2.9483200175296744 for k = 7 (the value recorded in fig_cd_*.json).
"""

import csv
import math
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RS = os.path.join(ROOT, "results_summary")
TAB = os.path.join(ROOT, "outputs", "tables")
OUT = os.path.join(ROOT, "outputs", "tables_supplementary")
COMPETITORS = ["DE", "LSHADE", "RIME", "PSO", "GWO", "WOA"]
ALGOS = ["RTI-DE"] + COMPETITORS
ABLATION = ["RTI-DE-noChain", "RTI-DE-halfStep", "RTI-DE-chainAlways", "RTI-DE-noLPSR", "RTI-DE-fixedF",
            "RTI-DE-noCR"]
SENSITIVITY = ["RTI-DE-cf50", "RTI-DE-cf90", "RTI-DE-p05", "RTI-DE-p20", "RTI-DE-n10D", "RTI-DE-n25D",
               "RTI-DE-cr70", "RTI-DE-cr95"]
SUITES = ["cec2022", "cec2017"]
ENGINEERING = ["composite", "composite_contiguity", "uav"]
ALPHA = 0.05
Q_NEMENYI = 2.9483200175296744
# Names as printed in the LaTeX tables; the keys above and in the result files stay unchanged
# (same mapping as figlib.DISPLAY_NAME, repeated here because this script uses the standard library only).
DISPLAY_NAME = {"LSHADE": "L-SHADE"}


def display(algo):
    return DISPLAY_NAME.get(algo, algo)


# ----------------------------------------------------------------------------- statistics helpers
def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def avg_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test of the pairs (x_i, y_i).
    Returns (p, w_plus, w_minus, n_used); w_plus = rank sum of the pairs with x > y."""
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return float("nan"), 0.0, 0.0, 0
    r = avg_ranks([abs(v) for v in d])
    w_plus = sum(rk for rk, v in zip(r, d) if v > 0)
    w_minus = sum(rk for rk, v in zip(r, d) if v < 0)
    ties = len(set(abs(v) for v in d)) < n
    if not ties and n <= 50:
        # exact null distribution of W+ (integer ranks 1..n)
        total = n * (n + 1) // 2
        counts = [0] * (total + 1)
        counts[0] = 1
        for k in range(1, n + 1):
            for s in range(total, k - 1, -1):
                counts[s] += counts[s - k]
        w = int(round(min(w_plus, w_minus)))
        p = 2.0 * sum(counts[: w + 1]) / 2.0 ** n
        return min(1.0, p), w_plus, w_minus, n
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    groups = defaultdict(int)
    for v in r:
        groups[v] += 1
    var -= sum(t ** 3 - t for t in groups.values()) / 48.0
    if var <= 0:
        return float("nan"), w_plus, w_minus, n
    z = (min(w_plus, w_minus) - mean) / math.sqrt(var)
    return min(1.0, 2.0 * norm_sf(abs(z))), w_plus, w_minus, n


def mann_whitney(x, y):
    """Two-sided Mann-Whitney U test, normal approximation with tie and continuity correction."""
    n1, n2 = len(x), len(y)
    r = avg_ranks(list(x) + list(y))
    u1 = sum(r[:n1]) - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    groups = defaultdict(int)
    for v in r:
        groups[v] += 1
    tie = sum(t ** 3 - t for t in groups.values())
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return float("nan")
    u = max(u1, n1 * n2 - u1)
    z = (u - mu - 0.5) / math.sqrt(var)
    return min(1.0, 2.0 * norm_sf(z))


def holm(pvals):
    """Holm-adjusted p-values; nan stays nan."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    idx.sort(key=lambda i: pvals[i])
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for k, i in enumerate(idx):
        running = max(running, (m - k) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def holm_smaller(rows, m):
    """'F1: 3, F3: 5' for the functions of rows whose Holm family has fewer than m comparisons with a
    defined test (finite p, see holm); '' if there are none. rows: dicts with 'function' and float 'p'."""
    defined = {}
    for r in rows:
        defined[r["function"]] = defined.get(r["function"], 0) + (not math.isnan(r["p"]))
    return ", ".join(f"{f}: {k or 'none'}" for f, k in defined.items() if k < m)


def holm_smaller_text(analyses, m):
    """analyses: [(label, rows)]. Analyses with the same smaller families are listed together:
    'unpaired (F1: 3, F3: 5); tie tolerance and seed-paired (F1: 3, F3: 4)'."""
    groups = {}
    for label, rows in analyses:
        t = holm_smaller(rows, m)
        if t:
            groups.setdefault(t, []).append(label)
    return "; ".join(f"{join_and(labels)} ({t})" for t, labels in groups.items())


def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def chi2_sf_even(x, df):
    """Survival function of the chi-square distribution for an even number of degrees of freedom."""
    h = x / 2.0
    term, s = 1.0, 1.0
    for i in range(1, df // 2):
        term *= h / i
        s += term
    return math.exp(-h) * s


def friedman(table):
    """table: list of rows (one per function), each a list of k values (lower is better)."""
    n, k = len(table), len(table[0])
    sums = [0.0] * k
    tie_term = 0.0
    for row in table:
        r = avg_ranks(row)
        for j in range(k):
            sums[j] += r[j]
        groups = defaultdict(int)
        for v in r:
            groups[v] += 1
        tie_term += sum(t ** 3 - t for t in groups.values())
    mean_ranks = [s / n for s in sums]
    chi2 = 12.0 / (n * k * (k + 1)) * sum(s * s for s in sums) - 3.0 * n * (k + 1)
    chi2 /= 1.0 - tie_term / (n * k * (k * k - 1))
    p = chi2_sf_even(chi2, k - 1) if (k - 1) % 2 == 0 else float("nan")
    cd = Q_NEMENYI * math.sqrt(k * (k + 1) / (6.0 * n))
    return mean_ranks, chi2, p, cd


def verdict(p, w_better, w_worse):
    """'+' if significant and the row algorithm is better, '-' if significant and worse, else '≈'."""
    if math.isnan(p) or p >= ALPHA:
        return "≈"
    return "+" if w_better > w_worse else "-"


# ----------------------------------------------------------------------------- data
LINE = re.compile(r"^\S+ (\S+) (\S+) (\S+) run(\d+) ok f=(\S+) ")


def read_log(path):
    """{(suite, fid, algo): {run: f}}; engineering problems use fid 'F1'."""
    data = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = LINE.match(line)
            if not m:
                continue
            suite, fid, algo, run, f = m.groups()
            data[(suite, fid, algo)][int(run)] = float(f)
    return data


def f_star():
    fs = {}
    for suite, name in (("cec2022", "A1_cec2022_functions.csv"), ("cec2017", "A2_cec2017_functions.csv")):
        with open(os.path.join(TAB, name), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fs[(suite, row["No"])] = float(row["F*"])
    return fs


def errors(data, fs, suite, fid, algo):
    runs = data[(suite, fid, algo)]
    if suite in fs and (suite, fid) in fs:
        e = {r: v - fs[(suite, fid)] for r, v in runs.items()}
        return {r: (0.0 if v < 1e-8 else v) for r, v in e.items()}
    return dict(runs)                                   # engineering problems: minimised objective


def functions(data, suite):
    return sorted({k[1] for k in data if k[0] == suite}, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(name, rows, fields):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def wtl(rows, key):
    c = {"+": 0, "≈": 0, "-": 0}
    for r in rows:
        c[r[key]] += 1
    return f"{c['+']}/{c['≈']}/{c['-']}"


# ----------------------------------------------------------------------------- analyses
def paired_against(data, fs, suite, ref, others):
    """Seed-paired signed-rank tests of ref against each of others, per function, Holm within function."""
    rows = []
    for fid in functions(data, suite):
        block = []
        for o in others:
            if (suite, fid, o) not in data:
                continue
            a, b = errors(data, fs, suite, fid, ref), errors(data, fs, suite, fid, o)
            runs = sorted(set(a) & set(b))
            p, w_ref_worse, w_ref_better, n = signed_rank([a[r] for r in runs], [b[r] for r in runs])
            block.append({"suite": suite, "function": fid, "reference": ref, "other": o, "pairs": len(runs),
                          "non_tied_pairs": n, "W_ref_lower": w_ref_better, "W_ref_higher": w_ref_worse,
                          "p": p, "verdict": verdict(p, w_ref_better, w_ref_worse)})
        adj = holm([r["p"] for r in block])
        for r, pa in zip(block, adj):
            r["p_holm"] = pa
            r["verdict_holm"] = verdict(pa, r["W_ref_lower"], r["W_ref_higher"])
        rows += block
    return rows


def main():
    fs = f_star()
    main_log = read_log(os.path.join(RS, "paper", "run_log.txt"))
    abl_log = read_log(os.path.join(RS, "paper_ablation", "run_log.txt"))
    sen_log = read_log(os.path.join(RS, "paper_sensitivity", "run_log.txt"))
    for k, v in abl_log.items():
        main_log.setdefault(k, v)
    for k, v in sen_log.items():
        main_log.setdefault(k, v)
    data = main_log
    lines = ["# Supplementary statistics (generated by scripts/supplementary_stats.py)", ""]

    # --- R0: validation against the scipy values stored in the generated tables
    val = []
    for suite, ckey in (("cec2022", "B2_cec2022_ranksum.csv"), ("cec2017", "C2_cec2017_ranksum.csv")):
        for row in read_csv(os.path.join(TAB, ckey)):
            if not row["p"]:
                continue
            a = list(errors(data, fs, suite, row["function"], "RTI-DE").values())
            b = list(errors(data, fs, suite, row["function"], row["competitor"]).values())
            p = mann_whitney(a, b)
            val.append({"check": "rank-sum from run log vs table", "suite": suite, "item": f"{row['function']} {row['competitor']}",
                        "table": float(row["p"]), "this_script": p,
                        "same_decision": (float(row["p"]) < ALPHA) == (p < ALPHA)})
    d3 = read_csv(os.path.join(TAB, "D3_signed_rank.csv"))
    d1 = read_csv(os.path.join(TAB, "D1_friedman.csv"))
    means = {}
    for suite, sub in (("cec2022", "cec2022_D20"), ("cec2017", "cec2017_D100")):
        for row in read_csv(os.path.join(RS, "paper", sub, "summary.csv")):
            means[(suite, row["function"], row["algorithm"])] = (float(row["mean"]), float(row["median"]))
    for row in d3:
        suite = row["suite"]
        comp = row["competitor"]
        fids = functions(data, suite)
        x = [means[(suite, f, comp)][0] for f in fids]
        y = [means[(suite, f, "RTI-DE")][0] for f in fids]
        p, _, _, _ = signed_rank(x, y)
        val.append({"check": "signed-rank over means vs table D3", "suite": suite, "item": comp,
                    "table": float(row["p"]), "this_script": p,
                    "same_decision": (float(row["p"]) < ALPHA) == (p < ALPHA)})
    for row in d1:
        suite = row["suite"]
        fids = functions(data, suite)
        tab = [[means[(suite, f, a)][0] for a in ALGOS] for f in fids]
        _, chi2, p, cd = friedman(tab)
        val.append({"check": "Friedman chi2 on means vs table D1", "suite": suite, "item": "chi2",
                    "table": float(row["friedman_chi2"]), "this_script": chi2, "same_decision": True})
        val.append({"check": "Nemenyi CD vs table D1", "suite": suite, "item": "CD",
                    "table": float(row["nemenyi_cd"]), "this_script": cd, "same_decision": True})
    write_csv("R0_validation.csv", val, ["check", "suite", "item", "table", "this_script", "same_decision"])
    rel = [abs(v["this_script"] - v["table"]) / max(abs(v["table"]), 1e-300) for v in val if v["check"] != "rank-sum from run log vs table"]
    mism = [v for v in val if not v["same_decision"]]
    lines += ["## R0 Validation", "",
              f"- Checks: {len(val)}; decisions that differ from the generated tables: {len(mism)}.",
              f"- Signed-rank, Friedman and CD from full-precision means: largest relative difference to the "
              f"tables {max(rel):.2e}.", ""]
    for v in mism:
        lines.append(f"  - differs: {v['check']} {v['suite']} {v['item']}: table p = {v['table']:.3g}, "
                     f"here {v['this_script']:.3g}")
    lines.append("")

    # --- R1: paired tests against the competitors
    r1 = []
    for suite in SUITES + ENGINEERING:
        r1 += paired_against(data, fs, suite, "RTI-DE", COMPETITORS)
    write_csv("R1_paired_competitors.csv", r1, ["suite", "function", "reference", "other", "pairs", "non_tied_pairs",
                                                "W_ref_lower", "W_ref_higher", "p", "verdict", "p_holm", "verdict_holm"])
    lines += ["## R1 Seed-paired signed-rank tests, RTI-DE against each competitor", "",
              "| Suite | " + " | ".join(COMPETITORS) + " |", "|---|" + "---|" * len(COMPETITORS)]
    for suite in SUITES + ENGINEERING:
        cells = []
        for c in COMPETITORS:
            rows = [r for r in r1 if r["suite"] == suite and r["other"] == c]
            cells.append(f"{wtl(rows, 'verdict')} (Holm {wtl(rows, 'verdict_holm')})")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- R1u: unpaired rank-sum tests on the run-log values (7 significant digits, i.e. with a tie
    #          tolerance), to separate the effect of the tolerance from that of the pairing
    r1u = []
    for suite in SUITES + ENGINEERING:
        for fid in functions(data, suite):
            a = list(errors(data, fs, suite, fid, "RTI-DE").values())
            block = []
            for c in COMPETITORS:
                b = list(errors(data, fs, suite, fid, c).values())
                p = mann_whitney(a, b)
                ra = avg_ranks(a + b)
                lower = sum(ra[:len(a)]) < sum(ra[len(a):])
                block.append({"suite": suite, "function": fid, "competitor": c, "p": p, "rti_lower_ranks": lower})
            adj = holm([r["p"] for r in block])
            for r, pa in zip(block, adj):
                r["verdict"] = "≈" if math.isnan(r["p"]) or r["p"] >= ALPHA else ("+" if r["rti_lower_ranks"] else "-")
                r["p_holm"] = pa
                r["verdict_holm"] = "≈" if math.isnan(pa) or pa >= ALPHA else ("+" if r["rti_lower_ranks"] else "-")
            r1u += block
    write_csv("R1u_unpaired_tolerance.csv", r1u, ["suite", "function", "competitor", "p", "verdict", "p_holm",
                                                   "verdict_holm", "rti_lower_ranks"])
    lines += ["## R1u Unpaired rank-sum tests on the run-log values (tie tolerance of 7 significant digits)", "",
              "| Suite | " + " | ".join(COMPETITORS) + " |", "|---|" + "---|" * len(COMPETITORS)]
    for suite in SUITES + ENGINEERING:
        cells = []
        for c in COMPETITORS:
            rows = [r for r in r1u if r["suite"] == suite and r["competitor"] == c]
            cells.append(f"{wtl(rows, 'verdict')} (Holm {wtl(rows, 'verdict_holm')})")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    changed = []
    for suite, ckey in (("cec2022", "B2_cec2022_ranksum.csv"), ("cec2017", "C2_cec2017_ranksum.csv")):
        for row in read_csv(os.path.join(TAB, ckey)):
            m = [r for r in r1u if r["suite"] == suite and r["function"] == row["function"]
                 and r["competitor"] == row["competitor"]][0]
            if m["verdict_holm"] != row["verdict_holm"]:
                changed.append(f"{suite} {row['function']} {row['competitor']}: table {row['verdict_holm']} "
                               f"(p = {row['p'] or 'n/a'}), with tolerance {m['verdict_holm']} (p = {m['p']:.3g})")
    lines += ["", "Holm decisions that change when the tie tolerance is applied:"] + [f"- {c}" for c in changed] + [""]

    # --- R2: paired tests against the variants
    r2 = []
    for suite in SUITES:
        r2 += paired_against(data, fs, suite, "RTI-DE", ABLATION)
        r2 += paired_against(data, fs, suite, "RTI-DE", SENSITIVITY)
    # Holm above was over the union of both families per function; redo per family
    for fam in (ABLATION, SENSITIVITY):
        for suite in SUITES:
            for fid in functions(data, suite):
                block = [r for r in r2 if r["suite"] == suite and r["function"] == fid and r["other"] in fam]
                adj = holm([r["p"] for r in block])
                for r, pa in zip(block, adj):
                    r["p_holm"] = pa
                    r["verdict_holm"] = verdict(pa, r["W_ref_lower"], r["W_ref_higher"])
    write_csv("R2_paired_variants.csv", r2, ["suite", "function", "reference", "other", "pairs", "non_tied_pairs",
                                             "W_ref_lower", "W_ref_higher", "p", "verdict", "p_holm", "verdict_holm"])
    lines += ["## R2 Seed-paired signed-rank tests, RTI-DE against each variant (+ = RTI-DE better)", "",
              "| Variant | CEC2022 | CEC2017 |", "|---|---|---|"]
    for v in ABLATION + SENSITIVITY:
        cells = []
        for suite in SUITES:
            rows = [r for r in r2 if r["suite"] == suite and r["other"] == v]
            cells.append(f"{wtl(rows, 'verdict')} (Holm {wtl(rows, 'verdict_holm')})")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- R3: global Holm over all functions x competitors of a suite (unpaired p of the tables)
    r3 = []
    for suite, ckey in (("cec2022", "B2_cec2022_ranksum.csv"), ("cec2017", "C2_cec2017_ranksum.csv")):
        rows = read_csv(os.path.join(TAB, ckey))
        ps = [float(r["p"]) if r["p"] else float("nan") for r in rows]
        adj = holm(ps)
        for r, p, pa in zip(rows, ps, adj):
            better = r["verdict"] == "+"
            v = "≈" if math.isnan(pa) or pa >= ALPHA else ("+" if better or r["verdict"] == "+" else "-")
            if not math.isnan(pa) and pa < ALPHA and r["verdict"] == "≈":
                v = "?"                                       # cannot happen: global Holm is stricter
            r3.append({"suite": suite, "function": r["function"], "competitor": r["competitor"], "p": p,
                       "p_holm_global": pa, "verdict_table": r["verdict"], "verdict_holm_global": v})
    write_csv("R3_holm_global.csv", r3, ["suite", "function", "competitor", "p", "p_holm_global", "verdict_table",
                                         "verdict_holm_global"])
    lines += ["## R3 Unpaired rank-sum tests with Holm adjustment over all functions and competitors of a suite", "",
              "| Suite | " + " | ".join(COMPETITORS) + " |", "|---|" + "---|" * len(COMPETITORS)]
    for suite in SUITES:
        cells = [wtl([r for r in r3 if r["suite"] == suite and r["competitor"] == c], "verdict_holm_global")
                 for c in COMPETITORS]
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- R4: Friedman on medians
    r4 = []
    lines += ["## R4 Friedman test and Nemenyi CD on the median final errors", ""]
    for suite in SUITES:
        fids = functions(data, suite)
        tab = [[means[(suite, f, a)][1] for a in ALGOS] for f in fids]
        mr, chi2, p, cd = friedman(tab)
        row = {"suite": suite, "n": len(fids), "chi2": chi2, "p": p, "cd": cd}
        row.update({a: m for a, m in zip(ALGOS, mr)})
        r4.append(row)
        lines.append(f"- {suite}: chi2 = {chi2:.2f}, p = {p:.2e}, CD = {cd:.2f}; mean ranks: " +
                     ", ".join(f"{a} {m:.2f}" for a, m in zip(ALGOS, mr)))
    write_csv("R4_friedman_medians.csv", r4, ["suite", "n", "chi2", "p", "cd"] + ALGOS)
    lines.append("")

    # --- R5: variant without the chain against the competitors
    r5 = []
    lines += ["## R5 RTI-DE-noChain against the six competitors (unpaired rank-sum, Holm within function; "
              "and seed-paired signed-rank)", "",
              "| Suite | " + " | ".join(COMPETITORS) + " |", "|---|" + "---|" * len(COMPETITORS)]
    for suite in SUITES:
        for fid in functions(data, suite):
            a = errors(data, fs, suite, fid, "RTI-DE-noChain")
            block = []
            for c in COMPETITORS:
                b = errors(data, fs, suite, fid, c)
                p = mann_whitney(list(a.values()), list(b.values()))
                ma = sum(a.values()) / len(a)
                mb = sum(b.values()) / len(b)
                ra = avg_ranks(list(a.values()) + list(b.values()))
                lower = sum(ra[:len(a)]) < sum(ra[len(a):])
                block.append({"suite": suite, "function": fid, "competitor": c, "p": p, "noChain_lower_ranks": lower,
                              "mean_noChain": ma, "mean_competitor": mb})
            adj = holm([r["p"] for r in block])
            for r, pa in zip(block, adj):
                r["verdict"] = "≈" if math.isnan(r["p"]) or r["p"] >= ALPHA else ("+" if r["noChain_lower_ranks"] else "-")
                r["p_holm"] = pa
                r["verdict_holm"] = "≈" if math.isnan(pa) or pa >= ALPHA else ("+" if r["noChain_lower_ranks"] else "-")
            r5 += block
        cells = []
        for c in COMPETITORS:
            rows = [r for r in r5 if r["suite"] == suite and r["competitor"] == c]
            cells.append(f"{wtl(rows, 'verdict')} (Holm {wtl(rows, 'verdict_holm')})")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    write_csv("R5_nochain_vs_competitors.csv", r5, ["suite", "function", "competitor", "p", "verdict", "p_holm",
                                                     "verdict_holm", "noChain_lower_ranks", "mean_noChain",
                                                     "mean_competitor"])
    # Friedman with the variant without the chain in place of RTI-DE (means from the ablation summary)
    lines.append("")
    for suite, sub in (("cec2022", "cec2022_D20"), ("cec2017", "cec2017_D100")):
        nc = {r["function"]: float(r["mean"]) for r in read_csv(os.path.join(RS, "paper_ablation", sub, "summary.csv"))
              if r["algorithm"] == "RTI-DE-noChain"}
        fids = functions(data, suite)
        tab = [[nc[f]] + [means[(suite, f, a)][0] for a in COMPETITORS] for f in fids]
        mr, chi2, p, cd = friedman(tab)
        lines.append(f"- Friedman on means with RTI-DE-noChain in place of RTI-DE, {suite}: mean ranks " +
                     ", ".join(f"{a} {m:.2f}" for a, m in zip(["RTI-DE-noChain"] + COMPETITORS, mr)) +
                     f"; chi2 = {chi2:.2f}, CD = {cd:.2f}")
    lines.append("")
    # --- LaTeX tables for the paper
    def tex_table(name, label, caption, head, rows, note, align):
        body = [f"% Generated by scripts/supplementary_stats.py - do not edit by hand ({name})",
                "\\begin{table}[htbp]", "\\centering\\footnotesize",
                f"\\caption{{{caption}}}\\label{{tab:{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
                " & ".join(head) + " \\\\", "\\midrule"]
        for r in rows:
            body.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
        body += ["\\bottomrule", "\\end{tabular}",
                 f"\\par\\smallskip\\noindent\\parbox{{\\linewidth}}{{\\footnotesize {note}}}", "\\end{table}", ""]
        with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(body))

    def cnt(rows, key):
        return wtl(rows, key).replace("≈", "$\\approx$")

    d2 = {r["competitor"]: r for r in read_csv(os.path.join(TAB, "D2_win_tie_loss.csv"))}
    head = ["Suite", "Analysis"] + [display(c) for c in COMPETITORS]
    rows = []
    for suite, nm in (("cec2022", "CEC2022"), ("cec2017", "CEC2017")):
        rows.append([nm, "unpaired (Table~\\ref{tab:D2_win_tie_loss})"] + [d2[c][f"{suite}_holm"] for c in COMPETITORS])
        rows.append(["", "unpaired, tie tolerance"] + [cnt([r for r in r1u if r["suite"] == suite and r["competitor"] == c], "verdict_holm") for c in COMPETITORS])
        rows.append(["", "seed-paired"] + [cnt([r for r in r1 if r["suite"] == suite and r["other"] == c], "verdict_holm") for c in COMPETITORS])
        rows.append(["", "unpaired, Holm over the suite"] + [cnt([r for r in r3 if r["suite"] == suite and r["competitor"] == c], "verdict_holm_global") for c in COMPETITORS])
        rows.append(["", "RTI-DE-noChain, tie tolerance"] + [cnt([r for r in r5 if r["suite"] == suite and r["competitor"] == c], "verdict_holm") for c in COMPETITORS])
        rows.append("MID")
    for prob, nm in (("composite", "Laminate"), ("composite_contiguity", "Laminate (contig.)"), ("uav", "UAV")):
        rows.append([nm, "seed-paired"] + [cnt([r for r in r1 if r["suite"] == prob and r["other"] == c], "verdict_holm") for c in COMPETITORS])
    # Holm families with fewer than six comparisons: the test is not defined (nan) where all compared values
    # are equal, and holm() leaves such comparisons out of the family
    names = {"cec2022": "CEC2022", "cec2017": "CEC2017", "composite": "Laminate",
             "composite_contiguity": "Laminate (contig.)", "uav": "UAV"}
    unpaired = {suite: [{"function": r["function"], "p": float(r["p"]) if r["p"] else float("nan")}
                        for r in read_csv(os.path.join(TAB, ckey))]
                for suite, ckey in (("cec2022", "B2_cec2022_ranksum.csv"), ("cec2017", "C2_cec2017_ranksum.csv"))}
    smaller = []
    for suite in SUITES + ENGINEERING:
        analyses = [("``seed-paired''", [r for r in r1 if r["suite"] == suite])]
        if suite in SUITES:
            analyses = ([("``unpaired''", unpaired[suite]),
                         ("``unpaired, tie tolerance''", [r for r in r1u if r["suite"] == suite])] + analyses +
                        [("``RTI-DE-noChain, tie tolerance''", [r for r in r5 if r["suite"] == suite])])
        t = holm_smaller_text(analyses, len(COMPETITORS))
        if t:
            smaller.append(f"Smaller families on {names[suite]} (comparisons per function): {t}.")
    global_m = {s: sum(not math.isnan(r["p"]) for r in r3 if r["suite"] == s) for s in SUITES}
    tex_table("R7_robustness_competitors", "R7_robustness_competitors",
              "Robustness of the comparison with the competitors: $+/\\approx/-$ of RTI-DE (rows ``RTI-DE-noChain'': of the variant without the chain) under alternative test procedures.",
              head, [r.replace("≈", "$\\approx$") if isinstance(r, str) else [c.replace("≈", "$\\approx$") for c in r] for r in rows],
              "Holm adjustment for the six comparisons of a function unless stated otherwise, or for fewer where the test is "
              "not defined because all compared values are equal. " + "".join(s + " " for s in smaller) +
              "``Holm over the suite'': over all functions and competitors of the suite with a defined test "
              f"({global_m['cec2022']} comparisons on CEC2022, {global_m['cec2017']} on CEC2017). "
              "Tie tolerance: final values compared at the seven significant digits of the run log, which on CEC2022 ($F^*$ between 300 and 2700) corresponds to an absolute tolerance between about $10^{-4}$ and $10^{-3}$. "
              "Seed-paired: two-sided Wilcoxon signed-rank test on the runs with the same seed. "
              "The row ``unpaired, Holm over the suite'' uses the full-precision $p$-values of Tables~\\ref{tab:B2_cec2022_ranksum} and~\\ref{tab:C2_cec2017_ranksum}. "
              "In the seed-paired rows, only L-SHADE shares the initial population with RTI-DE; for the other competitors the runs are matched by seed index only. "
              "Computed with the Python standard library.",
              "ll" + "r" * len(COMPETITORS))
    f1 = {r["variant"]: r for r in read_csv(os.path.join(TAB, "F1_ablation_summary.csv"))}
    g1 = {r["variant"]: r for r in read_csv(os.path.join(TAB, "G1_sensitivity_summary.csv"))}
    rows = []
    for v in ABLATION + SENSITIVITY:
        src = f1.get(v) or g1.get(v)
        cells = [v]
        for suite in SUITES:
            cells.append(src[f"{suite}_wtl_holm"])
            cells.append(cnt([r for r in r2 if r["suite"] == suite and r["other"] == v], "verdict_holm"))
        rows.append([c.replace("≈", "$\\approx$") for c in cells])
        if v == ABLATION[-1]:
            rows.append("MID")
    smaller = []
    for fam, what in ((ABLATION, "ablation"), (SENSITIVITY, "sensitivity")):
        for suite in SUITES:
            t = holm_smaller([r for r in r2 if r["suite"] == suite and r["other"] in fam], len(fam))
            if t:
                smaller.append(f"{what} on {names[suite]} ({t})")
    tex_table("R8_paired_variants", "R8_paired_variants",
              "Ablation and sensitivity with seed-paired tests: $+/\\approx/-$ of RTI-DE against each variant ($+$: RTI-DE significantly better), Holm-adjusted.",
              ["Variant", "CEC2022 unpaired", "CEC2022 paired", "CEC2017 unpaired", "CEC2017 paired"], rows,
              "Unpaired: the Holm-adjusted counts of Tables~\\ref{tab:F1_ablation_summary} and~\\ref{tab:G1_sensitivity_summary}. "
              "Paired: two-sided Wilcoxon signed-rank test on the runs with the same seed, on the final values of the run log (seven significant digits); "
              "Holm adjustment for the 6 (ablation) or 8 (sensitivity) comparisons of a function, or for fewer where the "
              "test is not defined because every pair of runs has the same value" +
              (f"; smaller families (comparisons per function): {'; '.join(smaller)}." if smaller else "."),
              "lrrrr")

    with open(os.path.join(OUT, "R6_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
