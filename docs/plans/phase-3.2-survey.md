# Phase 3.2: survey of solvers, decision and results

This file is part of phase 3.2 (`phase-3.2-qpbo.md`). The coordinator asked
for it before the solver PR. It has four parts:

1. a finding on the box that changes the plan;
2. libraries;
3. algorithms, with GPU support, whole-brain scale and determinism;
4. the decision, the results of the comparison, and what to keep for the
   whole-brain phase.

The full survey texts are in the shared notes folder of round 1
(`phase-3.2-survey-libraries.md`, `phase-3.2-survey-algorithms.md`). This file keeps the
conclusions.

Terms: N is the number of binary nodes (Ishikawa: one for each variable and
label above 0). E is the number of binary pairwise terms. A is the number of
arcs of the QPBO graph, about 2N + 4E.

## 1. Finding: int32 capacities are too coarse

### What the box showed

The plan's rule is: costs × 1000, rounded to integers, int32 capacities, and
a hard cost of 1 + the total below 2³⁰. With this rule, the V1V2 box at full
density (`joint_site.py --spacing 0.16`) gave a lower bound above the energy
in every try. In the fine pass the bound was 4845.24 and the energy 4838.38. A
valid lower bound cannot be above the energy.

### Cause (measured)

- The solver finds an exact optimum of the rounded energy. On this box QPBO
  labels every node, so the bound equals the rounded energy of the labels.
- The rounding error of the costs that are active at those labels sums to
  +1.869 on the light box (0.3 mm spacing, 729,006 active costs). This is
  exactly the excess of the bound over the true energy.
- The costs' fractional parts at 1/1000 are not uniform, so the errors do
  not cancel. 2.6 M of 4.7 M costs are below 0.0005 and round to 0.
- The non-submodular costs are not lost. Of 1.27 M non-submodular entries,
  1.26 M round to 0, but their sum is 5·10⁻¹¹: they are float residue of
  second differences. The real non-submodular costs (sum about 1,600) stay.

### Consequence

- With int32, the labels, the certificates and the bound are those of the
  rounded energy, not of the true energy. The error grows with the number of
  terms.
- At superblock scale (2·10⁹ arcs), int32 leaves less than one unit for each
  arc, so the rule cannot work there at all.
- The plan's lesson "capacities below 2³⁰" (§5c step 3, §2) is correct for
  scipy, but scipy is the wrong engine for this model.

### Test of the fix (light box)

Five rules, each with the same solver. The test sets were the light V1V2 box
(0.3 mm), its fine pass, and 300 small models with brute force (costs near
0.002 and near 10⁻⁶).

| rule | bound − E(labels), light box | final energy | time | small models |
|---|---|---|---|---|
| int32, ×1000, nearest (the plan) | +2.68 (not valid) | 1907.766 | 149 s | bound above the minimum; wrong labels |
| int32, ×1000, down | −54.76 (valid, loose) | 1909.285 | 262 s | wrong weak labels on 91 of 300 |
| int32, down, scale from a feasible cut (3.69·10⁵) | −0.243 | 1907.375 | 649 s | clean |
| int64, ×10⁹, nearest | +3.3·10⁻⁶ (not valid) | 1907.428 | 109 s | bound above the minimum by up to 1.6·10⁻⁹ |
| **int64, ×10⁹, down (adopted)** | **−5·10⁻⁵** | **1907.366** | **68 s** | **clean** |

The adopted rule had a peak of 4.1 GB. All five rules certified 100 % of
the nodes on the light box.

When the costs are rounded down, every flow of the rounded graph is a flow
of the true graph. So the bound is a valid lower bound of the true energy by
weak duality, at every scale. The bound is below the true one by at most the
total rounding error, which the log now reports. At ×10⁹ this is at most
10⁻⁹ for each arc.

### Certified labels and ties

A certified label is persistent for the rounded energy. For the true
energy, a labelling with the other value can be better than the labelling
found by at most the total rounding error. Where two true minima differ only
on one variable, the rounding can break the tie. In that case the solver
labels the variable strongly, at one of the two values. This was found in a
comparison with dwave's roof duality on costs with 3 decimals: the two
solvers broke one tie in opposite directions (model of 51 variables,
total rounding error 8.8·10⁻⁸). Every solver that rounds the costs to integers
has this property.

### 100 % certified on the V1V2 box: a possible effect of the rounding

The report's thinqpbo runs certified 96.6 % of the V1V2 box. The int32 run in
this phase certified 100 % in the fine pass. The int32 rounding sets 2.6 M
small costs to 0, and this can remove frustrated cycles. Thus the 100 %
can be an effect of the rounding, not of the solver. On the light box, the
int64 rule also certifies 100 %. This does not settle the question, because
thinqpbo's 96.6 % is for the full-density box. A full-density V1V2 run with
the int64 rule settles it.

## 2. Libraries

33 candidates were checked. The licences were read from the repository
licence file or the PyPI metadata (URLs in the full text). Nothing was
installed, and no GPL source was read.

### Licence-clean, with a fit

| Library | Licence | What it gives us | Maintained | uv, macOS arm64 / Linux CUDA |
|---|---|---|---|---|
| OR-Tools (`ortools`) | Apache-2.0 | `SimpleMaxFlow`: push-relabel, int64 capacities, numpy bulk API; CP-SAT (exact); PDLP (LP) | yes (9.15, 2026-01) | wheels / wheels (CPU only) |
| dwave-preprocessing | Apache-2.0 | `roof_duality`: roof-dual bound and strong or weak persistencies, own push-relabel | yes (2025-10) | wheels / wheels |
| pytoulbar2 | MIT (bundled Boost and GMP licences not checked) | exact solver for cost-function networks: takes the multi-label energy directly | yes (2026-09) | wheels / wheels |
| highspy (HiGHS) | MIT | LP and MIP; GPU PDLP from HiGHS 1.10 | yes (2026-07) | wheels / wheels |
| cuOpt (`cuopt-cu12`) | Apache-2.0 | LP on the GPU (PDLP) at 10⁸+ variables | yes (2026-10) | none / wheels |
| ECL-MaxFlow | BSD-3-Clause | GPU push-relabel with GPU global relabel | 2025-11 | no Python binding |
| scipy | BSD-3-Clause | `maximum_flow` (Dinic): int32 only, one thread | yes | wheels / wheels |
| networkx | BSD-3-Clause | max flow in pure Python: an exact reference for tests only | yes | wheels / wheels |

### Excluded

- **GPL:** thinqpbo, thinmaxflow, PyMaxflow, python-igraph, FastDOG
  (LPMP/BDD).
- **No licence file:** pyqpbo (it also downloads GPL QPBO), WBPR,
  LPMP/LP_MP.
- **Research use only:** GCO (gco-wrapper, pygco), thinhpf.
- **A permissive wrapper around GPL or research-only code:** OpenGM (its
  QPBO, TRW-S, max-flow and GCO paths), nifty (its QPBO is GPL-3.0), and
  slgbuilder (it needs thinqpbo and thinmaxflow).
- **LGPL:** AD3 (also stale) and graph-tool (not installable with uv).
- **Unknown origin:** shrdr (MIT, parallel QPBO). Its README points its QPBO
  to a fork of the GPL QPBO with no licence file. Do not use it unless its
  authors confirm that the code is their own.
- **No fit:** PyTorch, JAX, cuGraph and rustworkx have no s-t max flow.

## 3. Algorithms

References are in the full text. Two facts apply to every method:

- **Ties are submodular after a label reversal.** With x̄_b = K − 1 − x_b, a
  tie w|x_a + x_b − L| becomes an L1 term on a difference. So the energy is
  non-submodular only on cycles with an odd number of ties (for example a
  loop across a fold and back round the fundus). QPBO already gets what a
  reversal gives (roof duality does not change under it). Methods that need
  submodularity can work on the rest.
- **Integer max flow is deterministic in its output.** The minimum cut value
  is unique. The smallest source set (BFS from s in the residual graph) and
  the strongly connected components of the residual graph are the same for
  every maximum flow (Picard and Queyranne 1980). So labels defined from them
  are the same on every device, whatever order a parallel push-relabel uses.
  QPBO-I depends on the order of variables, so that order must be fixed.

### Candidates

| Candidate | Gives | Gives up against QPBO | Gains |
|---|---|---|---|
| **0. QPBO + QPBO-I (current, own code)** | exact optimum if submodular; roof-dual bound; persistent labels | — | — |
| **1. QPBO on a blocked (region) integer max flow** (Delong and Boykov 2008; Shekhovtsov and Hlaváč 2013; Strandmark and Kahl 2010) | the same as 0 | nothing | memory per region; neighbouring superblocks agree exactly |
| **2. Dual decomposition on the GPU** (column chains by DP; Komodakis et al. 2011) **+ maximum-persistency certificates** (Shekhovtsov et al. 2018) | local-polytope LP bound, labels, certificates | exact flow; convergence only to a tolerance | GPU (CUDA, MPS); memory O(E·K) floats; no doubled graph |
| **3. Convex lifting by primal-dual on the GPU** (Pock et al. 2008, 2010; Chambolle and Pock 2011), only to narrow each variable's label band, then exact QPBO | the same as 0, on a smaller graph | needs a check that no label is at a band edge | GPU; a 2–10× smaller exact solve |
| **4. Fusion moves with QPBO** (Lempitsky et al. 2010) | a labelling never worse than its two inputs | no bound | joins overlapping superblocks with one node per overlap column |
| **5. Exact solve of uncertified parts** (CP-SAT or toulbar2) | a certified optimum where the parts are small | — | removes the 0–3.4 % left to QPBO-I |
| TRW-S, MPLP | LP bound (TRW-S can stop below it), labels | certificates; sequential schedule | memory |
| Alpha-expansion, range moves | labels only | bound; ties and hard order are not metrics | — |
| LP (HiGHS), CP-SAT, MaxSAT on the whole energy | exact bound or optimum | size: 10⁸–10¹⁰ variables per superblock | — |
| Hochbaum and L♮-convex methods | exact on convex unaries | our unaries are not convex | — |

### GPU, scale and determinism

Sizes per column were measured on the V1V2 box (23,124 columns). Coarse
pass: 35 binary nodes and 474 merged pairs for each column. Fine pass: 26
and 270. Peak memory of the current solver: about 130 bytes per arc (A = 2N +
4E). "Superblock" means 10⁶ columns; "whole brain" means 3.5·10⁸ columns,
about 420 superblock solves for each pass with 20 % overlap.

| | Superblock: N, E, A | Whole brain: A for each pass |
|---|---|---|
| coarse pass | 3.5·10⁷, 4.7·10⁸, 2.0·10⁹ | 6.9·10¹¹ |
| fine pass | 2.6·10⁷, 2.7·10⁸, 1.1·10⁹ | 4.0·10¹¹ |

| | GPU | Time and memory | One superblock | Same result on every device? | Capacities |
|---|---|---|---|---|---|
| **0. Current (scipy Dinic, int32, CPU)** | none. A GPU version needs: the max flow as synchronous push-relabel in tensor operations (int64 scatter-add, global relabel by BFS); the two residual BFS passes as frontier BFS (easy); the SCC by forward–backward methods (harder). QPBO-I stays on the CPU. | Dinic O(N²A) worst case (Ishikawa graphs have long chains of hard arcs); BFS and SCC O(N + A); 130 B/arc | 147–256 GB: does not fit | the cut value and the labels (from the residual BFS) yes; QPBO-I yes with its fixed seed | int32: too coarse (§1) |
| **0′. Current with an int64 engine (OR-Tools push-relabel, CPU)** | none (OR-Tools is CPU only) | push-relabel: O(N²A) worst case, O(N²√A) with highest-label selection (which OR-Tools uses is not checked); the rest as 0 | does not fit at 130 B/arc; about 23–40 GB at a lean 20 B/arc | yes, as 0 | int64: 10⁻⁹ resolution |
| **1. Blocked integer max flow** | the inner solver can be GPU push-relabel (no mature PyTorch code); BFS and SCC on the GPU; QPBO-I on the CPU | sweeps of O(A) over regions; memory for each region only (2–4 GB for 10⁵ columns at 20 B/arc) | fits a laptop with streamed regions | yes: labels from the residual graph | int64 |
| **2. Dual decomposition + certificates** | yes: batched DP in PyTorch (CUDA, MPS); no code exists | O(E) per iteration; 10²–10⁴ iterations; about 8 B per binary pair | 2–4 GB | float32 iterates differ by rounding; to make the reported result exact, round the final multipliers to a grid, then compute the bound, certificates and labels in int64 | float32 inside, int64 for the results |
| **3. Lifting + band + exact QPBO** | yes for the relaxation (CUDA, MPS) | O(N + A) per iteration; 10³–10⁴ iterations; about 16 B/arc in float32 | 18–32 GB, then a 2–10× smaller QPBO | the final result comes from the exact QPBO, so yes | float32, then int64 |
| **4. Fusion moves** | not needed | one node per overlap column | small | yes | int64 |
| **5. Exact solve of uncertified parts** | no | depends on the part sizes (not measured); CP-SAT to about 10²–10³ variables per part | small | yes with one worker and a fixed tie-break | int64 |

Note on memory: the 130 B/arc of the current solver comes from the build
(int64 rows and columns, float costs, the sort for merging) and from scipy's
internal copies. A lean CSR with an int32 head, an int32 reverse-arc index
and an int64 residual capacity needs about 16 B/arc plus about 24 B per
node.

## 4. Decision and results

### Decided, in phase 3.2

1. **Keep our own QPBO and QPBO-I.** They pass every exact test. On the box,
   their labels and bound are correct for the energy that they receive.
2. **int64 capacities.** Costs × 10⁹ (`SCALE`, a parameter), rounded down.
   The hard cost is 1 + the total of the finite capacities. `ValueError` if
   it reaches 2⁶¹. The log gives the scale, the hard cost, the largest
   rounding error of one arc and the total rounding error.
3. **The engine: OR-Tools `SimpleMaxFlow`** (Apache-2.0, push-relabel,
   int64), behind one function, `_flow.max_flow`. `ortools` moves to the
   main dependencies. `cut()` uses the same engine. **For the user to
   confirm.** The alternative is our own int64 max flow in numba now. It is
   more work now, but it is also the first step to the blocked max flow of
   the whole-brain phase.
4. PR #96 has the rule and the exact tests that fail under int32. The solver
   PR is stacked on it.

### Results of the comparison

| check | models | result |
|---|---|---|
| brute force (`test_roof.py`, `test_qpbo_solver.py`) | 2–12 binary variables, random tables, forbidden entries | bound ≤ minimum; strong labels agree with every minimum; weak labels with some minimum |
| dwave-preprocessing `roof_duality` (Apache-2.0, an independent implementation) | 300 models of up to 60 variables | the same strong labels and bounds (difference ≤ 1.8·10⁻⁸); with integer costs, an exact match (test `test_roof_against_dwave`) |
| HiGHS local-polytope LP (scipy `linprog`) | 300 models of up to 80 variables | LP − roof bound between −7·10⁻¹⁵ and 2.1·10⁻⁸: the same bound, as the theory says (test `test_roof_bound_is_the_local_polytope_bound`) |
| CP-SAT exact optimum (`test_surfcut.py`) | folded two-bank models with ties | unchanged from the report: the energy is never below the optimum; the certified labels lie on an optimum |
| determinism | the report's benchmark model (21,600 binary nodes), two runs with `np.random` and `random` seeded differently | the same labels, intervals, certified labels, bound (522.629) and energy (524.395) |

The LP must merge the tables of each pair first. With separate tables for
one pair, the LP is looser than the roof dual. A first version of the test
did not merge them, and its bound was below the minimum by 0.25.

dwave-preprocessing (with dimod) is a dev dependency only. The LP uses
scipy's HiGHS, which does not clash with OR-Tools (see below), so no new
dependency is necessary for it.

### The HiGHS clash

When the HiGHS of the `highspy` package is loaded first, OR-Tools fails to
import ("Symbol not found"). This is true for `SimpleMaxFlow` and for
CP-SAT. When OR-Tools is imported first, both work. scipy's own HiGHS
(`linprog`, `milp`) does not clash. `highspy` and `cvxpy` stay in
`tests/reference`, which has its own environment.

### Earlier solver work in this repository

- **#14, scipy's `maximum_flow`.** It was chosen because PyMaxflow is GPL.
  With int64 capacities, scipy returned a flow of 0 on a test graph. The cut
  was checked against PyMaxflow on 1,130 boundaries. That model was
  submodular, with costs that the int32 scaling did not damage.
- **CP-SAT.** It was never a production solver. It is the exact reference in
  the tests. On the benchmark model, QPBO-I was better than CP-SAT after
  60 s (energy 1022.9 against 1029.2 with 8 workers).
- **#32, the hard cost.** thinqpbo's bound was sometimes above the energy
  until the hard cost became 1 + the finite total.
- **#33, determinism.** thinqpbo's QPBO-I used C `rand()`. Our QPBO-I uses
  `numpy.random.Generator(PCG64(seed))` and no global state.
- **Dual decomposition by subgradient.** It was tried and dropped: within
  0.015 of the LP bound after 3,000 iterations on small models, and the
  Polyak step cycled.
- **`tv.py`.** A CPU float64 Chambolle–Pock solver for total variation.
  It is not a cut solver.
- **#61.** The GPU work (torch) is for the antialiasing solver only.

### Later: the whole-brain phase (§5d, scaling item 2)

- Candidate 1 (QPBO on a blocked integer max flow, lean CSR, int64) as the
  main route. It is the only exact route that fits a laptop at superblock
  scale. Its output does not depend on the device.
- Candidate 4 (fusion moves) to join overlapping superblocks.
- Candidate 2 (GPU dual decomposition with certificates) as the research
  alternative, if the blocked max flow is too slow.
- A GPU push-relabel as an experiment for the inner solver. Its speed on
  Ishikawa graphs (long chains of hard arcs) is not known.

### Changes to the plan document

- §2 and §5c step 3: "capacities below 2³⁰ (int32)" becomes "int64
  capacities, costs rounded down, total below 2⁶¹". scipy's `maximum_flow`
  is not used for the joint warp.
- §5d phase 2: "our own on scipy's `maximum_flow`" becomes "our own QPBO on
  an int64 max flow (OR-Tools now; our own blocked max flow for the whole
  brain)".
- §5d scaling item 2: add the memory per arc (130 B now, 16–20 B needed) and
  the superblock sizes above.
