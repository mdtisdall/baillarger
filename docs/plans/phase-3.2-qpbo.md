# Phase 3.2: a QPBO that is not GPL

Stream 3, phase 2 of `docs/version-2-plan.md` (§5d). This phase needs no
other phase. It is in round 1 (`docs/plans/round-1.md`).

## Goal

Replace `thinqpbo` (GPL-3) with our own QPBO and QPBO-I. Keep the public API
of `baillarger.surfcut`. After this phase, no file in the repository imports
`thinqpbo`, and `thinqpbo` is not in `pyproject.toml` or `uv.lock`.

## Done when

1. `surfcut.qpbo`, `surfcut.qpbo_pruned` and `surfcut.reduce` have the same
   signatures and the same `Result` fields as on `main` today.
2. The solver uses only scipy (BSD) and numpy (BSD). It uses no GPL code.
   It uses no GPL code as a test reference.
3. The exact tests in "Tests" pass.
4. `_thinqpbo.py` is deleted. `pyproject.toml` and `uv.lock` do not list
   `thinqpbo`. The mypy entry for `thinqpbo` is deleted.
5. `prototypes/scripts/warp_cut.py` calls `surfcut`, not `thinqpbo`.
6. `nix develop --command scripts/check` passes.
7. The PR body gives the complexity and memory analysis (see "Analysis").

## Facts from the current code

All paths are relative to the repository root.

- `src/baillarger/surfcut/_thinqpbo.py` wraps `thinqpbo.QPBODouble`. It also
  seeds the C library `rand()` through ctypes. That is global state.
- `src/baillarger/surfcut/qpbo.py`:
  - `qpbo(M, start_labels=None, improve=True, log=None, seed=0) -> Result`.
  - `qpbo_pruned(M, start_labels=None, log=None, seed=0) -> Result`. It runs
    QPBO without improve, fixes the certified variables with `reduce`, then
    runs QPBO-I on the free variables.
  - `reduce(M, fixed, labels, big=None)`. Do not change it.
  - The energy is built from `cut.graph(M, extra)` (Ishikawa arcs). The
    binary node y(i,k) is [x_i ≥ k]. Arcs give submodular pairwise terms.
    Ties and table entries with a positive second difference give
    non-submodular E11 terms.
  - The thinqpbo features in use: `solve`, `compute_weak_persistencies`,
    `get_label` (negative means unlabelled), `compute_twice_lower_bound`,
    `set_label`, `improve()`. Probe is not in use.
  - All terms are float64 now.
- `src/baillarger/surfcut/cut.py` already solves the submodular case with
  `scipy.sparse.csgraph.maximum_flow(method="dinic")`. It scales capacities by
  `SCALE = 1000` and rounds them. It does not check the 2³⁰ limit.
- Sizes on the boxes ("Crossing paths and contacts", report lines 578–613):
  21,600 binary nodes and 308,425 pairwise terms in the benchmark. V1V2:
  23,124 traces, 41,998 pairs, 1,597 ties, 96.6 % certified. M1S1: 99.8 %
  certified. Each run took 9–11 min and 8–10 GB.
- Callers outside `src/`: `prototypes/scripts/warp_cut.py` (imports thinqpbo
  directly), `joint_levels.py`, `edit_priority.py`, `edit_resolve.py`,
  `pia_editor.py`, `edit_data.py`, and their tests in `prototypes/scripts/`.
  These tests are in the pytest `testpaths`.
- Licences in the environment: scipy BSD-3, numpy BSD-3, networkx BSD-3,
  ortools Apache-2.0 (dev group only). thinqpbo is GPL.

## Design

### The QPBO solve

Use the construction of Kolmogorov and Rother (2007), "Minimizing
nonsubmodular functions with graph cuts — a review". Write it from the paper,
not from any GPL source.

1. Make a graph with two nodes for each binary variable: p and p̄.
2. Put each term in normal form. Each unary and each pairwise term becomes
   arcs on the doubled graph. A submodular term gives arcs between p and q,
   and between q̄ and p̄. A non-submodular term gives arcs between p and q̄,
   and between q and p̄. Each arc carries half the reparametrized cost.
3. Solve the max flow with `scipy.sparse.csgraph.maximum_flow`.
4. Find the strongly persistent labels. Do a BFS from the source and a BFS
   to the sink on the residual graph. Use the labelling rules in §3 of the
   paper.
5. Find the weak persistencies. Compute the strongly connected components
   of the residual graph with
   `scipy.sparse.csgraph.connected_components(connection="strong")`. Use the
   rule in the paper that labels whole components.
6. The lower bound is the flow plus the constant from the reparametrization.
   Report it in the same units as `energy(M, x)`. The current offset
   (`qpbo.py` line 246) shows how the result is aligned now.

### Integer capacities

`maximum_flow` takes int32 capacities. With int64 it returns a flow of 0
and gives no error (#14). Write one function that converts float costs to
int32 capacities, and use it in `cut.py` and in the new QPBO:

1. Multiply every cost by one scale factor. Round to the nearest integer.
2. Make the hard cost 1 + the total of the finite capacities. Do not use
   1000 × anything.
3. If the total capacity is 2³⁰ or more, raise `ValueError`. Give the total
   and the scale in the message.
4. Record the scale factor and the largest rounding error in the result's
   log.

The scale factor is a parameter with a default. Do not tune the default to
make a test pass. If a real model hits the limit, stop and report to the
coordinator.

### QPBO-I

Use the "improve" method of Rother et al. (2007), "Optimizing binary MRFs via
extended roof duality".

1. Start from a full labelling: `start_labels`, or `cut(M)` when it is
   `None`. This is the same rule as now.
2. In each pass, pick a subset of the unlabelled variables. Fix the other
   variables to the current labelling. Solve QPBO on the subset.
3. Keep the new labels only where QPBO labels them, and only if the energy
   does not increase.
4. Stop after a fixed number of passes, or when a full round gives no
   decrease.
5. Draw the subsets from `numpy.random.Generator(PCG64(seed))`. Use no global
   random state: no `np.random.*` module functions, no `random`, no C
   `rand()`.

### Determinism

The same model and seed give the same `Result`, bit for bit, on every run.
Sort every edge list before it goes to the solver. Use stable sorts.

## Tests

All tests are exact. Put them in `tests/test_surfcut.py` or a new
`tests/test_qpbo_solver.py`.

1. **Brute force on small models.** Use the current `random_model` (2–4
   variables, K 2–5) and pure binary models with up to 12 variables.
   - Each strongly persistent label agrees with every minimizer.
   - The lower bound is at most the true minimum.
   - With no non-submodular term, every variable is labelled and the energy
     equals the brute-force minimum.
2. **Submodular equals min cut.** On submodular models, the QPBO energy
   equals `cut(M)`'s energy.
3. **Known-answer graphs.** Write two or three small graphs from the paper's
   examples by hand. Give the expected labels in the test.
4. **Integer capacities.** A model whose total is just below 2³⁰ solves. A
   model whose total is 2³⁰ raises `ValueError`.
5. **Determinism.** Two runs give the same `Result`. A change to
   `np.random.seed` and to `random.seed` between the runs changes nothing.
   Delete `test_qpbo_does_not_depend_on_the_c_rand_state`. This test replaces
   it.
6. **Keep the current tests.** The CP-SAT tests on `folded_model`, the
   interval tests, `test_qpbo_bound_is_precise_with_forced_nodes` and
   `test_reduce_default_big_with_large_tables` stay. They must pass without
   a change to their tolerances. If one fails, diagnose the cause. Do not
   change the tolerance.

## Diagnostic on the boxes

Do this step after the tests pass. It is a diagnostic, not a gate.

1. Find the inputs of the V1V2 and M1S1 joint-warp runs (see
   `prototypes/scripts/joint_levels.py` and the report). If they are not in
   `data/` or `prototypes/`, write this in the PR and skip the step.
2. Run each box with the new solver, under `scripts/memguard`, with the cap at
   about 15 GB. Run one box at a time.
3. Put a table in the PR body: for each box, the percentage certified, the
   energy, the lower bound and the wall time. Put the numbers from the report
   beside them.
4. If the percentage certified falls by more than a few points, report it to
   the coordinator before you open the PR.

## Analysis

Write this in the PR body. Use the sizes that grow: binary nodes N and
pairwise terms E.

- The time of Dinic's max flow on the doubled graph (2N nodes, about 4E
  arcs), and of the two BFS passes and the SCC pass.
- The memory: bytes for each arc in the CSR arrays and the residual graph.
  Give the peak for the benchmark size above.
- What QPBO-I adds for each pass.
- One paragraph on the whole-brain case: the plan solves the joint warp on
  overlapping superblocks (§5d, scaling item 2). Give the size of one solve
  at that scale if a superblock holds about 10⁶ columns. Do not design the
  blocked solve here.

## Work breakdown

The phase session owns the design and reviews every diff. Use the
`parallel-agents` skill for sub-agents. Sub-agents do not run git commands
that write.

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Write the integer-capacity function and its tests. Change `cut.py` to use it. | Sonnet sub-agent | 2 |
| 2 | Write the test harness: brute force, known-answer graphs, determinism. Write tests against the public API only. | Sonnet sub-agent | 1 |
| 3 | Write the QPBO solve (§ "The QPBO solve"). | Phase session | after 1 |
| 4 | Write QPBO-I. | Phase session | after 3 |
| 5 | Change `qpbo.py` to use the new solver. Delete `_thinqpbo.py`. | Phase session | after 4 |
| 6 | Change `warp_cut.py` and the prototype test docstrings that say `uv run --with thinqpbo`. Remove `thinqpbo` from `pyproject.toml`. Run `uv lock`. | Haiku sub-agent | after 5 |
| 7 | Diagnostic on the boxes. | Phase session | after 6 |
| 8 | Analysis, PR body, `ship` skill. | Phase session | after 7 |

## PRs

One PR is the default: `feature/qpbo-permissive`. If step 1 is ready before
the solver, you can open it as its own PR first, and stack the solver PR on
its branch.

## Coordination

- `pyproject.toml` and `uv.lock` can also change in phase 1.0. If there is a
  conflict, merge `origin/main` into your branch and run `uv lock` again.
- Send questions to the coordinator session. Do not change a spec or the
  design without an answer.
