# Review of the version-2 plan, round 2

A planning file: the findings of a review of `docs/version-2-plan.md`, worked
through one by one on this branch. Each fix goes into the plan (or a spec);
this file is deleted before the branch merges.

The review read the plan as it is after #118 (open PRs table, phase 4's
comparison from #83, the flat map's closed pieces). Status: **open** (not yet
discussed), **fixed** (the plan now says it), **kept** (discussed; no change),
**moved** (to another PR or session), **deferred** (noted; settled
when the work it concerns starts).

Deferred, the validation and learning measures: they are settled at phase 9,
when generations are trained and scored. What the next work builds is only the
data they are computed from, so the edit log now records raw facts (§1 item 4,
phase 3 step 2, viewer phase 2) and §1 calls the measures provisional. B1 and
B2 wait for the deflected ray (stream 2).

## A. Contradictions

| # | Where | Finding | Status |
|---|---|---|---|
| A0 | §1 item 3 and data table; §3c step 6; phase 10; §5c Open (carry) | The outcome measures do not state (1) the version they judge, (2) a reference made against that version, (3) exposure: what was examined and how it was chosen (ranked or random), (4) the goal they serve. So counts are not rates, the queue (chosen by the model) scores the model, and σ, the spread and carry reach are judged on sites they chose. Proposed rule for §1: every outcome measure states these four; quantities the model computes about itself (queue length, margins, spread) are diagnostics only. Settles A1, A9, phase 10's diagnostic and carry reach; touches B1, B2, C12. | deferred |
| A1 | §1 item 3; §2 manual inputs; §3c step 6 | The success measure counts "answers (site answers and scribbles) that change voxels of the `dseg`", but scribbles are not applied to the `dseg` ("too sparse to apply"). For scribbles the count is always 0, and "each answer serves twice … corrects this hemisphere now" is false for them. Redefine (for example "disagrees with the `dseg`") or settle the Open item on applying scribbles. | deferred; log fixed |
| A2 | §1 item 2; §3c step 1 coupling 3 | Temperature scaling divides logits by T, so logit 0 stays 0 and the P = 0.5 contour, where handle walls fall, does not move. It rescales only the margins. Moving the 0.5 crossing needs a bias term (Platt or vector scaling). Also, generation 1 has no labels in the 0.5 band (seed labels are ignored there), so its calibration near 0.5 cannot be measured, yet phase 8 removes the labels-only path at that point. | deferred |
| A3 | §3c step 1 coupling 5; step 4; phase 10 | Two meanings of P. Coupling 5 says the majority rule "keeps P close to a tissue fraction", and `antialias` reads P − 0.5 as a sub-voxel position. Phase 10 reads P as a probability, P = Φ(−μ/σ). A calibrated classifier on majority labels estimates Pr(fraction > ½ \| image), which sharpens to a step as the image gets more informative: not a fraction. | deferred |
| A4 | §3c step 2, tight folds | s = 0.30 mm "must not be a whole number of voxels along any axis", but 0.30 mm is 6 voxels at 50 µm. The data table checks constants on the next hemisphere, not at a new resolution. | open |
| A5 | §4b standing decisions | The 8 mm cap is justified by "cortex is at most about 4.5 mm thick … crosses a fused contact into the far bank's WM even when the ray meets it at an angle". Two 4.5 mm banks face to face need 9 mm, more at an angle (1/cos θ). | open |
| A6 | §5b standing decisions; §5c step 4 | "Only bands of at least 2–3 voxels; the stria is at this limit" against step 4's 1.5-voxel minimum, chosen because the stria is 1.67 voxels and a 2.5-voxel threshold drops it. | open |
| A7 | §6c step 3 (PE lesson); §3c step 2; §5c step 7 | "Edits … replay exactly in any order" against stream 1 and stream 3 answers that replay in time order, the later winning. | open |
| A8 | §3c Open, enclosed ventricles | One option keeps an enclosed ventricle as "an inner surface piece, b2 > 0", but `dseg` refuses to write unless Betti is (b0, 0, 0), and §3a and `wm-dseg/1` require genus-0 pieces with no cavities. The option needs a spec change that the plan does not state. | open |
| A9 | §3c step 1 coupling 4; §1 data table; step 6 | "The number of handles and cavities … sets the length of the review queue", but the queue also holds folds, margins, spread, stream 3 sites and random sites. It ranks with no threshold, so nothing sets its length, yet the data table calls its length a success measure. | deferred |
| A10 | §5b standing decisions; §5c steps 3 and 5 | "Where there is none, depth follows equal volume" against "the current estimator is distance, until an equal-volume estimator is validated". | open |
| A11 | §1 Goals; §3c step 4 | "Whatever can be computed locally from published files is a sampler, not a file", but normals and \|n̄\| are local functions of `white-surf/1` and are published as files. Curvature follows the rule. | open |
| A12 | §6c flat map (#118); §3c steps 2 and 4; `docs/plans/phase-1.0-pipeline.md` | The number of white-surface pieces disagrees: 21 WM islands kept as closed pieces (#118); "b0 = 28 … the excision step removes them"; "one closed genus-0 piece" (antialias result); "b0 is 2" (phase 1.0 plan). Every island vertex is a column root, so the count matters to streams 2 and 3 too. | open |
| A13 | §3c step 2, the filled mask | FreeSurfer's rule keeps "the largest 18-connected component per hemisphere", which would delete the islands of A12. The plan does not say which connectivity ours uses (§2 allows only Kuhn adjacency) or why islands stay. | open |

## B. Circularities and loops

| # | Where | Finding | Status |
|---|---|---|---|
| B1 | §4c, later-phase methods; §4c Open | The case against fields counts "own-bank returns", which needs a same-bank test. The plan drops `SAME_MM` as "wrong at both faces" and says no step labels banks. The same count is a criterion for the deflected ray. | deferred |
| B2 | §4c step 2; §4c Open; §5c step 5 | Normal rays' 0.17 mm at M1/S1 is "the bar" for the deflected ray, but it was measured with the mirrored template, which §5c calls a record that assumes face-to-face banks. The case for straight rays over fields leans on the same measure. | deferred |
| B3 | §5c steps 3, 4 and 5; §9 | Bands are found in τ from the warp, and the warp places levels at bands. Thickness is a warp term, and thickness evidence comes from the warp. Contact tops use "the end of its own layer sequence", which needs layer identity (step 4). No order of iteration or stopping rule. The §9 table says 3.4 needs only 3.1, but step 4's inputs list step 3. | open |
| B4 | §1 item 2; phase 8; phase 10 | With one hemisphere, inference runs on the training hemisphere: at each voxel 4 of 5 fold models trained on it. The spread is mostly one out-of-fold model against four in-sample ones, not "what more labels can fix". | deferred |
| B5 | §3c step 2, the topology guarantee | An answer overridden by the guarantee goes back to the queue, is answered again, and can be overridden again. No exit from the loop. | open |

## C. Unsupported or doubtful claims

| # | Where | Finding | Status |
|---|---|---|---|
| C1 | §5c step 3, extents by censoring | "Since a sample does not depend on the extent, growing it changes no level that was already inside." The joint warp is a coupled energy: new samples change the data term and can move levels in the same column and its neighbours. | open |
| C2 | §5c steps 3 and 4 | τ "has no zero" and "only orders the levels", but r_ℓ comes "from the spacing of the levels", and bands sit at τ 0.18, 0.37, 0.81. Spacing needs a metric on τ. | open |
| C3 | §5c step 3 | "Certifies 96.6–100 % of variables, so the remaining errors belong to the model." True only for the certified variables; up to 3.4 % can carry solver error. | open |
| C4 | §2 the image; §3c step 3 | "Whitening by C is the maximum-likelihood answer" is not derived, and the noise is spatially correlated. In the log domain noise is about σ/I, so a global C is wrong where signal varies. The domain of "echo 1 noise ≈ 0.6 of echo 2's" is not stated (thermal magnitude noise does not depend on TE). | open |
| C5 | phase 10 | σ = −μ/Φ⁻¹(P) is 0/0 at the boundary (μ → 0, P → 0.5), where the antialias weight 1/σ² acts. | deferred |
| C6 | §3c step 4 | \|n̄\| "separates thin blades" rests on a minimum of 0.038 and Spearman 0.307 against blade thickness: weak support for weighting roots by it. | open |
| C7 | §1 constraints | "No discrete jump that rounding can flip" cannot hold as written: wall order by P and its ties, the 0.25× and 2.0 thresholds, majority labels and QPBO tie-breaks are discrete decisions on float inputs. | open |
| C8 | §2, no bias-field correction | "Every intensity comparison here is local" against the excision detector's global threshold (bootstrap only; limit 1 admits it). | open |
| C9 | §3c step 3, local cortex level | The bound takes the slowest laminar wavelength as T. A monotone depth trend across T has a fundamental nearer 2T, which roughly doubles the lower bound on σ and puts it above the T/2 upper bound. | open |
| C10 | §3c step 2; §5c Open, cut tops | The excision map is the test set for cut tops, but it has a known one-voxel edge error at cut faces (limit 2), where cut tops are measured. | open |
| C11 | §3c step 4, curvature | The rank correlation 0.85–0.98 was measured on the old twicing mesh at 0.32 mm, not on the antialiased surface. | open |
| C12 | §3d phase 4 (#118) | The #83 variants are compared by tight-fold sites, voxels changed and area, and "the comparison answers" fills against thickening. The standing decision judges surfaces by normal quality (angle spread, misfacing faces, blade cancellation, agreement with the data normal), and "acceptable" is a judgment for views (diagnostics over criteria). | open |

## D. Gaps

| # | Where | Finding | Status |
|---|---|---|---|
| D1 | §3c step 2, air bubbles; §4c, §5c inputs | Streams 2 and 3 "distrust samples inside" the bubble mask, but no input table of theirs lists it and no code reads it. After phase 8 no step makes it. | open |
| D2 | §4c step 3; §3c step 3; label-set/1 | `label-set/1` and the ray-sampler spec put fluid in outside (label 0). The plan writes "cortex, WM, fluid or outside", "fluid and outside as obstacles", "open (fluid before any WM)" as if they were separate. The cut-face rule (WM to outside, no cortex) cannot tell a sulcal channel against WM from a cut. | open |
| D3 | §5c step 7; phase 7 | How a point answer becomes anchors: which columns, how far it reaches; what "two conflicting answers" are in phase 7's test. | open |
| D4 | §5c step 4 | "Bands per region" and Ward merging, but `regions` is not ported and the design does not define a region. | open |
| D5 | §1 constraints; §3c step 1 | How a network trained at 160 µm runs at 50 µm without resampling, under the native-resolution rule (the patch context shrinks about 3×). | open |
| D6 | §3c step 4; §5c step 1; §7 | Scales fixed in mesh steps or mm: the 10 normal-smoothing steps (and so \|n̄\|), the 0.04 mm column step. §7 asks to restate hard-wired scales on purpose. | open |
| D7 | §3c step 3 outputs | Names "stream 3 (planned checks of laminar orientation)"; §5 has no such check. | open |
| D8 | §7 drop list; §3c step 2 | `wellcomposed` is dropped, but fold repair "makes the mask well-composed". Nothing in `src/` uses it; #85 is not merged. | open |
| D9 | §6c flat map (#118) | "The `exp` patch method" is named but defined nowhere in the plan. #55's slit was "written for the box flat map", so it may need a port that §6d does not list. | open |

## E. Not current

| # | Where | Finding | Status |
|---|---|---|---|
| E1 | §7, the bug | "#94 has the fix; it lands first …": #94 and #113 are merged. | open |
| E2 | About this document | "Only stream 1 has run on the whole hemisphere": stream 2 phase 0 reports a 27 s hemisphere run. | open |
| E3 | §4c step 4 | The ray-sampler spec is "none yet" and the traces "a prototype `.npz`", but `docs/formats/ray-sampler.md` exists. | open |
| E4 | §8 | Mixes "exists" with phase names for specs that exist (ray sampler, column volume); no row for `label-set/1`; `wm-dseg-fixes/1` is inside `wm-dseg.md`, not a file of its own. | open |
| E5 | §3c steps 4 and 5 | 12.7 M vertices, 25.3 M → 210 k faces: from before the ventricles moved outside (+5.9 %). | open |
| E6 | §4d; §9 | Rerun lists differ: §4d names stream 1 phases 3, 4, 6, 8 (3 is formats, 6 phantoms only); §9 names 1.4, 1.8, 1.9. | open |
| E7 | §6d; §9 | §6d says viewer phase 5 needs viewer phase 3; §9 says 4.0.5 and 3.7. §6d's order omits 4.0.5 before 4.2. Stream 3 scaling is "alongside phases 3–5" in §5d and "needs 3.3, 3.5" in §9. | open |
| E8 | §3c step 2; §5c Open; §6c | Tight folds "152 at 91 sites" then "74 of 90 sites"; §5c Open says normal rays "use [the closed form] at 0.5 mm", but stream 2 now measures A(s); FreeSurfer's 0.1 weight stated under "Decided" before the result moves to 1. | open |
