# Flow-field phantoms

This report gives the results of the stage in `docs/plans/flow-field-phantoms.md`. The
stage tests the flow-field column solver on phantoms with a closed-form truth. Each
section comes from one PR of the plan. This version has the section *Kissing surfaces*
(PR 4). The sections *Hairpins*, *Jumps*, *Grid*, *Uncertainty* and *Checkpoint* come
from later PRs.

## Kissing surfaces

### What ran

The plan's PR 4 runs use a uniform grid at 0.16 mm. One such run takes about 7–8 h and
about 16–17 GB. The coordinator decided (CD-11) to run stand-ins at 0.32 mm first. The
stand-ins use the same code and the same defaults. At 0.32 mm the FEM prior has its
correct length ℓ = 0.5 mm (CD-6). **Every number in this section is a 0.32 mm stand-in
result.** One 0.16 mm run exists: ckiss_smooth, oracle, noise 0. Its results are in *The first 0.16 mm run*, below. Each other number is a 0.32 mm stand-in result.

Table 1. The PR 4 matrix and the state of each run (0.32 mm stand-in stage).

| phantom | source | noise | variant | state |
|---|---|---|---|---|
| ckiss_smooth | oracle | 0 | top | done at 0.32 mm (stand-in) |
| ckiss_sharp | oracle | 0 | top | done at 0.32 mm (stand-in) |
| ckiss_smooth | oracle | 0 | shiftB (R6) | done at 0.32 mm (stand-in) |
| ckiss_smooth | oracle | 0 | anchor (Q6) | done at 0.32 mm (stand-in; 2566 s, 2.33 GB) |
| ckiss_smooth | oracle | 0 | coarse first (Q6) | done at 0.32 mm (stand-in; 5152 s, 2.42 GB) |
| ckiss_smooth | oracle | 0 | top, 0.16 mm | done (10,592 s, peak 11.26 GB) |
| all four ckiss phantoms | image | 0 and 1 | top | not run: time |
| all four ckiss phantoms | oracle | 1 | top | not run: time |
| ckiss_smooth_flat, ckiss_sharp_flat | oracle | 0 | top | not run: time |
| every run at 0.16 mm after the first | | | | not run: time |

"Time" means this: at this cost, the plan's matrix does not fit in one night.

### Inputs, code and cost

- The stop field is the oracle: p = Φ(stop_sdf / 0.08 mm), clipped to [0.02, 0.98]
  (CD-4). The tensor is the oracle too.
- The solver uses the plan's schedule: widths 4, 2, 1, at most 200 sweeps for each width.
  No setting was changed. The re-centring option of CD-13 is off. The contrast term uses
  one profile line per direction (CD-7).
- The columns start at the roots (the plan's rule). The stand-ins were re-measured from
  their saved fields after PR 3's fix 88bc488. No new solve was made.
- The views and numbers come from `prototypes/scripts/ff_kiss_views.py`.

Table 2. Cost and inference (0.32 mm stand-in). Each run has 156,065 nodes.

| run | time (min) | peak memory (GB) | sweeps per width | end of each width | free-energy rises | git commit |
|---|---|---|---|---|---|---|
| ckiss_smooth top | 61.3 | 2.71 | 59 / 120 / 99 | free-energy tolerance (all) | 0 | 86f856f |
| ckiss_sharp top | 63.8 | 2.79 | 60 / 122 / 98 | free-energy tolerance (all) | 0 | 1105ed1 |
| ckiss_smooth shiftB | 56.9 | 2.68 | 59 / 98 / 102 | free-energy tolerance (all) | 0 | 6a92779 |

No stand-in reached the cap of 200 sweeps. In the 0.64 mm dev runs, the anchor run and
the coarse run reached the cap at width 4 (dev runs only). Commit 6a92779 adds the anchor
origin and the CD-13 option to the solver. The option is off in the shiftB run. The
solver's docstring states that the default path then gives the same bits; this report
did not check that.

![Free energy per sweep, per width](figures/flow-field-phantoms/kiss_free_energy_u0.32.png)

### Question 1: the fused contact

**Expected (plan).** With the oracle stop field, every stop is within one voxel of the
true contact, and no flow line crosses it.

Table 3. Columns whose true stop is a fused contact (CONTACT), 0.32 mm stand-in.
Distributions: median / p90 / max. A negative signed error is a stop in tissue, before
the true stop.

| run | bank | abs stop error (mm) | signed stop error (mm) | stopped / ended at an unresolved cell | own side of the slot plane |
|---|---|---|---|---|---|
| ckiss_smooth top | A | 0.272 / 0.431 / 0.553 | −0.272 / −0.133 / 0.074 | 445 / 25 | 1.000 |
| ckiss_smooth top | B | 0.170 / 0.293 / 0.483 | −0.170 / −0.060 / 0.121 | 577 / 46 | 0.998 |
| ckiss_sharp top | A | 0.290 / 0.459 / 0.588 | −0.290 / −0.142 / 0.165 | 548 / 22 | 1.000 |
| ckiss_sharp top | B | 0.181 / 0.325 / 0.505 | −0.181 / −0.065 / 0.029 | 629 / 20 | 0.998 |
| ckiss_smooth shiftB | A | 0.292 / 0.437 / 0.552 | −0.292 / −0.169 / 0.028 | 429 / 41 | 1.000 |
| ckiss_smooth shiftB | B | 0.123 / 0.215 / 0.391 | −0.123 / −0.030 / 0.075 | 561 / 62 | 0.991 |

- The CONTACT stops are early. The median error is 0.12–0.29 mm. Bank A has the larger
  error. At the open pia (PIA) the median error is 0.08–0.09 mm.
- Contact crossings: 1 (smooth), 1 (sharp), 3 (shiftB). All of them are bank-B columns.
  The tolerance is 0.02 mm (the half-width of the fused slot). The same count on the
  true paths is 0. The overlap and focal counts are 0 in all three runs.
- The width of the solved stop boundary (q_stop from 0.2 to 0.8 on the line through
  each stop) has a median of 0.21 mm, at CONTACT and at PIA stops alike.

**At 0.16 mm.** The CONTACT stops must come to within one voxel (0.16 mm). The contact
crossings must go to 0.

### Question 3: the end of the gap

**Expected (plan).** The smooth change behaves like the open and fused parts. The sharp
change can show grazing columns, sinks or a wrong boundary near the end face.

Table 4. Abs stop error (mm, median / p90 / max) by the local z of the true stop, and
places in the window round the gap's end (± 2.5 mm smooth, ± 0.5 mm sharp), 0.32 mm
stand-in.

| run | open part | window | fused part | unresolved cells: window / elsewhere | moving cortex nodes: window / elsewhere |
|---|---|---|---|---|---|
| ckiss_smooth top | 0.078 / 0.161 / 0.240 | 0.115 / 0.238 / 0.425 | 0.236 / 0.428 / 0.602 | 139 / 135 | 6056 / 19334 |
| ckiss_sharp top | 0.079 / 0.161 / 0.240 | 0.079 / 0.169 / 0.331 | 0.252 / 0.432 / 0.602 | 9 / 156 | 1165 / 24027 |
| ckiss_smooth shiftB | 0.082 / 0.165 / 0.391 | 0.094 / 0.194 / 0.358 | 0.207 / 0.403 / 0.552 | 154 / 170 | 6105 / 19436 |

- The stop error is small in the open part and larger in the fused part. In the fused
  part the error increases with z (figure).
- The window of the smooth phantom holds about half of the unresolved contact cells,
  but less than a quarter of the moving cortex nodes.
- There are no sinks and no focal points in any run, in the window or elsewhere.

**At 0.16 mm.** Compare the fused part with the open part. Look again for grazing
columns at the sharp end face.

![Stop error against the local z of the true stop](figures/flow-field-phantoms/kiss_q1q3_stop_vs_z_u0.32.png)

### Question 4: the stopped region

**Expected (plan).** The outside medium is stopped at the open slot. At the fused contact
the stopped sheet is thinner than a cell. On a uniform grid the sheet is not resolved.

Table 5. The stopped region, 0.32 mm stand-in.

| run | sheet length at the contact, q_stop > 0.5 (mm, median / p90 / max, 9 lines) | true 2g (mm) | open slot: medium nodes with q_stop > 0.5 | open slot: deep cortex nodes with q_stop > 0.5 |
|---|---|---|---|---|
| ckiss_smooth top | 0.552 / 0.669 / 0.741 | 0.04 | 0.716 (n 1603) | 0 (n 8718) |
| ckiss_sharp top | 0.584 / 0.706 / 0.734 | 0.04 | 0.716 (n 1603) | 0 (n 8718) |
| ckiss_smooth shiftB | 0.538 / 0.661 / 0.741 | 0.04 | 0.729 (n 1603) | 0 (n 8718) |

- At the fused contact the solved sheet is about 0.55 mm thick. That is more than one
  grid spacing (0.32 mm) and about 14 times the true sheet (0.04 mm).
- At the open slot, about 72 % of the medium nodes are stopped. No cortex node deeper
  than one spacing is stopped.

**At 0.16 mm.** The sheet length must scale with the spacing. The plan expects a
resolved sheet only where the octree refines (PR 5).

![q_stop sections, ckiss_smooth](figures/flow-field-phantoms/kiss_q4_sections_ckiss_smooth_u0.32.png)

![q_stop sections, ckiss_sharp](figures/flow-field-phantoms/kiss_q4_sections_ckiss_sharp_u0.32.png)

### Question 5: τ at a contact

**Expected (plan).** The bank speed ratio is near tA / tB = 2.5 / 1.2 = 2.08. The levels
match the truth after one shared monotone map.

Table 6. Speeds at the moving cortex nodes of each bank, 0.32 mm stand-in (median / p90 /
max).

| run | bank | solved speed | true speed | solved / true | τ arrival at CONTACT (true 0) |
|---|---|---|---|---|---|
| ckiss_smooth top | A | 1.04 / 1.09 / 1.80 | 1.35 (all nodes) | 0.771 / 0.839 / 1.33 | −0.559 / −0.418 / −0.128 |
| ckiss_smooth top | B | 1.06 / 1.18 / 1.66 | 0.649 / 0.679 / 0.989 | 1.61 / 1.80 / 2.56 | −0.511 / −0.396 / −0.105 |
| ckiss_sharp top | A | 1.04 / 1.11 / 1.86 | 1.35 (all nodes) | 0.774 / 0.854 / 5.00 | −0.573 / −0.435 / −0.021 |
| ckiss_sharp top | B | 1.07 / 1.19 / 1.77 | 0.649 / 0.680 / 0.989 | 1.62 / 1.83 / 8.76 | −0.526 / −0.414 / −0.008 |

- The solved speed is about 1.05 in both banks. The solved bank ratio A/B is 0.98 (run.json:
  0.983 smooth, 0.975 sharp), against 2.08.
- One possible cause is the speed prior: log v near 0, SD 0.3. log 2.08 = 0.73, about
  2.4 SD. The prior was not changed.
- The levels: in bank A the median signed deviation is near 0 at φ = 0.1–0.3 and 0.06–0.46
  mm at φ = 0.4–0.9. In bank B the levels are deep: −0.24 to −0.31 mm at φ = 0.2–0.5 and
  −0.08 to −0.18 mm at φ = 0.6–0.9. No bank-B column has a level at φ = 0.1. Both runs
  give these values. The abs level deviation in bank B is 0.19 / 0.32–0.33 / 0.60–0.69 mm
  (median / p90 / max).

**At 0.16 mm.** Look for the same speed ratio. A ratio near 1 at 0.16 mm also points at
the speed prior, not at the grid.

### Question 6: the origin of τ

The plan expects the same levels after one constant, and fewer sweeps with the top origin.

Table 8. Question 6 (0.32 mm stand-in, ckiss_smooth, oracle, noise 0).

| run | sweeps per width (fine grid) | other sweeps | result |
|---|---|---|---|
| top origin | 59 / 120 / 99 (278) | — | reference |
| anchor origin | 60 / 97 / 55 (212) | — | τ_anchor = τ_top + c, c = 1.31, over 151,687 nodes moving in both runs: \|residual\| 0.0028 / 0.0075 / 0.30 (median / p90 / max). Level points after c: 0.0069 / 0.030 / 0.19 mm apart (16,625 of 26,181 found in both runs) |
| top origin, coarse grid first | 68 / 177 / 101 (346) | 0.64 mm solve: 189 / 60 / 20 | CONTACT stop error median 0.34 / 0.26 mm (bank A / B), against 0.27 / 0.17 without the coarse grid; direction against ∇τ median 5.3°, against 1.9° |

- The levels agree after one constant. This supports R4.
- The anchor run needs fewer sweeps than the top run (212 against 278). The plan expects the opposite. This does not support the plan's claim that the top origin converges faster.
- The coarse grid first needs more sweeps on the fine grid (346 against 278), and ends at a different field. So the warm start from 0.64 mm does not help here. Every width of every stand-in ended by the free-energy tolerance.
- The anchor run and the coarse run were measured from the roots (commit 518aca3 for the coarse run; the anchor run was re-measured).

![Q6: the anchor run against the top run](figures/flow-field-phantoms/kiss_q6_ckiss_smooth_oracle_0_u0.32.png)


- The plan's coarse grid is an initial field only, not a prior (CD-9). So the sweep
  comparison measures a warm start. The 0.64 mm first stage carries a different FEM
  prior (CD-6).
- In PR 3 the plan's exact test *Origin* failed in both parts (CD-12). A moved anchor
  bends the field locally, by up to 0.066 in τ. With the top origin, |τ arrival| is
  0.05–0.24, against a τ step Δτ of 0.04.

### The independence test (R6)

The shiftB run changes only bank B's image (its profile moves 0.2 mm toward the top).
The oracle stop field, the tensor and the truth do not change.

Table 7. |Δ tau_mean| and the angle between the two mean directions at nodes that move
in both runs, by the distance from the true stop surface, 0.32 mm stand-in (median / p90
/ max).

| distance (mm) | bank A: abs Δτ | bank A: angle (°) | bank B: abs Δτ | bank B: angle (°) |
|---|---|---|---|---|
| 0–0.32 | 0.0006 / 0.027 / 0.11 | 0.28 / 10.5 / 28.2 | 0.026 / 0.066 / 0.27 | 1.25 / 12.7 / 33.8 |
| 0.32–0.64 | 0.0046 / 0.040 / 0.11 | 0.22 / 4.3 / 40.7 | 0.020 / 0.044 / 0.24 | 0.20 / 1.6 / 28.4 |
| 0.64–0.96 | 0.0032 / 0.022 / 0.097 | 0.02 / 0.18 / 11.7 | 0.012 / 0.029 / 0.096 | 0.05 / 0.22 / 15.8 |
| 0.96–1.28 | 0.0019 / 0.011 / 0.072 | 0.01 / 0.06 / 2.1 | 0.006 / 0.017 / 0.063 | 0.02 / 0.08 / 1.3 |
| 1.28–2.0 | 0.0011 / 0.0053 / 0.060 | 0.0042 / 0.028 / 0.66 | 0.0030 / 0.0096 / 0.048 | 0.011 / 0.044 / 0.63 |

The last row joins three bins (1.28–1.6, 1.6–1.92, 1.92–2.0 mm). Each value in it is the
largest value of the three bins.

- Bank A changes when only bank B's image changes. In the first bin (0–0.32 mm) the p90
  of abs Δτ is 0.027 in bank A and 0.066 in bank B. In the next bin they are almost
  equal (0.040 and 0.044). The change falls with the distance from the stop surface in
  both banks.
- Per column (bank A, CONTACT stops, n 470): the stops move by 0.021 / 0.18 / 0.36 mm,
  and 42 columns change their end reason.
- Possible causes, not separated: coupling across unresolved contact cells (the stopped
  sheet is 0.55 mm at 0.32 mm; the true sheet is 0.04 mm), and the mean field that ends
  at a different restricted minimum. PR 3 found that most moving nodes end at the edge of
  their kept τ candidates; the CD-13 option is under test.

![R6: bank A and bank B change](figures/flow-field-phantoms/kiss_r6_ckiss_smooth_oracle_0_u0.32.png)

### The first 0.16 mm run

ckiss_smooth, oracle inputs, noise 0, uniform grid at 0.16 mm (the plan's grid). 1,179,648 nodes. Time 10,592 s (2 h 57 min), peak RSS 11.26 GB (memguard), 4 threads. Sweeps per width 18 / 38 / 39; every width ended by the free-energy tolerance; 0 free-energy rises. Measured from the roots (re-measured from the saved field after 88bc488).

Table 9. The 0.16 mm run against the 0.32 mm stand-in of the same phantom.

| measure | 0.16 mm | 0.32 mm stand-in |
|---|---|---|
| CONTACT stop error, bank A / B (median; p90; max of A) | 0.124 / 0.077 mm; 0.16; 0.19 | 0.27 / 0.17 mm |
| stop error by kind, median: PIA / CONTACT / HAIRPIN | 0.074 / 0.097 / 0.057 mm | 0.08 / 0.21 / 0.12 mm |
| stops on own side (flat banks) A / B | 1 / 1 | 1 / 0.999 |
| contact crossings (flat banks); on true paths | 0; 0 | 1; 0 |
| overlap / focal points / sinks | 0 / 0 / 0 | 0 / 0 / 0 |
| unresolved contact cells; columns ended UNRESOLVED | 304; 247 | 274; 237 |
| stopped sheet at the fused contact (true 0.04 mm) | 0.22 mm | 0.55 mm |
| stop ramp q_stop 0.2 → 0.8, median, CONTACT / PIA | 0.107 / 0.107 mm | 0.21 / 0.21 mm |
| speed ratio A/B (true 2.08); solved/true bank A, bank B | 0.976; 0.76, 1.61 | 0.983; 0.77, 1.61 |
| τ_arrival at CONTACT, median, bank A / B | −0.19 / −0.22 | −0.56 / −0.51 |
| integrator jump > 0 (no abrasion) | 2661 of 2662, median 0.083 | 2672 of 2672, median 0.33 |
| level deviation, abs, median / p90 / max | 0.149 / 0.42 / 0.55 mm | — |
| spacing near the stop, solved / true, median | 1.01 | 1.20 |
| direction against ∇τ, median | 1.1° | 1.9° |
| sweeps per width | 18 / 38 / 39 | 59 / 120 / 99 |

- Question 1: the CONTACT stops are within one voxel (0.16 mm) at the median and the p90 of both banks; the largest error is 0.19 mm (bank A). No flow line crosses the contact, and no cortex overlaps.
- Question 4: the stopped sheet at the fused contact is 0.22 mm, about 1.4 spacings, against the true 0.04 mm. The plan expects the sheet to be resolved only where the octree refines.
- Question 5: the speed ratio stays near 1 (0.976), against 2.08. The grid does not change it. One possible cause is the speed prior (log v near 0, SD 0.3; log 2.08 = 0.73, about 2.4 SD). The prior is not changed.
- R2 and R3: the integrator's jump is still positive at almost every stop, but it falls with the spacing (median 0.33 → 0.083). This agrees with the caveat in the R2 row: the jump comes from the masked interpolation, over about one spacing.
- The solve needed far fewer sweeps at 0.16 mm than at 0.32 mm (95 against 278).

![q_stop in sections at 0.16 mm](figures/flow-field-phantoms/kiss_q4_sections_ckiss_smooth_u0.16.png)

![Stop error against z at 0.16 mm](figures/flow-field-phantoms/kiss_q1q3_stop_vs_z_u0.16.png)

![Free energy at 0.16 mm](figures/flow-field-phantoms/kiss_free_energy_u0.16.png)

### Design decisions R1–R7 and R11

Evidence: the 0.32 mm stand-ins, and one 0.16 mm run (ckiss_smooth, oracle, noise 0; Table 9). R6 and R4 have 0.32 mm evidence only. R8–R10 are not in PR 4.

| # | supports | does not support | failure signature (plan) seen |
|---|---|---|---|
| R1 stop as a node state | stops on the own side of the slot plane (≥ 0.991); no cortex node stopped at the open slot. At 0.16 mm: CONTACT stops within one voxel at the median and p90 (0.124 / 0.077 mm; max 0.19 mm), own side 1 / 1 | CONTACT stops early by 0.12–0.29 mm (median); PIA 0.08–0.09 mm | not decided: the stop error and the sheet are at the scale of the spacing; no image run |
| R2 speed to zero at the stop, τ jumps to τ_top | — | the integrator's jump −τ_arrival is positive at all stops: 2672 of 2672 (smooth; median 0.33), 2994 of 2994 (sharp), 2581 of 2581 (shiftB), also at the pia | seen, with a caveat: τ at the q_stop = 0.5 crossing comes from the last moving nodes, up to one spacing below the stop (masked interpolation). It can measure the integrator, not the solver's jump state. A likely cause is that of CD-12. The provisional change for R2 (#208) failed again; its diagnosis agrees: the column jump is −τ_arrival from the masked interpolation, and the stop boundary sits beyond the pia because of the entropy of the moving candidates |
| R3 jump prior near zero | — | as R2 | seen, with the caveat of R2 |
| R4 τ_top = 0 as origin | the anchor run gives the same levels after one constant (\|residual\| median 0.003, p90 0.0075) | the top origin does not need fewer sweeps (278 against 212 with the anchor); the coarse grid first needs more (346) | different levels: not seen. Sweeps: the top origin needs 31 % more than the anchor (278 against 212); whether this is "many more sweeps" is for the user |
| R5 no overlap | overlap 0; contact crossings 0 on the true paths. At 0.16 mm: contact crossings 0, overlap 0 | contact crossings 1, 1, 3, all bank B. Each crossing column starts at its root in bank B, far from the fundus (local y 4.2–6.4 mm): not a start effect. Four of the five end STOPPED 0.000–0.010 mm past the far face of the fused slot (g = 0.02 mm): they overshoot the contact by less than 1/16 voxel. The fifth (shiftB) ends STOPPED inside the slot at local z = 0.74 mm, where g = 0.092 mm (the window of the gap's end): 0.024 mm past the slot plane, 0.068 mm before the far face. No crossing column runs on into bank A | seen (count above zero), small |
| R6 no interpolation across the stop boundary | bank A changes less than bank B | bank A changes when only bank B changes (Table 7) | seen |
| R7 non-crossing, sinks checked | sinks 0, sink columns 0, focal points 0, overlap 0 (also at 0.16 mm; reversed edges 9 there) | reversed edges 3, 3, 13 (their places against the stop set are not checked here) | not seen for sinks, focal points and overlap at this grid |
| R11 direction tied to ∇τ | direction against ∇τ: median 1.9–2.5° (1.1° at 0.16 mm) | — | not seen |

### The image inputs (CD-5)

The image stop terms are ranking scores, not calibrated log-odds (CD-5). No image run
ran in PR 4. So this section has no evidence on the stop field from the image.
