# Drop NiiVue from the viewer

The viewer's volume pane has two engines: NiiVue on a box round the
selection, and our own bricks viewer on the whole volume. Two engines double
the work of each new volume-pane feature, and NiiVue loads one box into the
browser, which does not scale to 50 µm whole brain (plan §9, phase 0.5). The
3D view of the volume pane is not needed: the surface pane already shows the
white surface in 3D. This plan makes the bricks viewer the only volume
viewer and removes NiiVue.

Read the plan's §1 (working rules) and the viewer entries of §9.

## Goal

1. The bricks viewer does everything in the volume pane that the review
   needs.
2. NiiVue, its box routes and its 3D view are gone.

## Facts from the current code

What NiiVue does now (`static/index.html`, "volume pane (NiiVue, world
mm)"):

- It loads a box round the selection: the echo (1, 2 or 3, the `echo`
  menu), an overlay image (WM mask, handle cut, handle fill, cavity fill;
  the `overlay` opacity slider) and the white surface clipped to the box.
  Routes: `/white/echo<k>.nii.gz`, `/white/overlay.nii.gz`,
  `/white/surface.gii` (`white_view.box`).
- Views: the three slices, one slice, or a 3D render (the `view` menu).
- The WM selection as a sphere, and the columns as `.tck` lines.
- A shift-click in a slice moves the WM selection.
- It is the default engine. The page imports it from a CDN
  (`@niivue/niivue@0.69.0`).

What the bricks viewer has now (`bricks.py`, `volumes.py`,
`static/overlays.js`, the brick pane in `index.html`):

- The whole volume as a brick pyramid at levels of detail, the three slices.
- Label layers as their own pyramids: the segmentation and the fixes image
  (`desc-wmtopofixes`), with colour tables, fill or outline.
- The outlines of the true white surface on each slice.
- The columns' crossings as dots, and (#151) the shift-click section in the
  clicked slice.
- One echo: the `--echo` image.

## Parity: what the bricks viewer needs first

1. **The echo choice.** Echo 1, 2 or 3 as the background, from
   `--echo-dir`, each its own pyramid, built on first use like the others.
2. **The WM mask as a layer.** The filled labels of `desc-wmtopo` (its
   `FilledLabels`) as one label layer, so the brick pane shows what NiiVue's
   overlay shows: the WM mask, and the handle cuts, handle fills and cavity
   fills from the fixes image. Check whether the segmentation layer with the
   fixes layer already gives this; add the mask layer only if not.
3. **One opacity control** for the label layers, like NiiVue's slider.
4. **The WM selection marker** on the slices, if the crosshair does not
   already show it.
5. **The volume pane's views:** the three slices, or one slice large. No 3D
   render.

Each item gets an exact test where a route gives it (for example, the echo
pyramids against block means of the echo files; the mask layer against the
filled labels).

## Removal

1. The bricks viewer becomes the default and the only engine. The `viewer`
   menu goes. `?vol=bricks` is accepted and ignored, so old links still
   work.
2. NiiVue goes: its import, `initVolume`, `loadWhiteBox`, the NiiVue parts
   of the selection, crosshair and shift-click code, and its `view`
   options.
3. The box routes go: `/white/echo<k>.nii.gz`, `/white/overlay.nii.gz`,
   `/white/surface.gii`, and `white_view.box` if nothing else uses it.
4. The `.tck` loading into NiiVue goes. The `.tck` download link of the
   columns pane stays: it is a standard format that other tools read. So
   `tck.py` stays; its test's NiiVue transcription is renamed to say what
   it checks (the line offsets of a reader that starts the first line at
   the header).
5. The docs: the NiiVue notes in plan §9 (the viewer entries) and in the
   module docstrings.

## Exact tests

- The parity items as above.
- `test_the_white_mode_serves_every_route`: the box routes give 404; every
  other route of 4.0, 4.0.5, 4.2, 4.3 and the viewer requests still serves.
- The page has no reference to NiiVue (a test that reads `index.html` and
  the static scripts).

## The check

In the built-in browser pane, on `scratch/v2/derivatives-main/`: the
volume pane opens with the bricks viewer; each echo; the label layers
against the old NiiVue overlay at 10909480 (screenshots side by side, the
old one from `main` before the change); the shift-click section; the
columns and roots. The page loads nothing from the NiiVue CDN (the network
log).

## PRs

1. `feature/bricks-parity`: the parity items. From `origin/main`, after
   #151 merges (it changes the brick pane's column controls).
2. `refactor/drop-niivue`: the removal. After PR 1 merges.

## Decisions for review (made by the coordinator)

- Parity first, then removal, in two PRs, so the review never runs a
  viewer without the overlay it needs.
- The `.tck` download stays: removing NiiVue does not remove the need to
  open columns in other tools.
- No 3D render in the volume pane; the surface pane is the 3D view.
