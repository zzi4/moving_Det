# Motion Evidence Denoising Design

## Goal

Replace the dense max-frame-difference diagnostic with a robust, sparse motion
proposal signal that preserves moving 20x40-pixel VRUs while suppressing
photometric flicker, compression residuals, static edge halos, and isolated
noise. This phase is a three-scene proof of concept and does not retrain
MG-VTOD.

## Evidence motivating the change

On the fixed day, evening, and night 1024x1024 tiles, moving-target masks
dilated by 8 pixels occupy 1.31%-1.74% of the image while the current motion
map marks 38.09%-38.92% of pixels at or above 0.5. Only 3.08%-3.60% of those
hot pixels fall inside the dilated moving-target masks. Cached global ECC
correlations are 0.998994-0.999848 and toggling the cached transform changes
the hot fraction by less than 0.6 percentage points, so coarse global
registration is not the primary failure on these samples.

## Approved processing chain

1. Warp all valid support frames into the center-frame coordinates using the
   immutable alignment cache.
2. Apply robust per-support affine photometric correction in grayscale. Gain
   and bias are estimated from low-gradient pixels and clipped to conservative
   ranges so moving objects cannot dominate the fit.
3. Build a temporal-median background and measure the center residual against
   that background instead of taking the maximum difference over supports.
4. Normalize residuals by local robust noise and penalize high-gradient center
   pixels to suppress sub-pixel edge halos.
5. Convert the soft score to sparse proposals with hysteresis and connected
   components. Reject components outside configurable area and geometry limits,
   then close small gaps and return both a soft map and binary proposal mask.

## Boundaries

- Ground-truth OBBs are used only for evaluation and visualization, never to
  construct an inference-time proposal mask.
- The RGB detector remains responsible for static VRUs. Motion evidence is an
  optional enhancement and must not hard-mask RGB features.
- The existing `compute_motion_strength` API and the trained epoch-6 checkpoint
  remain unchanged in this diagnostic phase.
- No new third-party dependency is added; implementation uses PyTorch, NumPy,
  OpenCV, and Pillow already present in the project environment.

## Acceptance criteria

- Synthetic 20x40 moving rectangles survive photometric flicker and small
  camera residuals while static textured backgrounds remain sparse.
- Identical or globally brightness-shifted static clips produce no proposal.
- Invalid support frames and out-of-frame samples do not create proposals.
- For the three fixed real tiles, report current and improved hot fractions,
  moving-target coverage, hot-pixel concentration near moving targets, and
  connected-component counts.
- Produce one six-stage PNG per scene: RGB/GT, current map, photometric temporal
  residual, edge-suppressed score, binary proposals, and cleaned overlay.
