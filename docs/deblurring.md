# Deblurring stage

Where it sits: camera → FPGA (dark-reference subtraction, FIR filtering) → **deblurring on the Jetson** → hazard
models → ground. Same pattern as the hazard models: learned model vs classic method, test data used once.

## Status

| Step | What | Status |
|---|---|---|
| Data | Sharp chips from the hazard datasets + synthetic blur and noise (`src/gstp/blur.py`, `src/gstp/data/deblur.py`) | Done (code + tests) |
| Classic baseline | Unsharp mask, Wiener, Richardson–Lucy, tuned on validation (`src/gstp/deblur_classic.py`, notebook 10) | Ready to run on Colab |
| Learned model | NAFNet on blurred/sharp pairs | Next |
| Evaluation | PSNR / SSIM, and flood / burn-scar IoU on blurred vs deblurred test chips | After the model |
| Export | ONNX + rough speed check | Last |

## Design choices

- **Sharp images** are the HLS Burn Scars (6 bands, 30 m) and Sen1Floods11 Sentinel-2 (12 bands, 10 m) chips, with
  the same train / val / test splits as the hazard models, so the deblurred test chips can go straight into the
  burn-scar and flood models.
- **Blur types** (sizes in image pixels): optics (Gaussian, always present), extra Gaussian, defocus (disk), platform
  motion (line), attitude jitter (random path), and *none* (noise only). Noise = read noise + photon noise.
  The ranges are placeholders until the camera PSF/MTF and exposure time are known (`blur.RANGES`).
- **One PSF per image, same for all bands** — one exposure moves or defocuses all bands the same way.
- **Test set is fixed**: each test chip gets one seeded blur, blur types are assigned in turn so each is equally
  represented, and the recipes are saved to `results/deblur/test_degradations.json`.
- **Classic methods get two PSF settings**: the true PSF (best case, never this exact on orbit) and one nominal PSF
  (realistic when only a lab measurement exists). The learned model gets no PSF.
- **The *none* group checks double sharpening**: a method that sharpens an image that is already sharp lowers its
  PSNR. This matters if the FPGA FIR filter already sharpens.
- **The learned model will work band by band** (1 channel in, 1 out), so it does not depend on the camera's band set.

## Questions for the camera / FPGA team

1. What causes the blur we expect: platform motion or jitter during the exposure, defocus, or the optics themselves?
2. Is the camera PSF or MTF measured? What is the exposure (integration) time and the expected platform drift during it?
3. What does the FIR filter on the FPGA do — smoothing, sharpening, or both? (to avoid sharpening twice)
4. What arrives at the Jetson after the FPGA: format (YUYV / PGM / raw), bit depth, image size, bands?
5. What time budget per image does the Jetson have for deblurring before the hazard models run?
