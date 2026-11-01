# Deblurring stage

Where it sits: camera → FPGA (dark-reference subtraction, FIR filtering) → **deblurring on the Jetson** → hazard
models → ground. Same pattern as the hazard models: learned model vs classic method, test data used once.

## Status

| Step | What | Status |
|---|---|---|
| Data | Sharp chips from the hazard datasets + synthetic blur and small leftover noise (`src/gstp/blur.py`, `src/gstp/data/deblur.py`) | Notebook 10 |
| Classic baseline | Wiener and Richardson–Lucy, true PSF and nominal PSF, tuned on validation (`src/gstp/deblur_classic.py`) | Notebook 10 |
| Learned models | Plain autoencoder (no skips) and NAFNet (skips + residual), band by band (`src/gstp/deblur_models.py`, `src/gstp/train_deblur.py`) | Notebook 11 |
| Evaluation | PSNR / SSIM per blur type, and flood / burn-scar IoU on sharp vs blurred vs deblurred test chips (`src/gstp/hazard_check.py`) | Notebook 11 |
| Export | ONNX (256×256 tiles) + speed on T4 and CPU | Notebook 11 |

## Design choices

- **Sharp images** are the HLS Burn Scars (6 bands, 30 m) and Sen1Floods11 Sentinel-2 (12 bands, 10 m) chips, with
  the same train / val / test splits as the hazard models, so the deblurred test chips can go straight into the
  burn-scar and flood models.
- **Blur types** (sizes in image pixels): motion (platform motion as a straight line, or attitude jitter as a random
  path), defocus (disk), Gaussian (optical softness), plus a *no extra blur* check group. A weak base optics blur is in
  every image. The ranges are placeholders until the camera PSF/MTF and exposure time are known (`blur.RANGES`).
- **Noise:** noise removal is the FPGA's job, so the model is not a denoiser. Only a small leftover noise is added
  (std up to 0.002 reflectance, placeholder) so a method that amplifies noise is penalised.
- **One PSF per image, same for all bands** — one exposure moves or defocuses all bands the same way.
- **Test set is fixed**: each test chip gets one seeded blur, blur types are assigned in turn so each is equally
  represented, and the recipes are saved to `results/deblur/test_degradations.json`.
- **Classic methods get two PSF settings**: the true PSF (best case, never this exact on orbit) and one nominal PSF
  (realistic when only a lab measurement exists). The learned model gets no PSF.
- **The *no extra blur* group checks double sharpening**: a method that sharpens an image that is already sharp lowers its
  PSNR. This matters if the FPGA FIR filter already sharpens.
- **The learned models work band by band** (1 channel in, 1 out) on 256×256 tiles, so they do not depend on the
  camera's band set and keep memory small on the Jetson. They never get the PSF.
- **Downstream check:** the trained burn-scar and flood (S2) U-Nets run on the sharp, blurred and deblurred versions
  of every test chip, with their own validation threshold, and IoU is compared. Reflectance is clipped to [0, 1], so
  the "sharp" IoU can differ slightly from the hazard models' own test numbers.

## Questions for the camera / FPGA team

1. What causes the blur we expect: platform motion or jitter during the exposure, defocus, or the optics themselves?
2. Is the camera PSF or MTF measured? What is the exposure (integration) time and the expected platform drift during it?
3. How much noise is left after the FPGA? What does the FIR filter on the FPGA do — smoothing, sharpening, or both? (to avoid sharpening twice)
4. What arrives at the Jetson after the FPGA: format (YUYV / PGM / raw), bit depth, image size, bands?
5. What time budget per image does the Jetson have for deblurring before the hazard models run?
