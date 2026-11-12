# sfcnn: CNN classification of star-forming vs non-star-forming cores in JCMT SCUBA-2 maps

Final-year project code. A convolutional neural network looks at a small cutout of a
JCMT SCUBA-2 850 µm dust-continuum map centred on a dense core and predicts whether
the core is **star-forming** (contains an embedded protostar) or **non-star-forming**
(starless/prestellar core or diffuse clump).

```
SCUBA-2 map ──► find cores ──► label by YSO cross-match ──► 32×32 cutouts ──► SFNet CNN ──► P(star-forming)
   (FITS)      (peaks / catalogue)   (Class 0/I/Flat within 1 beam)   (S/N + shape channels)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests                              # 8 tests, ~10 s
python -m sfcnn demo                                # synthetic survey → train → evaluate → predict (~5 min on CPU)
```

Or step by step:

```bash
python -m sfcnn synth                                # 12 synthetic SCUBA-2 regions + YSO catalogues
python -m sfcnn build                                # detect, label, cut out → data/processed/dataset.npz/.csv
python -m sfcnn train                                # → runs/sfnet/best.pt, training_curves.png
python -m sfcnn evaluate --checkpoint runs/sfnet/best.pt
python -m sfcnn classify --checkpoint runs/sfnet/best.pt --map some_map.fits
python -m sfcnn probmap  --checkpoint runs/sfnet/best.pt --map some_map.fits
```

All settings live in [configs/default.yaml](configs/default.yaml). Use `-c your.yaml` to switch config.

> **Windows note:** if `astropy` fails with *"DLL load failed … filename or extension is too long"*,
> your virtualenv path is too long. Create it at a short path (e.g. `C:\venvs\sfcnn`) or enable
> Windows long paths.

## Method

### 1. Data
JCMT SCUBA-2 850 µm maps (14.1″ effective beam, typically 3″ pixels). The loader
([sfcnn/fits_io.py](sfcnn/fits_io.py)) handles the usual ORAC-DR/SMURF FITS layout: a 3-D
cube with a degenerate frequency axis plus a `VARIANCE` extension.

### 2. Core detection ([sfcnn/sources.py](sfcnn/sources.py))
Local maxima of the lightly smoothed S/N map above 5σ, at least one beam apart. Alternatively
supply a published core catalogue (`source_catalog` in the manifest). Using an existing catalogue
is the more defensible choice for a thesis.

### 3. Labels ([sfcnn/labels.py](sfcnn/labels.py))
Labels come from cross-matching with infrared YSO catalogues:

| condition | label |
|---|---|
| Class 0 / I / flat-spectrum protostar within 15″ (~1 beam) of the peak | **1: star-forming** |
| no protostar within 30″ **and** no Class II/III on the peak | **0: non-star-forming** |
| anything in between | dropped (ambiguous) |

The gap between the two radii, and dropping cores with a Class II star on the peak, keeps
astrometric scatter and misclassified YSOs from turning into label noise.

### 4. Inputs ([sfcnn/cutouts.py](sfcnn/cutouts.py))
Each core becomes a 96″ × 96″ cutout resampled to 32 × 32 pixels with two channels:
1. `asinh(S/N / 3)`: significance with compressed dynamic range;
2. `flux / central flux`, clipped to [-1, 2]: shape only, independent of brightness.

Both channels are **unit-free**, so maps in pW and in mJy/beam, or with different depths, can be
mixed.

### 5. Model ([sfcnn/model.py](sfcnn/model.py))
SFNet is a small VGG-style CNN: three blocks of (conv-BN-ReLU)×2 with widths 32/64/128, then
global average pooling, dropout and two dense layers. It has about 0.3 M parameters, sized for
hundreds to thousands of labelled cores.
- **Augmentation:** all 8 rotations/flips (the sky has no preferred orientation) plus ±1 px shifts.
- **Class imbalance:** positive-class weight in the BCE loss = N_neg / N_pos.
- **Optimisation:** AdamW with cosine LR schedule. Early stopping on validation ROC-AUC.
- **Inference:** test-time augmentation averaged over the 8 dihedral transforms.

### 6. Avoiding data leakage ([sfcnn/dataset.py](sfcnn/dataset.py))
Neighbouring cores share overlapping cutouts and the same cloud environment, so a random
train/test split **overestimates** accuracy. `split.mode` controls the split:
- `region` (default): whole maps are held out. This is the honest test of "does it work on a new cloud?".
- `spatial`: each map is cut into 300″ blocks, and each block goes to only one split. Training cores
  whose cutouts overlap a held-out core are dropped (the buffer).
- `random`: only for showing how much a naive split inflates the scores.

### 7. Evaluation ([sfcnn/evaluate.py](sfcnn/evaluate.py))
On the held-out test set the CNN is compared with three baselines:
- logistic regression and random forest on **hand-crafted morphology features**
  ([sfcnn/features.py](sfcnn/features.py)): peak S/N, aperture S/N, concentration, half-max area,
  axis ratio, background;
- peak S/N alone.

The decision threshold is tuned on the **validation** set (best F1), never on the test set.

Outputs in `runs/<run>/eval/`:
- `metrics.json`, `comparison.csv`
- `roc_pr.png`
- `confusion_calibration.png`
- `gallery_gradcam.png`: TP/TN/FP/FN examples with **Grad-CAM** saliency ([sfcnn/explain.py](sfcnn/explain.py))
- `test_predictions.csv`

### 8. Applying to new maps ([sfcnn/predict.py](sfcnn/predict.py))
- `classify` writes a per-core CSV of P(star-forming) and an overlay PNG.
- `probmap` writes a sliding-window P(star-forming) FITS image with valid WCS, for DS9 or CARTA.
  Treat it as illustrative: the model was trained on peak-centred cutouts, so values away
  from emission peaks are out of distribution.

## Results on the synthetic survey

These numbers come from 12 synthetic regions and 868 labelled cores, split by region
(3 test regions, 233 cores, 66 star-forming). They check that the pipeline works and are not a
scientific result.

| model | ROC-AUC | PR-AUC | accuracy | balanced acc. | F1 |
|---|---|---|---|---|---|
| **CNN (SFNet)** | **0.946** | 0.909 | 0.884 | 0.864 | 0.800 |
| Random forest (features) | 0.942 | 0.914 | 0.893 | 0.870 | 0.812 |
| Logistic regression (features) | 0.902 | 0.861 | 0.863 | 0.831 | 0.758 |
| Peak S/N only | 0.661 | 0.419 | 0.524 | 0.613 | 0.493 |

On synthetic data the CNN only matches a random forest on hand-made features. That is expected:
the simulator separates the two classes mainly through central concentration, which the
features measure directly. Peak brightness alone is a weak predictor. Whether the CNN adds
anything over classical morphology on **real** data is the interesting question for the thesis.

## Using real JCMT data

1. **Maps.** Get reduced SCUBA-2 850 µm maps from the JCMT Science Archive at CADC. Good sources
   are the public **JCMT Gould Belt Survey** maps of nearby clouds (Orion A/B, Perseus, Ophiuchus,
   Serpens, Aquila, Taurus, …). The **JCMT Transient Survey** and **SCOPE** maps also work. Prefer
   nearby clouds: at 400 pc a 96″ cutout is about 0.19 pc.
2. **YSO catalogues** (VizieR), for example:
   - Dunham et al. 2015, ApJS 220, 11: Spitzer c2d + Gould Belt YSOs with extinction-corrected classes;
   - Megeath et al. 2012, AJ 144, 192: Spitzer Orion YSOs;
   - Furlan et al. 2016, ApJS 224, 5: HOPS Orion protostars, including Herschel-detected Class 0.

   Export RA/Dec in degrees and the class column to CSV.
3. **Core catalogues** (optional but recommended): published GBS core catalogues for the cloud
   you use.
4. Copy [configs/real_manifest_example.yaml](configs/real_manifest_example.yaml), fill in paths and
   column names, then point `data.manifest` at it in a copy of `default.yaml`.
5. With fewer than about 5 maps, use `split.mode: spatial`. With more, use `region` and name the
   test clouds explicitly (`test_regions`), so every model is scored on the same clouds.

*Verify every citation and catalogue ID against ADS/VizieR before putting it in the thesis.*

## Limitations to discuss in the report
- **Label noise.** Spitzer-based catalogues miss some deeply embedded Class 0 sources, so a few
  "starless" labels are wrong. Herschel/HOPS catalogues reduce this. A protostar that only lies
  *in projection* near a core produces false positives.
- **Fixed angular cutout.** Clouds at different distances are seen at different physical scales.
  A possible extension is to scale `size_arcsec` with distance.
- **CO(3–2) contamination** of the 850 µm band by outflows can add structure around protostars.
- **Small samples.** Report the spread over several seeds and splits, not a single number.
- **Spatial filtering** by the SCUBA-2 map-maker removes extended emission (>~5′), which
  changes core profiles.

## Ideas for extensions
- Add 450 µm as a third channel (gives a temperature/β proxy). Do **not** add 70 µm: that would
  leak the label, since 70 µm detections are how protostars are identified.
- Compare `random`, `spatial` and `region` splits to quantify leakage. This makes a good results
  chapter.
- Measure accuracy as a function of S/N and of distance. The synthetic generator can control both.
- Train on Orion and test on Perseus/Ophiuchus to measure cross-cloud transfer.
- Add a three-class problem: starless / prestellar / protostellar.

## Project layout
```
sfcnn/
  fits_io.py     SCUBA-2 FITS loading (cube squeeze, VARIANCE, WCS)
  sources.py     peak finding / external core catalogues
  labels.py      YSO cross-match labelling
  cutouts.py     resampled cutouts + 2-channel normalisation
  dataset.py     dataset building, leakage-free splits, augmentation
  model.py       SFNet, TTA, checkpoint loading
  train.py       training loop
  evaluate.py    metrics, baselines, plots
  features.py    hand-crafted features for the baselines
  explain.py     Grad-CAM
  predict.py     classify new maps, probability maps
  synthetic.py   synthetic SCUBA-2 survey generator
  __main__.py    CLI
configs/         default config + real-data manifest template
tests/           pytest suite
```
