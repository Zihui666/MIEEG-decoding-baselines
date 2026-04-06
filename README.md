# BCIC IV-2a PyTorch Baselines

PyTorch reproduction and benchmark scripts for motor imagery EEG decoding on **BCI Competition IV 2a**.

This repository provides a unified pipeline for **preprocessing**, **training**, **evaluation**, and **result export** for several common baselines, including:

- **EEGNet**
- **DeepConvNet**
- **TCANet**
- **EEG-Conformer**

The repository is intended for:

- reproducing a **strict BCIC IV-2a baseline protocol**
- comparing multiple EEG models under a **shared preprocessing and training pipeline**
- exporting per-subject and aggregate results in a clean, reusable format

## Important note on scope

This repository contains **PyTorch reproduction / adaptation scripts for BCIC IV-2a baselines**.

The **strict preprocessing and training protocol** is primarily adapted from the public **TCANet** implementation for BCIC IV-2a, while the **unified benchmark framework**, **CLI interface**, and **result-export logic** are reorganized and extended in this repository.

In other words:

- the **protocol** is TCANet-style
- the repository is **not limited to TCANet**
- additional backbones such as **EEGNet**, **DeepConvNet**, and **EEG-Conformer** are included under the same benchmark framework

## What is implemented

### 1. BCIC IV-2a preprocessing

The preprocessing script follows a TCANet-style BCIC IV-2a setup:

- no extra band-pass filtering
- EOG channels removed
- 4-second epochs with `tmin=0.0`, `tmax=3.996`
- training samples extracted from `A0xT.gdf` using event codes `769 / 770 / 771 / 772`
- evaluation samples extracted from `A0xE.gdf` using event code `783`
- evaluation labels loaded from `A0xE.mat`
- outputs saved as compressed `.npz` files for direct PyTorch training

### 2. Unified multi-model training

The training script provides one shared training pipeline for the following models:

- `eegnet`
- `deepconvnet`
- `tcanet`
- `eegconformer`

The current framework includes:

- unified command-line interface
- subject-wise training on BCIC IV-2a
- train/validation split derived from the training session
- optional TCANet-style segment-recombination augmentation (`interaug`)
- standardized result export

### 3. Result export

For each model, the training script saves:

- per-subject training history (`*_history.csv`)
- per-subject predictions (`*_predictions.csv`)
- per-subject checkpoints (`*.pt`)
- subject summary table (`summary.csv`)
- aggregate performance summary (`aggregate.json`)

## Repository structure

```text
.
├── preprocess_bcic2a.py
├── train_mi_baselines.py
├── README.md
├── requirements.txt
├── LICENSE
```

## Dataset layout

Expected raw data layout:

```text
D:\zhuomian\EEG_MI\
├── BCICIV_2a_gdf/
│   ├── A01T.gdf
│   ├── A01E.gdf
│   └── ...
└── true_labels/
    ├── A01E.mat
    └── ...
```

## Preprocessing

Example:

```bash
python preprocess_bcic2a.py --raw_dir "D:\zhuomian\EEG_MI\BCICIV_2a_gdf" --label_dir "D:\zhuomian\EEG_MI\true_labels" --out_dir "D:\zhuomian\EEG_MI\processed_bcic2a"
```

This will generate files such as:

```text
A01_train.npz
A01_test.npz
...
A09_train.npz
A09_test.npz
```

## Training

### TCANet

```bash
python train_mi_baselines.py --data_dir "D:\zhuomian\EEG_MI\processed_bcic2a" --model tcanet --subjects A01,A02,A03,A04,A05,A06,A07,A08,A09 --batch_size 72 --epochs 1000 --lr 0.001 --beta1 0.5 --beta2 0.999 --number_aug 1 --number_seg 8 --save_dir "D:\zhuomian\EEG_MI\results_strict"
```

### EEGNet

```bash
python train_mi_baselines.py --data_dir "D:\zhuomian\EEG_MI\processed_bcic2a" --model eegnet --subjects A01,A02,A03,A04,A05,A06,A07,A08,A09 --batch_size 72 --epochs 1000 --lr 0.001 --beta1 0.5 --beta2 0.999 --number_aug 1 --number_seg 8 --save_dir "D:\zhuomian\EEG_MI\results_strict"
```

### DeepConvNet

```bash
python train_mi_baselines.py --data_dir "D:\zhuomian\EEG_MI\processed_bcic2a" --model deepconvnet --subjects A01,A02,A03,A04,A05,A06,A07,A08,A09 --batch_size 72 --epochs 1000 --lr 0.001 --beta1 0.5 --beta2 0.999 --number_aug 1 --number_seg 8 --save_dir "D:\zhuomian\EEG_MI\results_strict"
```

### EEG-Conformer

```bash
python train_mi_baselines.py --data_dir "D:\zhuomian\EEG_MI\processed_bcic2a" --model eegconformer --subjects A01,A02,A03,A04,A05,A06,A07,A08,A09 --batch_size 72 --epochs 1000 --lr 0.001 --beta1 0.5 --beta2 0.999 --number_aug 1 --number_seg 8 --save_dir "D:\zhuomian\EEG_MI\results_strict"
```

## Output files

For each model, outputs are saved to:

```text
<save_dir>/<model>/
```

Example:

```text
D:\zhuomian\EEG_MI\results_strict\tcanet\
├── summary.csv
├── aggregate.json
├── A01_history.csv
├── A01_predictions.csv
├── tcanet_A01.pt
└── ...
```

## Attribution

This repository includes a **TCANet-style strict preprocessing/training protocol** for BCIC IV-2a and extends it into a unified benchmark framework.

Please note:

- the **strict preprocessing and training protocol** is adapted from the public **TCANet** implementation
- the repository additionally includes **EEGNet**, **DeepConvNet**, and **EEG-Conformer** under the same benchmark pipeline
- the **framework structure**, **CLI**, and **result-export code** are reorganized and extended in this repository

If you use this repository, please also cite the original papers and official repositories for the included model families.

## References

- **TCANet**  
  Zhao, W., Lu, H., Zhang, B. et al. *TCANet: a temporal convolutional attention network for motor imagery EEG decoding.*

- **EEGNet**  
  Lawhern, V. J. et al. *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.*

- **DeepConvNet / ShallowConvNet**  
  Schirrmeister, R. T. et al. *Deep learning with convolutional neural networks for EEG decoding and visualization.*

- **EEG-Conformer**  
  Song, Y. et al. *EEG Conformer: convolutional transformer for EEG decoding and visualization.*

## Notes

- This repository is intended for **benchmarking and reproduction**, not for claiming original authorship of the underlying published model architectures.
- Reported results may differ from those in original papers because model performance depends on preprocessing details, augmentation choices, random seeds, and training protocol.
- PyTorch with CUDA should be installed separately according to your local CUDA version.
