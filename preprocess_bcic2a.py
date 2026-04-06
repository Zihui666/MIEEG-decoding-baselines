from pathlib import Path
import argparse
import numpy as np
import mne
from scipy.io import loadmat

# Strict-ish TCANet reproduction for BCIC IV-2a preprocessing.
# Matches the official repo on the parts that affect the final tensors:
# - no extra band-pass filtering
# - remove EOG channels
# - 4 s epochs using tmin=0, tmax=3.996
# - train set from 769/770/771/772 in T.gdf
# - test set from 783 in E.gdf
# - labels for E come from A0xE.mat
#
# Output labels are 0-based so they plug into PyTorch directly.

TRAIN_EVENT_ID = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3,
}

TEST_EVENT_ID = {
    "783": 0,  # unknown cue marker in evaluation file; true labels come from .mat
}


def load_raw_gdf(gdf_path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)


def load_eval_labels(label_path: Path) -> np.ndarray:
    mat = loadmat(label_path)
    if "classlabel" not in mat:
        raise KeyError(f"'classlabel' not found in {label_path}")
    y = mat["classlabel"].squeeze().astype(np.int64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D labels in {label_path}, got shape {y.shape}")
    if y.min() == 1 and y.max() == 4:
        y = y - 1
    return y


def get_eeg_picks(raw: mne.io.BaseRaw):
    # Match official script: mark EOG as bads and pick eeg only.
    for ch_name in ["EOG-left", "EOG-central", "EOG-right"]:
        if ch_name in raw.ch_names and ch_name not in raw.info["bads"]:
            raw.info["bads"].append(ch_name)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
    return picks


def extract_epochs(raw: mne.io.BaseRaw, event_id: dict, tmin: float = 0.0, tmax: float = 3.996):
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    selected_ids = [event_dict[k] for k in event_id.keys() if k in event_dict]
    if not selected_ids:
        raise RuntimeError(f"None of the requested event codes {list(event_id.keys())} were found. Available: {sorted(event_dict.keys())}")
    selected_events = events[np.isin(events[:, 2], selected_ids)]
    picks = get_eeg_picks(raw)
    epochs = mne.Epochs(
        raw,
        selected_events,
        {k: event_dict[k] for k in event_id.keys() if k in event_dict},
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        baseline=None,
        verbose=False,
    )
    return epochs


def save_npz(path: Path, X: np.ndarray, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


def preprocess_subject(raw_dir: Path, label_dir: Path, out_dir: Path, subject: str):
    train_gdf = raw_dir / f"{subject}T.gdf"
    eval_gdf = raw_dir / f"{subject}E.gdf"
    eval_mat = label_dir / f"{subject}E.mat"

    raw_train = load_raw_gdf(train_gdf)
    epochs_train = extract_epochs(raw_train, TRAIN_EVENT_ID, tmin=0.0, tmax=3.996)
    X_train = epochs_train.get_data(copy=True).astype(np.float32)

    # Convert event ids to 0..3 according to TRAIN_EVENT_ID mapping.
    inv_train_map = {v: TRAIN_EVENT_ID[k] for k, v in epochs_train.event_id.items()}
    y_train = np.array([inv_train_map[eid] for eid in epochs_train.events[:, -1]], dtype=np.int64)

    raw_eval = load_raw_gdf(eval_gdf)
    epochs_eval = extract_epochs(raw_eval, TEST_EVENT_ID, tmin=0.0, tmax=3.996)
    X_test = epochs_eval.get_data(copy=True).astype(np.float32)
    y_test = load_eval_labels(eval_mat)

    if len(X_test) != len(y_test):
        raise ValueError(f"{subject}: evaluation epochs {len(X_test)} != labels {len(y_test)}")

    save_npz(out_dir / f"{subject}_train.npz", X_train, y_train)
    save_npz(out_dir / f"{subject}_test.npz", X_test, y_test)

    print(
        f"{subject}: train={X_train.shape}, test={X_test.shape}, "
        f"train_classes={np.unique(y_train)}, test_classes={np.unique(y_test)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Strict TCANet-style preprocessing for BCIC IV-2a")
    parser.add_argument("--raw_dir", type=str, required=True, help="Folder containing A01T.gdf ... A09E.gdf")
    parser.add_argument("--label_dir", type=str, required=True, help="Folder containing A01E.mat ... A09E.mat")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for A01_train.npz ...")
    parser.add_argument("--subjects", type=str, default="A01,A02,A03,A04,A05,A06,A07,A08,A09")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    for subject in subjects:
        preprocess_subject(raw_dir, label_dir, out_dir, subject)

    print("Done.")


if __name__ == "__main__":
    main()
