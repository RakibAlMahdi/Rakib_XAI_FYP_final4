import os
import glob
import pandas as pd

SITES = ["AV", "MV", "PV", "TV"]

def load_metadata(csv_path):
    """Load CirCor CSV and return a DataFrame with essential columns.

    The returned DataFrame has columns:
        Patient ID   : int
        label        : int   (1 = Abnormal, 0 = Normal)
        murmur       : int   (0 Absent, 1 Unknown, 2 Present)
    """
    df = pd.read_csv(csv_path)

    # Map clinical outcome to binary label
    df["label"] = (df["Outcome"].str.strip().str.lower() == "abnormal").astype(int)

    # Map murmur text to ordinal label
    murmur_map = {"absent": 0, "unknown": 1, "present": 2}
    df["murmur"] = df["Murmur"].str.strip().str.lower().map(murmur_map)

    return df[["Patient ID", "label", "murmur"]]


def wav_paths_for_patient(patient_id: int, wav_root: str):
    """Return list of existing WAV file paths for a given patient ID."""
    pattern = os.path.join(wav_root, f"{patient_id}_*.wav")
    return sorted(glob.glob(pattern))


# ------------------------------------------------------------------
# PhysioNet 2016 helper
# ------------------------------------------------------------------

def load_physionet2016_metadata(csv_path: str):
    """Load PhysioNet 2016 Online_Appendix_training_set.csv.

    Returns DataFrame with columns:
        wav   : filename with .wav extension
        label : 0 normal, 1 abnormal
    """
    df = pd.read_csv(csv_path)
    wav_col = df.columns[0]  # 'Challenge record name'
    label_col = df.columns[4]  # Outcome/Class (-1=normal 1=abnormal)
    md = pd.DataFrame()
    md["wav"] = df[wav_col].astype(str) + ".wav"
    md["label"] = (df[label_col] == 1).astype(int)
    return md 