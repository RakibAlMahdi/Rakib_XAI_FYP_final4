import os, random, json, hashlib
import torch, torchaudio
from torch.utils.data import Dataset

CACHE_PATH = os.path.expanduser("~/.pcg_len_cache.json")
_LEN_CACHE = {}
if os.path.exists(CACHE_PATH):
    try:
        _LEN_CACHE = json.load(open(CACHE_PATH))
    except Exception:
        _LEN_CACHE = {}

class SegmentPCGDataset(Dataset):
    """Generic PCG dataset that outputs fixed-length waveform segments.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'wav_path', 'label', 'patient_id'.
    resample_hz : int
        Target sampling rate.
    segment_sec : int
        Length of each segment in seconds.
    augment : bool
        Whether to apply on-the-fly augmentation (light noise).
    """
    def __init__(self, df, *, resample_hz: int = 1000, segment_sec: int = 10, augment=False):
        self.df = df.reset_index(drop=True)
        self.resample_hz = resample_hz
        self.seg_len = segment_sec * resample_hz
        self.augment = augment
        # Build list of tuples (wav_path, label, seg_idx, patient_id)
        self.items = []
        for wav, lab, pid in zip(df["wav_path"], df["label"], df["patient_id"]):
            key = hashlib.md5(wav.encode()).hexdigest()
            if key not in _LEN_CACHE:
                info = torchaudio.info(wav)
                _LEN_CACHE[key] = int(info.num_frames / info.sample_rate * self.resample_hz)
            est_len = _LEN_CACHE[key]
            n_seg = max(1, est_len // self.seg_len)
            for sidx in range(n_seg):
                self.items.append((wav, lab, sidx, pid))

    def __len__(self):
        return len(self.items)

    def _preprocess(self, waveform, orig_sr):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if orig_sr != self.resample_hz:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.resample_hz)
        waveform = torchaudio.functional.highpass_biquad(waveform, self.resample_hz, 25)
        waveform = torchaudio.functional.lowpass_biquad(waveform, self.resample_hz, 400)
        if waveform.shape[1] < self.seg_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.seg_len - waveform.shape[1]))
        return waveform[:, :self.seg_len]

    def __getitem__(self, idx):
        wav_path, lab, sidx, pid = self.items[idx]
        wav, sr = torchaudio.load(wav_path)
        x = self._preprocess(wav, sr)
        # slice to requested segment
        start = sidx * self.seg_len
        end = start + self.seg_len
        if start >= x.shape[1]:
            start = 0; end = self.seg_len
        x = x[:, start:end]
        if self.augment:
            # light Gaussian noise
            if random.random() < 0.3:
                x = x + torch.randn_like(x) * 0.001
            # time-mask (SpecAugment-style) – zero 0.1-0.5 s window
            if random.random() < 0.3:
                L = random.randint(int(0.1*self.resample_hz), int(0.5*self.resample_hz))
                s = random.randint(0, x.shape[1]-L)
                x[:, s:s+L] = 0
            # ±10% time-stretch (speed perturbation)
            if random.random() < 0.3:
                rate = random.uniform(0.9, 1.1)
                L = x.shape[1]
                x = torch.nn.functional.interpolate(x.unsqueeze(0), scale_factor=rate,
                                                   mode="linear", align_corners=False).squeeze(0)
                if x.shape[1] < L:
                    x = torch.nn.functional.pad(x, (0, L - x.shape[1]))
                else:
                    x = x[:, :L]
        return x.float(), torch.tensor(lab, dtype=torch.float32), pid

# save cache at import exit only once
import atexit

@atexit.register
def _save_len_cache():
    if _LEN_CACHE:
        try:
            json.dump(_LEN_CACHE, open(CACHE_PATH, 'w'))
        except Exception:
            pass 