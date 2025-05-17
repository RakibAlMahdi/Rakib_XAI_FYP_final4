import os, random
import torch, torchaudio
from torch.utils.data import Dataset

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
            info = torchaudio.info(wav)
            est_len = int(info.num_frames / info.sample_rate * self.resample_hz)
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
        return x.float(), torch.tensor(lab, dtype=torch.float32), pid 