import wfdb
import numpy as np
from scipy.signal import resample, butter, filtfilt

# Sampling parameters
FS_TARGET = 128  # 1‑s window = 128 samples
WINDOW_LEN = FS_TARGET  # 1‑second window
STEP       = FS_TARGET  # non‑overlapping (or use FS_TARGET//2 for 50% overlap)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def preprocess_channel(ch, fs_old, fs_new=FS_TARGET):
    """Resample + filter + normalize one channel."""
    # 1. Resample
    num_new = int(len(ch) * fs_new / fs_old)
    ch = resample(ch, num_new)
    # 2. Band‑pass 0.5–30 Hz
    b, a = butter_bandpass(0.5, 30, fs_new, order=4)
    ch = filtfilt(b, a, ch)
    # 3. Z‑score
    ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)
    return ch


def load_chbmit_record(record_path, ann_ext="seizures"):
    """
    Load one CHB‑MIT record (e.g., "chb01/chb01_01") and return:
        X: [N_windows, N_channels, 128]
        y: [N_windows] (binary labels)
    """
    # Load EEG signal
    r = wfdb.rdrecord(record_path, physical=False)  # ensure this path is correct
    sig = r.p_signal.T  # shape: [channels, samples]
    fs_old = r.fs

    # Preprocess all channels
    n_ch, n_s = sig.shape
    n_s_new = int(n_s * FS_TARGET / fs_old)
    sig_proc = np.zeros((n_ch, n_s_new))

    for i in range(n_ch):
        sig_proc[i] = preprocess_channel(sig[i], fs_old, fs_new=FS_TARGET)

    # Load seizure annotations
    try:
        ann = wfdb.rdann(record_path, ann_ext)
        seizure_starts_old = ann.sample[::2]  # one start per seizure
        seizure_ends_old   = ann.sample[1::2]
    except:
        # No seizures in this file → all windows are 0
        seizure_starts_old = np.array([])
        seizure_ends_old   = np.array([])

    # Convert seizure times to new sampling rate
    seizure_starts = (seizure_starts_old * FS_TARGET / fs_old).astype(int)
    seizure_ends   = (seizure_ends_old   * FS_TARGET / fs_old).astype(int)

    # Create 1‑second windows
    X = []  # [N_windows, n_ch, 128]
    y = []  # [N_windows]

    n_samples = sig_proc.shape[1]

    for i in range(0, n_samples - WINDOW_LEN + 1, STEP):
        window = sig_proc[:, i:i + WINDOW_LEN]  # [n_ch, 128]
        start_sec = i / FS_TARGET
        end_sec   = (i + WINDOW_LEN) / FS_TARGET

        label = 0  # non‑seizure by default
        for st, en in zip(seizure_starts, seizure_ends):
            # if this window overlaps with ANY seizure
            if end_sec > st / FS_TARGET and start_sec < en / FS_TARGET:
                label = 1
                break

        X.append(window)
        y.append(label)

    X = np.array(X, dtype=np.float32)  # shape: [N_samples, N_channels, 128]
    y = np.array(y, dtype=np.int32)    # shape: [N_samples]

    return X, y


# Example usage
if __name__ == "__main__":
    # Adapt path according to where you downloaded CHB‑MIT
    record_name = "chb01_01"  # e.g., path without .edf (wfdb auto‑detects)
    X, y = load_chbmit_record(record_name, ann_ext="seizures")

    print("X shape:", X.shape)   # e.g., (N_samples, 23, 128)
    print("y shape:", y.shape)   # e.g., (N_samples,)
    print("Seizure ratio:", np.mean(y))

    # Optionally save
    np.savez("chbmit_windowed.npz", X=X, y=y)
