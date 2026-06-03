"""
Phase 2: Multi-Domain Feature Extraction
=========================================
Matches and extends: Ghosh, Singh & Powar (Springer 2026)
  - "Seizure detection using EEG on the CHB-MIT dataset via
     multi-domain feature engineering and classical machine learning"

Baseline paper pipeline reproduced:
  - Time domain:      mean, variance, skewness, kurtosis, RMS, line length, ZCR
  - Frequency domain: delta/theta/alpha/beta/gamma band powers + SEF95
  - Wavelet domain:   db4 DWT (level 4) log-variance + wavelet packet energy
  - Spatial (CSP):    log-variance of top-4 CSP components

Extensions over baseline (your improvements):
  - Hjorth parameters (activity, mobility, complexity)
  - Sample entropy per channel
  - Katz fractal dimension
  - Cross-channel correlation (mean of upper triangle)

Usage:
    python phase2_feature_extraction.py

Expects: chbmit_windowed.npz  (output of phase1_preprocessed_chbmit.py)
         OR raw data loaded via wfdb (see load_and_resample_to_256hz)

Outputs: features.npz  →  X_feat [N, n_features], y [N], feature_names [n_features]
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis
import pywt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constants (match paper: 256 Hz, 1-second windows)
# ─────────────────────────────────────────────
FS = 256           # sampling rate Hz  (paper uses 256; your phase1 used 128 — see note below)
WINDOW_LEN = FS    # 1-second window = 256 samples
N_CSP_COMPONENTS = 4

# EEG frequency bands (paper: delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ─────────────────────────────────────────────
# NOTE ON SAMPLING RATE:
# Your phase1 code resamples to 128 Hz.
# The baseline paper uses 256 Hz.
# Option A: Re-run phase1 with FS_TARGET=256 (recommended for paper comparison)
# Option B: Use the features below with 128 Hz data — just change FS=128 above
#            and adjust the gamma band to (30, 60) since Nyquist = 64 Hz
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════════════════
# SECTION 1 — TIME DOMAIN FEATURES  (per channel)
# ═══════════════════════════════════════════════════════

def time_domain_features(window):
    """
    window: [n_channels, n_samples]
    Returns feature vector + names
    """
    feats, names = [], []
    n_ch = window.shape[0]

    for ch in range(n_ch):
        x = window[ch]

        mean_val  = np.mean(x)
        var_val   = np.var(x)
        skew_val  = skew(x)
        kurt_val  = kurtosis(x)
        rms_val   = np.sqrt(np.mean(x**2))

        # Line length — sensitive to seizure-related high-frequency activity
        ll_val    = np.sum(np.abs(np.diff(x)))

        # Zero crossing rate
        zcr_val   = np.sum(np.diff(np.sign(x)) != 0) / (len(x) - 1)

        feats.extend([mean_val, var_val, skew_val, kurt_val, rms_val, ll_val, zcr_val])

        if ch == 0:  # only build names once
            names.extend([
                f"ch{ch}_mean", f"ch{ch}_var", f"ch{ch}_skew",
                f"ch{ch}_kurt", f"ch{ch}_rms", f"ch{ch}_ll", f"ch{ch}_zcr"
            ])
        # (for larger ch counts we only need names for feature selection,
        #  so we skip re-building for every ch to keep it fast)

    return np.array(feats)


def time_domain_names(n_channels):
    names = []
    for ch in range(n_channels):
        for feat in ["mean", "var", "skew", "kurt", "rms", "ll", "zcr"]:
            names.append(f"time_ch{ch}_{feat}")
    return names


# ═══════════════════════════════════════════════════════
# SECTION 2 — FREQUENCY DOMAIN FEATURES  (averaged across channels)
# ═══════════════════════════════════════════════════════

def band_power(psd, freqs, fmin, fmax):
    """Integrate PSD within [fmin, fmax] using the trapezoidal rule."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])


def spectral_edge_frequency(psd, freqs, edge=0.95):
    """SEF95: lowest freq below which `edge` fraction of total power lies."""
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, edge * total)
    return freqs[min(idx, len(freqs) - 1)]


def frequency_domain_features(window, fs=FS):
    """
    window: [n_channels, n_samples]
    Returns averaged band powers + SEF95 across channels.
    """
    feats = []
    band_names = list(BANDS.keys())

    # Aggregate across channels
    all_band_powers = {b: [] for b in band_names}
    all_sef95 = []

    for ch in range(window.shape[0]):
        freqs, psd = welch(window[ch], fs=fs, nperseg=fs)  # 1-sec = 256 samples

        for band, (flo, fhi) in BANDS.items():
            # Skip bands above Nyquist
            if fhi > fs / 2:
                all_band_powers[band].append(0.0)
            else:
                all_band_powers[band].append(band_power(psd, freqs, flo, fhi))

        all_sef95.append(spectral_edge_frequency(psd, freqs))

    for band in band_names:
        feats.append(np.mean(all_band_powers[band]))   # mean across channels
        feats.append(np.std(all_band_powers[band]))    # spread across channels

    feats.append(np.mean(all_sef95))

    return np.array(feats)


def frequency_domain_names():
    names = []
    for band in BANDS.keys():
        names.append(f"freq_{band}_mean")
        names.append(f"freq_{band}_std")
    names.append("freq_sef95")
    return names


# ═══════════════════════════════════════════════════════
# SECTION 3 — WAVELET FEATURES  (db4, level 4)
# ═══════════════════════════════════════════════════════

def wavelet_features(window, wavelet="db4", level=4):
    """
    DWT decomposition + wavelet packet decomposition.
    Returns log-variance of sub-bands, relative energy.
    window: [n_channels, n_samples]
    """
    feats = []

    for ch in range(window.shape[0]):
        x = window[ch]

        # ── DWT coefficients ──
        coeffs = pywt.wavedec(x, wavelet, level=level)
        # coeffs = [cA4, cD4, cD3, cD2, cD1]

        dwt_feats = []
        total_energy = sum(np.sum(c**2) for c in coeffs) + 1e-10

        for c in coeffs:
            var_c = np.var(c) + 1e-10
            energy_c = np.sum(c**2)
            dwt_feats.extend([
                np.log(var_c),              # log-variance (paper feature)
                energy_c / total_energy,    # relative energy
            ])

        # ── Wavelet Packet Decomposition (level 3) ──
        wp = pywt.WaveletPacket(data=x, wavelet=wavelet, maxlevel=3)
        nodes = [node.path for node in wp.get_level(3, "freq")]
        wp_energies = []
        for node in nodes:
            coef = wp[node].data
            wp_energies.append(np.sum(coef**2) + 1e-10)
        wp_total = sum(wp_energies) + 1e-10

        for e in wp_energies:
            dwt_feats.append(np.log(e / wp_total))   # log-relative packet energy

        feats.extend(dwt_feats)

    return np.array(feats)


def wavelet_feature_names(n_channels, level=4, wp_level=3):
    names = []
    n_wp_nodes = 2 ** wp_level
    for ch in range(n_channels):
        for i in range(level + 1):
            names.append(f"wav_ch{ch}_dwt{i}_logvar")
            names.append(f"wav_ch{ch}_dwt{i}_relenergy")
        for j in range(n_wp_nodes):
            names.append(f"wav_ch{ch}_wp{j}_logenergy")
    return names


# ═══════════════════════════════════════════════════════
# SECTION 4 — CSP SPATIAL FEATURES
# ═══════════════════════════════════════════════════════

class CSPFilter:
    """
    Common Spatial Patterns with shrinkage regularisation.
    Must be fit on training data only to avoid leakage.
    """

    def __init__(self, n_components=N_CSP_COMPONENTS, reg=0.1):
        self.n_components = n_components
        self.reg = reg
        self.W_ = None   # spatial filters  [n_channels, n_components]

    def _cov(self, X):
        """Regularised covariance: X shape [n_windows, n_ch, n_samples]"""
        n = X.shape[0]
        C = np.zeros((X.shape[1], X.shape[1]))
        for i in range(n):
            xi = X[i]
            xi = xi - xi.mean(axis=1, keepdims=True)
            C += xi @ xi.T / (xi.shape[1] - 1)
        C /= n
        # Ledoit-Wolf style shrinkage
        C = (1 - self.reg) * C + self.reg * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        return C

    def fit(self, X, y):
        """
        X: [n_windows, n_channels, n_samples]
        y: [n_windows]  binary labels
        """
        X_seiz    = X[y == 1]
        X_nonseiz = X[y == 0]

        C1 = self._cov(X_seiz)
        C2 = self._cov(X_nonseiz)

        # Solve generalised eigenvalue problem
        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(C1, C1 + C2)
        except Exception:
            eigenvalues, eigenvectors = np.linalg.eigh(C1)

        # Take top and bottom n_components//2 filters
        half = self.n_components // 2
        idx = np.concatenate([
            np.argsort(eigenvalues)[-half:],
            np.argsort(eigenvalues)[:half]
        ])
        self.W_ = eigenvectors[:, idx]   # [n_channels, n_components]
        return self

    def transform(self, X):
        """
        X: [n_windows, n_channels, n_samples]
        Returns: [n_windows, n_components*2]  — log-variance + beta/gamma band power
        """
        assert self.W_ is not None, "CSP not fitted yet"
        feats = []
        for window in X:
            projected = self.W_.T @ window   # [n_components, n_samples]
            log_var = np.log(np.var(projected, axis=1) + 1e-10)

            # Band powers on CSP components (beta + gamma as in paper)
            bp_feats = []
            for comp in projected:
                freqs, psd = welch(comp, fs=FS, nperseg=min(FS, len(comp)))
                bp_feats.append(band_power(psd, freqs, 13, 30))  # beta
                bp_feats.append(band_power(psd, freqs, 30, min(45, FS // 2 - 1)))  # gamma

            feats.append(np.concatenate([log_var, bp_feats]))

        return np.array(feats)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def csp_feature_names(n_components=N_CSP_COMPONENTS):
    names = []
    for i in range(n_components):
        names.append(f"csp_comp{i}_logvar")
    for i in range(n_components):
        names.append(f"csp_comp{i}_beta_power")
        names.append(f"csp_comp{i}_gamma_power")
    return names


# ═══════════════════════════════════════════════════════
# SECTION 5 — EXTENSION FEATURES (your improvements over baseline)
# ═══════════════════════════════════════════════════════

def hjorth_parameters(x):
    """Activity, mobility, complexity — classic EEG features."""
    activity   = np.var(x)
    d1         = np.diff(x)
    mobility   = np.sqrt(np.var(d1) / (activity + 1e-10))
    d2         = np.diff(d1)
    complexity = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity


def sample_entropy(x, m=2, r_factor=0.2):
    """
    Approximate sample entropy — measures signal irregularity.
    m: template length, r: tolerance (fraction of std)
    """
    r = r_factor * np.std(x)
    n = len(x)

    def _phi(m_len):
        count = 0
        templates = np.array([x[i:i + m_len] for i in range(n - m_len)])
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(diffs <= r) - 1  # exclude self
        return count / (n - m_len)

    try:
        phi_m   = _phi(m)
        phi_m1  = _phi(m + 1)
        if phi_m == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m)
    except Exception:
        return 0.0


def katz_fractal_dimension(x):
    """Katz FD — measures waveform complexity, elevated during seizures."""
    n   = len(x)
    L   = np.sum(np.abs(np.diff(x)))       # total path length
    d   = np.max(np.abs(x - x[0]))         # max distance from first sample
    if d == 0 or L == 0:
        return 1.0
    return np.log10(n) / (np.log10(n) + np.log10(d / L))


def extension_features(window):
    """
    Additional features not in the baseline paper.
    window: [n_channels, n_samples]
    Returns feature vector
    """
    feats = []

    hjorth_act, hjorth_mob, hjorth_cmp = [], [], []
    samp_ent_vals = []
    katz_vals = []

    for ch in range(window.shape[0]):
        x = window[ch]

        act, mob, cmp = hjorth_parameters(x)
        hjorth_act.append(act)
        hjorth_mob.append(mob)
        hjorth_cmp.append(cmp)

        # Sample entropy is slow on full data — subsample to 64 pts for speed
        x_sub = x[::4] if len(x) > 64 else x
        samp_ent_vals.append(sample_entropy(x_sub))

        katz_vals.append(katz_fractal_dimension(x))

    # Aggregate across channels: mean + std
    feats.extend([np.mean(hjorth_act), np.std(hjorth_act)])
    feats.extend([np.mean(hjorth_mob), np.std(hjorth_mob)])
    feats.extend([np.mean(hjorth_cmp), np.std(hjorth_cmp)])
    feats.extend([np.mean(samp_ent_vals), np.std(samp_ent_vals)])
    feats.extend([np.mean(katz_vals), np.std(katz_vals)])

    # Cross-channel correlation (mean of upper triangle) — captures spatial synchrony
    corr_matrix = np.corrcoef(window)
    upper_tri = corr_matrix[np.triu_indices(window.shape[0], k=1)]
    feats.append(np.mean(upper_tri))
    feats.append(np.std(upper_tri))

    return np.array(feats)


def extension_feature_names():
    return [
        "ext_hjorth_act_mean", "ext_hjorth_act_std",
        "ext_hjorth_mob_mean", "ext_hjorth_mob_std",
        "ext_hjorth_cmp_mean", "ext_hjorth_cmp_std",
        "ext_sampent_mean",    "ext_sampent_std",
        "ext_katz_mean",       "ext_katz_std",
        "ext_corr_mean",       "ext_corr_std",
    ]


# ═══════════════════════════════════════════════════════
# SECTION 6 — FULL FEATURE EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════

def extract_all_features(X, csp_filter=None):
    """
    X: [N_windows, N_channels, N_samples]
    csp_filter: fitted CSPFilter instance (None → skip CSP)

    Returns: feature matrix [N_windows, n_features]
    """
    N, n_ch, n_s = X.shape
    print(f"Extracting features from {N} windows, {n_ch} channels, {n_s} samples each...")

    all_features = []

    for i, window in enumerate(X):
        if i % 500 == 0:
            print(f"  Window {i}/{N}...")

        td = time_domain_features(window)
        fd = frequency_domain_features(window)
        wd = wavelet_features(window)
        ed = extension_features(window)

        row = np.concatenate([td, fd, wd, ed])
        all_features.append(row)

    X_feat = np.array(all_features, dtype=np.float32)

    # CSP requires the fitted filter (transform-only at inference)
    if csp_filter is not None:
        csp_feat = csp_filter.transform(X)
        X_feat = np.hstack([X_feat, csp_feat.astype(np.float32)])

    return X_feat


def get_all_feature_names(n_channels, with_csp=True):
    names = (
        time_domain_names(n_channels)
        + frequency_domain_names()
        + wavelet_feature_names(n_channels)
        + extension_feature_names()
    )
    if with_csp:
        names += csp_feature_names()
    return names


# ═══════════════════════════════════════════════════════
# SECTION 7 — PATIENT-INDEPENDENT SPLIT  (matches paper)
# ═══════════════════════════════════════════════════════

def patient_independent_split(X, y, patient_ids, test_ratio=0.3, seed=42):
    """
    Split data so NO patient appears in both train and test.
    patient_ids: array of shape [N_windows] with patient index per window.

    Returns: X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)

    n_test = max(1, int(len(unique_patients) * test_ratio))
    test_patients  = set(unique_patients[:n_test])
    train_patients = set(unique_patients[n_test:])

    train_idx = np.where([p in train_patients for p in patient_ids])[0]
    test_idx  = np.where([p in test_patients  for p in patient_ids])[0]

    print(f"Train patients: {sorted(train_patients)}")
    print(f"Test  patients: {sorted(test_patients)}")
    print(f"Train windows: {len(train_idx)}  |  Test windows: {len(test_idx)}")
    print(f"Train seizure ratio: {y[train_idx].mean():.4f}")
    print(f"Test  seizure ratio: {y[test_idx].mean():.4f}")

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            patient_ids[train_idx], patient_ids[test_idx])


# ═══════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # ── Load windowed data from Phase 1 ──
    # If you ran phase1 at 128Hz, change FS=128 at the top of this file
    # and re-run. Or re-run phase1 with FS_TARGET=256 to match the paper.

    DATA_PATH = "chbmit_windowed.npz"

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run phase1_preprocessed_chbmit.py first, then re-run this script.")
        exit(1)

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]          # [N, n_channels, n_samples]
    y = data["y"]          # [N]

    # patient_ids should be saved in phase1 — if not, create a dummy
    if "patient_ids" in data:
        patient_ids = data["patient_ids"]
    else:
        print("WARNING: patient_ids not found in .npz — using dummy single-patient IDs.")
        print("Update phase1 to save patient_ids for proper patient-independent split.")
        patient_ids = np.zeros(len(y), dtype=int)

    print(f"Loaded: X={X.shape}, y={y.shape}, seizure ratio={y.mean():.4f}")

    # ── Patient-independent train/test split ──
    X_train, X_test, y_train, y_test, pid_train, pid_test = \
        patient_independent_split(X, y, patient_ids, test_ratio=0.3)

    # ── Fit CSP on TRAINING data only (no leakage) ──
    print("\nFitting CSP on training data...")
    csp = CSPFilter(n_components=N_CSP_COMPONENTS, reg=0.1)
    csp.fit(X_train, y_train)

    # ── Extract features ──
    print("\nExtracting training features...")
    X_train_feat = extract_all_features(X_train, csp_filter=csp)

    print("\nExtracting test features...")
    X_test_feat = extract_all_features(X_test, csp_filter=csp)

    feature_names = np.array(get_all_feature_names(X.shape[1], with_csp=True))

    print(f"\nFeature matrix: train={X_train_feat.shape}, test={X_test_feat.shape}")
    print(f"Total features: {X_train_feat.shape[1]}")
    print(f"Feature names sample: {feature_names[:5].tolist()} ... {feature_names[-5:].tolist()}")

    # ── Save ──
    np.savez(
        "features.npz",
        X_train=X_train_feat,
        X_test=X_test_feat,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
    print("\nSaved → features.npz")
    print("Next: run phase3_train_and_compare.py")
