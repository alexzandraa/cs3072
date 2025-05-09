import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, iirnotch

# CONFIGURATION 
EEG_DIR       = "participant_data/eeg"
DECISIONS_DIR = "participant_data/decisions"
OUT_CSV       = "eeg_bandpower_preprocessed.csv"
SAMPLE_RATE   = 256.0  # Hz

# frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# notch filter to remove powerline (50 Hz)
def notch_filter(data, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

# band-pass filter
def bandpass_filter(data, fs, low=1.0, high=45.0, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def load_eeg(participant, stage):
    path = os.path.join(EEG_DIR, f"eeg_data_{participant}_Stage {stage}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, engine='python', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    if 'timestamp' not in df.columns:
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    data_cols = [c for c in df.columns if c.startswith('data_')]
    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=data_cols).reset_index(drop=True)
    return df[['timestamp'] + data_cols]

def load_decisions(participant):
    path = os.path.join(DECISIONS_DIR, f"user_decisions_{participant}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns or 'stage' not in df.columns:
        raise ValueError(f"Missing columns in {path}")
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp']).reset_index(drop=True)

def extract_epochs(eeg_df, decisions_df, stage, pre=1.0, post=2.0):
    subset = decisions_df[decisions_df['stage'] == f"Stage {stage}"]
    epochs = []
    for ts in subset['timestamp']:
        start, end = ts - pre, ts + post
        mask = (eeg_df['timestamp'] >= start) & (eeg_df['timestamp'] <= end)
        data = eeg_df.loc[mask, eeg_df.columns.str.startswith('data_')].values
        if data.shape[0] < int((pre + post) * SAMPLE_RATE * 0.5):
            continue
        epochs.append(data)
    return epochs

def compute_bandpower(epoch, fs):
    bp = {}
    for ch in range(epoch.shape[1]):
        sig = notch_filter(epoch[:, ch], fs)
        sig = bandpass_filter(sig, fs)
        nperseg = min(int(fs*2), len(sig))
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        for name, (lo, hi) in BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            bp[f"ch{ch}_{name}"] = np.trapezoid(psd[idx], freqs[idx])
    return bp

def main():
    out_rows = []
    for p in range(1, 11):
        try:
            decisions = load_decisions(p)
        except Exception as e:
            print(f" No decisions for P{p}: {e}")
            continue
        for s in range(1, 5):
            try:
                eeg_df = load_eeg(p, s)
            except Exception as e:
                print(f" Error loading P{p} S{s}: {e}")
                continue
            epochs = extract_epochs(eeg_df, decisions, s)
            print(f"P{p} S{s}: {len(epochs)} epochs")
            for i, epoch in enumerate(epochs):
                bp = compute_bandpower(epoch, SAMPLE_RATE)

                # Add all metadata
                bp.update({
                    'participant': p,
                    'stage': s,
                    'epoch': i,
                    'bias': "Biased" if s in [1, 2] else "Fair",
                    'transparency': "Yes" if s in [2, 4] else "No"
                })

                # Add label from decisions
                stage_subset = decisions[decisions['stage'] == f"Stage {s}"]
                if i < len(stage_subset):
                    bp["label"] = stage_subset.iloc[i].get("decision", "unknown")

                out_rows.append(bp)

    if out_rows:
        pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)
        print(f"Saved {OUT_CSV}")
    else:
        print(" No epochs extracted; output CSV is empty.")

if __name__ == '__main__':
    main()
