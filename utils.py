import pandas as pd
import h5py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, auc, roc_curve
from scipy.stats import linregress

def extract_patient_id_from_filename(filepath):
    base = os.path.basename(filepath)
    # Remove the extension if it ends with .h5
    if base.endswith('.h5'):
        base = base[:-3]
    parts = base.split('-')
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filepath}")
    return '-'.join(parts[:3])

def create_patient_dict(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Disease Free (Months)'])
    df['Disease Free (Months)'] = df['Disease Free (Months)'].astype(float)
    def parse_status(status):
        try:
            return int(status.split(':')[0].strip())
        except Exception as e:
            raise ValueError(f"Error parsing status '{status}': {e}")
    df['Disease Free Status'] = df['Disease Free Status'].apply(parse_status)

    patient_dict = {}
    for _, row in df.iterrows():
        patient_id = row['Patient ID']
        time = row['Disease Free (Months)']
        event = row['Disease Free Status']
        score = row['Leibovich Score']
        patient_dict[patient_id] = (time, event, score)
    
    return patient_dict

def generate_labels(features_dir, label_dict):
    features_list = []
    times_list = []
    events_list = []
    scores_list = []
    skipped_files = 0
    selected_files = 0
    patient_ids = []
    
    for h5_file in os.listdir(features_dir):
        full_path = os.path.join(features_dir, h5_file)
        patient_id = extract_patient_id_from_filename(h5_file)
        if patient_id not in patient_ids:
            patient_ids.append(patient_id)
        if patient_id in label_dict:
            with h5py.File(full_path, 'r') as f:
                features_array = f['features'][()]                
            features_list.append(features_array)
            time, event, score = label_dict[patient_id]
            times_list.append(time)
            events_list.append(event)
            scores_list.append(score)
            selected_files += 1
            # time, event = label_dict[patient_id]
            # for patch_feat in features_array:
            #     features_list.append(patch_feat)
            #     times_list.append(time)
            #     events_list.append(event)
            #     selected_files += 1
        else:
            skipped_files += 1

    print(f"Selected files: {selected_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Total patients processed: {len(patient_ids)}")
    print(patient_ids)

    # Convert the lists into tensors
    features_tensor = torch.tensor(np.stack(features_list, axis=0), dtype=torch.float32)
    times_tensor = torch.tensor(times_list, dtype=torch.float32)
    events_tensor = torch.tensor(events_list, dtype=torch.float32)
    scores_tensor = torch.tensor(scores_list, dtype=torch.float32) 

    return features_tensor, times_tensor, events_tensor, scores_tensor


def generate_ct_feature_vectors(
    features_dir: str,
    label_dict: dict
):
    """
    Load CT feature vectors (in .npy format) and the corresponding labels.

    Patient ID is taken as the part of the filename before the first '_'.
    """
    features_list = []
    times_list    = []
    events_list   = []
    scores_list   = []
    patient_ids   = []

    skipped = 0
    selected = 0

    for fn in sorted(os.listdir(features_dir)):
        if not fn.endswith('.npy'):
            continue

        # strip gz/npy extensions
        stem = fn
        if stem.endswith('.npy.gz'):
            stem = stem[:-7]
        elif stem.endswith('.npy'):
            stem = stem[:-4]

        # patient ID is everything before the first underscore
        patient_id = stem.split('_', 1)[0]

        if patient_id not in label_dict:
            skipped += 1
            continue

        vec = np.load(os.path.join(features_dir, fn))
        time, event, score = label_dict[patient_id]

        features_list.append(vec)
        times_list.append(time)
        events_list.append(event)
        scores_list.append(score)
        patient_ids.append(patient_id)
        selected += 1

    print(f"Selected files: {selected}")
    print(f"Skipped  files: {skipped}")
    print(f"Patient IDs:  {patient_ids}")

    if not features_list:
        raise ValueError("No feature vectors loaded — check your filenames vs. label_dict keys.")
    num_relapse = int(np.sum(events_list))
    print(f"Number of patients with a relapse indicator: {num_relapse} / {len(events_list)}")
    
    X = torch.from_numpy(np.stack(features_list, axis=0)).float()
    T = torch.tensor(times_list,  dtype=torch.float32)
    E = torch.tensor(events_list, dtype=torch.float32)
    S = torch.tensor(scores_list, dtype=torch.float32)

    return X, T, E, S

def generate_fusion_feature_vectors(
    ct_dir: str,
    wsi_dir: str,
    label_dict: dict
):
    """
    Return aligned CT and WSI feature tensors, plus one common
    (times, events, scores) tensor set for patients present in all three.
    WSI features are loaded *exactly* as in `generate_labels(...)`.
    """
    # 1) Build per‐patient CT map (as before)
    ct_map = {}
    for fn in os.listdir(ct_dir):
        if not fn.endswith('.npy'):
            continue
        pid = fn.split('_', 1)[0]
        ct_map[pid] = np.load(os.path.join(ct_dir, fn))

    # 2) Build per‐patient WSI map *copying generate_labels logic*
    wsi_map = {}
    skipped, selected = 0, 0
    patient_ids = []
    for h5_file in os.listdir(wsi_dir):
        full_path = os.path.join(wsi_dir, h5_file)
        pid = extract_patient_id_from_filename(h5_file)
        if pid not in patient_ids:
            patient_ids.append(pid)
        if pid in label_dict:
            with h5py.File(full_path, 'r') as f:
                features_array = f['features'][()]
            # exactly as generate_labels: append the *whole* vector
            wsi_map[pid] = features_array
            selected += 1
        else:
            skipped += 1

    print(f"Selected files: {selected}")
    print(f"Skipped files:  {skipped}")
    print(f"All patient IDs seen: {patient_ids}")

    # 3) Intersect patients
    common_pids = sorted(ct_map.keys() & wsi_map.keys() & label_dict.keys())
    if not common_pids:
        raise ValueError("No overlapping patients in CT, WSI, and labels!")

    # 4) Assemble aligned lists
    ct_feats, wsi_feats, times, events, scores = [], [], [], [], []
    for pid in common_pids:
        ct_feats.append(ct_map[pid])
        wsi_feats.append(wsi_map[pid])
        t, e, s = label_dict[pid]
        times.append(t)
        events.append(e)
        scores.append(s)

    # 5) Convert to tensors
    X_ct  = torch.tensor(np.stack(ct_feats,  axis=0), dtype=torch.float32)
    X_wsi = torch.tensor(np.stack(wsi_feats, axis=0), dtype=torch.float32)
    T     = torch.tensor(times,  dtype=torch.float32)
    E     = torch.tensor(events, dtype=torch.float32)
    S     = torch.tensor(scores, dtype=torch.float32)

    print(f"[fusion] {len(common_pids)} patients in common: {common_pids}")
    return X_ct, X_wsi, T, E, S, common_pids

def plot(pd_dir, hparams, epochs, train_cis, train_aucs, val_cis, val_aucs):
    os.makedirs(pd_dir, exist_ok=True)
    epochs = np.arange(1, len(train_cis) + 1)
    # C-index plot
    plt.figure()
    plt.plot(epochs, train_cis, label="Train C-index")
    plt.plot(epochs, val_cis,   label="Test C-index")
    plt.hlines(hparams['base_tr_ci'], 1, len(epochs), colors='red',   linestyles='dashed', label="Leibovich Train CI")
    plt.hlines(hparams['base_va_ci'], 1, len(epochs), colors='green', linestyles='dashed', label="Leibovich Test CI")
    plt.xlabel("Epoch"); plt.ylabel("C-index")
    plt.title(f"{hparams['modality']} Fold {hparams['fold']} C-index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pd_dir, f"{hparams['modality'].lower()}_fold_{hparams['fold']}_cindex.png"))
    
    plt.close()
    # AUROC plot
    plt.figure()
    plt.plot(epochs, train_aucs, label="Train AUROC5")
    plt.plot(epochs, val_aucs,   label="Test AUROC5")
    plt.hlines(hparams['base_tr_auroc5'], 1, len(epochs), colors='red',   linestyles='dashed', label="Leibovich Train AUROC5")
    plt.hlines(hparams['base_va_auroc5'], 1, len(epochs), colors='green', linestyles='dashed', label="Leibovich Test AUROC5")
    plt.xlabel("Epoch"); plt.ylabel("AUROC @ 5yr")
    plt.title(f"{hparams['modality']} Fold {hparams['fold']} AUROC @5yr")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pd_dir, f"{hparams['modality'].lower()}_fold_{hparams['fold']}_auroc.png"))
    
    plt.close()

def plot_average_metrics(
    modality, plot_dir, epochs,
    mean_train_cis, std_train_cis, mean_val_cis, std_val_cis,
    mean_train_aucs, std_train_aucs, mean_val_aucs, std_val_aucs,
    mean_base_ci, std_base_ci, mean_base_auc, std_base_auc,
    n_splits
):
    # 1) C-index
    plt.figure()
    plt.plot(epochs, mean_train_cis, label="Mean Train C-index")
    plt.fill_between(
        epochs,
        mean_train_cis - std_train_cis,
        mean_train_cis + std_train_cis,
        alpha=0.2
    )
    plt.plot(epochs, mean_val_cis, label="Mean Test C-index")
    plt.fill_between(
        epochs,
        mean_val_cis - std_val_cis,
        mean_val_cis + std_val_cis,
        alpha=0.2
    )
    plt.hlines(
        mean_base_ci,
        xmin=1,
        xmax=epochs[-1],
        colors="green",
        linestyles="--",
        label=f"Leibovich Mean CI ({mean_base_ci:.3f})"
    )
    plt.fill_between(
        [1, epochs[-1]],
        mean_base_ci - std_base_ci,
        mean_base_ci + std_base_ci,
        color="green",
        alpha=0.1,
        label=f"Leibovich ±1 SD ({std_base_ci:.3f})"
    )
    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    plt.title(f"{modality} Average C-index Across {n_splits} Folds")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{modality}_avg_cindex.png"))
    
    plt.close()

    # 2) AUROC @ horizon
    plt.figure()
    plt.plot(epochs, mean_train_aucs, label="Mean Train AUROC5")
    plt.fill_between(
        epochs,
        mean_train_aucs - std_train_aucs,
        mean_train_aucs + std_train_aucs,
        alpha=0.2
    )
    plt.plot(epochs, mean_val_aucs, label="Mean Test AUROC5")
    plt.fill_between(
        epochs,
        mean_val_aucs - std_val_aucs,
        mean_val_aucs + std_val_aucs,
        alpha=0.2
    )
    plt.hlines(
        mean_base_auc,
        xmin=1,
        xmax=epochs[-1],
        colors="green",
        linestyles="--",
        label=f"Leibovich Mean AUROC5 ({mean_base_auc:.3f})"
    )
    plt.fill_between(
        [1, epochs[-1]],
        mean_base_auc - std_base_auc,
        mean_base_auc + std_base_auc,
        color="green",
        alpha=0.1,
        label=f"Leibovich ±1 SD ({std_base_auc:.3f})"
    )
    plt.xlabel("Epoch")
    plt.ylabel("AUROC @ 5yr")
    plt.title(f"{modality} Average AUROC5 Across {n_splits} Folds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{modality}_avg_auroc.png"))
    
    plt.close()

def auroc_at_horizon(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    horizon: float
) -> float:
    # mask for those we can evaluate
    mask = (times >= horizon) | ((times < horizon) & (events == 1))
    if mask.sum() < 2:
        return float("nan")
    t_sub  = times[mask]
    e_sub  = events[mask]
    r_sub  = risks[mask]
    # binary label: relapse by horizon?
    labels = ((t_sub <= horizon) & (e_sub == 1)).astype(int)

    return roc_auc_score(labels, r_sub)

def compute_baseline_metrics(T_t, E_t, S_t, device, horizon):
    ci = concordance_index(T_t,-S_t, E_t)
    auroc =  auroc_at_horizon(
        times=T_t, events=E_t, risks=S_t, horizon=horizon
    )    
    return {"ci": ci, "auroc": auroc}

def save_per_patient_csv(
    model,
    device: torch.device,
    out_dir: str,
    fold: int,
    modality: str,
    split_label: str,
    full_loader: torch.utils.data.DataLoader,
    S_np: np.ndarray
) -> pd.DataFrame:
    """
    Runs the model on every sample in full_loader, collects per-patient risks,
    and writes out a CSV with columns [split, patient_id, T, E, risk, score].
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in full_loader:
            x_batch, t_batch, e_batch, ids = batch
            x_batch = x_batch.to(device)
            # forward pass → (B,) risk scores
            with torch.amp.autocast("cuda"):
                risks = model(x_batch).detach().cpu().numpy()
            
            T_np = t_batch.numpy()
            E_np = e_batch.numpy()
            
            for i, pid in enumerate(ids):
                rows.append({
                    "split":      split_label,
                    "patient_id": pid,
                    "T":          float(T_np[i]),
                    "E":          float(E_np[i]),
                    "risk":       float(risks[i]),
                    "score":      float(S_np[i])
                })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        out_dir,
        f"{modality}_fold_{fold}_{split_label}_patients.csv"
    )
    df.to_csv(csv_path, index=False)
    return df

def load_ct_features(ids, ct_folder):
    npy_paths = []
    for pid in ids:
        fn = f"{pid}.npy"
        full = os.path.join(ct_folder, fn)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"No pre‑normalized .npy found for {pid} at {full}")
        npy_paths.append(full)
    return np.array(npy_paths, dtype=object)

def stack_pad_nan(arrays):
    # 1) find the maximum length across all folds
    lengths = [len(arr) for arr in arrays]
    max_len = max(lengths)
    # 2) pad each array with NaNs up to max_len
    padded = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        n = arr.shape[0]
        if n < max_len:
            pad = np.full((max_len - n,), np.nan, dtype=arr.dtype)
            arr = np.concatenate([arr, pad])
        padded.append(arr)
    # 3) stack into shape (n_folds, max_len)
    return np.stack(padded)

def is_unstable(scores, window=50,
                min_slope=5e-4,   
                max_std=5e-3,
                min_ci = 0.5,
                smooth_alpha=None):  
    
    y = np.array(scores, dtype=float)
    if smooth_alpha is not None:
        # simple EMA
        ema = [y[0]]
        for v in y[1:]:
            ema.append(ema[-1] * (1 - smooth_alpha) + v * smooth_alpha)
        y = np.array(ema)
    
    if len(y) < window:
        return False, {"reason": f"need ≥{window} points, got {len(y)}"}
    
    recent = y[-window:]
    x = np.arange(window)
    slope, _, r_val, p_val, stderr = linregress(x, recent)
    std = recent.std()
    
    current_ci = y[-1]

    metrics = {
        "slope":    slope,
        "std":      std,
        "r_value":  r_val,
        "p_value":  p_val,
    }
    # print(metrics)
    trend_ok = (slope <= min_slope) or (std >= max_std)
    ci_ok    = (current_ci < min_ci)
    
    good = trend_ok and ci_ok
    return good, metrics


def save_per_patient_csv_multi(
    model,
    device: torch.device,
    out_dir: str,
    fold: int,
    modality: str,
    split_label: str,
    full_loader: torch.utils.data.DataLoader,
    S_np: np.ndarray
) -> pd.DataFrame:
    """
    Runs the model on every sample in full_loader, collects per-patient risks,
    and writes out a CSV with columns [split, patient_id, T, E, risk, score].
    """
    model.eval()
    rows = []
    with torch.no_grad():            
        for ct, path, time, event, ids in full_loader: 
            ct, path = ct.to(device), path.to(device)
            risks = model(ct, path).detach().cpu().numpy()
            
            T_np =  time.numpy()
            E_np =  event.numpy()
            
            for i, pid in enumerate(ids):
                rows.append({
                    "split":      split_label,
                    "patient_id": pid,
                    "T":          float(T_np[i]),
                    "E":          float(E_np[i]),
                    "risk":       float(risks[i]),
                    "score":      float(S_np[i])
                })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        out_dir,
        f"{modality}_fold_{fold}_{split_label}_patients.csv"
    )
    df.to_csv(csv_path, index=False)
    return df

def plot_multiple_roc_curves(
    y_true_list,
    y_pred_list,
    plot_dir="./plots",
    modality="model"
):
    os.makedirs(plot_dir, exist_ok=True)
    labels=[f"Fold {i+1}" for i in range(len(y_pred_list))]
    plt.figure(figsize=(6, 6))
    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        label = f"{labels[i]} (AUC = {roc_auc:.2f})" if labels else f"Fold {i+1} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, lw=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    filename = f"{modality.lower()}_roc.png"
    save_path = os.path.join(plot_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {save_path}")

def load_fold_predictions(plot_dir_final, modality, n_splits):
    """
    Loads per-patient CSVs from each fold and extracts labels and predicted risks.
    
    Returns:
        y_true_list: list of binary labels for AUROC (relapse before 5 yrs)
        y_pred_list: list of predicted risk scores
    """
    y_true_list = []
    y_pred_list = []

    for fold in range(1, n_splits + 1):
        csv_path = os.path.join(
            plot_dir_final,
            f"{modality}_fold_{fold}_val_patients.csv"
        )
        df = pd.read_csv(csv_path)

        # Create binary label: relapse before 5 years (T <= 60 months) AND event occurred
        labels = ((df["T"] <= 60) & (df["E"] == 1)).astype(int)
        risks  = df["risk"].values

        y_true_list.append(labels.values)
        y_pred_list.append(risks)
    
    return y_true_list, y_pred_list
