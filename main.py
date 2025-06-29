import os
import argparse
import numpy as np
import torch
import nibabel as nib 
from pathlib import Path

from hparams import *
from utils import *
from train_unimodal import nested_cross_validate_singlemodality 
from train_multi import nested_cross_validate_fusion

def get_outer_args(
    modality,
    plot_dir_tuning,
    plot_dir_final,
    n_splits=5,
    n_inner=3,
    n_trials=50,
    batch_size=2,
    epochs=10,
    start_early_stop=15,
    random_state=42,
    early_stop_patience=2000,
    horizon=5*12,
    tune_c_index=True,
    state_dict=False,
    resnet=True
):
    return {
        "n_splits":            n_splits,
        "n_inner":             n_inner,
        "n_trials":            n_trials,
        "device":              torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size":          batch_size,
        "epochs":              epochs,
        "start_early_stop":    start_early_stop,
        "random_state":        random_state,
        "early_stop_patience": early_stop_patience,
        "plot_dir_tuning":     plot_dir_tuning,
        "plot_dir_final":      plot_dir_final,
        "modality":            modality,
        "horizon":             horizon,
        "tune_c_index":        tune_c_index,
        "state_dict":          state_dict,
        "resnet":              resnet 
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested CV for survival models (unimodal / late-fusion / intermediate)."
    )

    parser.add_argument(
        "--patient_csv_file",
        type=str,
        required=True,
        help="Path to the CSV file containing patient clinical data."
    )

    parser.add_argument(
        "--path_wsi",
        type=str,
        required=True,
        help="Directory containing WSI feature vectors."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ct", "wsi", "late", "concat", "corr"],
        required=True,
        help="Fusion mode: 'ct', 'wsi', 'late', 'intermediate', or 'shared_specific'."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (e.g. 'CTonly', 'WSIonly', 'CT+WSI')."
    )
    parser.add_argument(
        "--n_trial",
        type=int,
        default=50,
        help="Number of Optuna trials in the inner loop (default: 50)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for tuning (default: 10)."
    )
    parser.add_argument(
        "--start_early_stop",
        type=int,
        default=15,
        help="Number of epochs for the final run (default: 15)."
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=2000,
        help="Early stopping."
    )
    parser.add_argument(
        "--ct_model",
        type=str,
        choices=["resnet", "swinunetr"],
        required=True,
        help="Select CT model."
    )

    args = parser.parse_args()

    patient_csv_file   = args.patient_csv_file
    
    path_wsi_features  = args.path_wsi
    mode               = args.mode
    exp_name           = args.exp_name  
    n_trials           = args.n_trial
    epochs             = args.epochs
    start_early_stop   = args.start_early_stop

    # ── Load patient dictionary & all feature vectors ──
    patients = create_patient_dict(patient_csv_file)
    
    if args.ct_model == "resnet":
        pretrained_ckpt_path = '/gpfs/home4/cblommestijn/Master_thesis/MedicalNet/pretrain/resnet_18_23dataset.pth'
        ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}  
        resnet = True
        path_ct_features   = "/scratch-shared/cblommestijn/data/nnUNet_results/TCGA_KIRC/npy_downsampled_zscore_all_axes_heavy_1"
    elif args.ct_model == "swinunetr":
        state_dict = torch.load("/scratch-shared/cblommestijn/data/model_swinvit.pt")
        resnet=False
        path_ct_features = "/scratch-shared/cblommestijn/data/nnUNet_results/TCGA_KIRC/npy_downsampled_ViT_2"
        
    path_ct_dummy = '/scratch-shared/cblommestijn/CT_TCGA_KIRC/feature_vectors'
    X_ct_fixed, X_wsi_tabular, T_tensor, E_tensor, S_tensor, ids = generate_fusion_feature_vectors(
        path_ct_dummy, path_wsi_features, create_patient_dict(patient_csv_file)
    )
    # Convert all to NumPy
    T = T_tensor.numpy()
    E = E_tensor.numpy()
    S = S_tensor.numpy()
    ids = list(ids)

    # Build the two plot directories once
    plot_dir_tuning = os.path.join("final/plots", exp_name , "tuning")
    plot_dir_final  = os.path.join("final/plots", exp_name , "final")
    os.makedirs(plot_dir_tuning, exist_ok=True)
    os.makedirs(plot_dir_final,  exist_ok=True)
    
    # Build outer_args from CLI
    outer_args = get_outer_args(
        modality=exp_name,
        plot_dir_tuning=plot_dir_tuning,
        plot_dir_final=plot_dir_final,
        n_trials=args.n_trial,
        epochs=args.epochs,
        start_early_stop=args.start_early_stop,
        early_stop_patience=args.early_stop,
        state_dict=state_dict,
        resnet=resnet
    )

    if mode == "ct" or mode == "wsi":
        ct_features = load_ct_features(ids, path_ct_features)
        if mode == "wsi":
            make_hparams_fn = make_hparams_wsi
        else:
            make_hparams_fn  = make_hparams_uni
        finalize_fn      = finalize_hparams_uni
        (ci_mean, ci_std), (auroc_mean, auroc_std) = nested_cross_validate_singlemodality(
            mode,
            X_wsi_tabular,
            ct_features,
            X_ct_fixed,
            T, E, S, ids,
            exp_name,
            outer_args,
            make_hparams_fn,
            finalize_fn
        )
    elif mode == "late":
        ct_features = load_ct_features(ids, path_ct_features)
        make_hparams_fn  = make_hparams_late
        finalize_fn      = finalize_hparams_late
        (ci_mean, ci_std), (auroc_mean, auroc_std) = nested_cross_validate_fusion(
            mode,
            ct_features,  X_wsi_tabular, X_ct_fixed,
            T, E, S, ids,
            exp_name, outer_args,
            make_hparams_fn, finalize_fn
        )
        
    elif mode == "concat" or mode == "corr":
        ct_features = load_ct_features(ids, path_ct_features)
        make_hparams_fn  = make_hparams_concat
        finalize_fn      = finalize_hparams_uni
        (ci_mean, ci_std), (auroc_mean, auroc_std) = nested_cross_validate_fusion(
            mode,
            ct_features,  X_wsi_tabular, X_ct_fixed,
            T, E, S, ids,
            exp_name, outer_args,
            make_hparams_fn, finalize_fn
        )

    else:
        raise ValueError(f"Invalid mode: {mode}") 
    
    
    
    
    


