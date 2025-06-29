import os
import numpy as np
import torch
import torch.nn.functional as F
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import time as timer
import torch.nn as nn
from scipy.stats import linregress
import pandas as pd

from model import (
    CoxModelFixed,      
    cox_loss,
    WSIDataset,
    CTDataset,  
    MedicalNet3D,
    FusionDataset,
    MultiModalCox3D,
    CorrFusionCox
)
from utils import (
    plot, 
    plot_average_metrics, 
    save_per_patient_csv, 
    save_per_patient_csv_multi,
    auroc_at_horizon, 
    compute_baseline_metrics,
    stack_pad_nan,
    is_unstable,
    load_fold_predictions,
    plot_multiple_roc_curves
)

import matplotlib.pyplot as plt
import seaborn as sns

def save_time_distribution_by_fold(T, E, fold_assignments, save_path):
    """
    Save KDE plot of time-to-event (E == 1) distribution per fold.

    Args:
        T (array-like): Time to event.
        E (array-like): Event indicator (1 = event occurred).
        fold_assignments (array-like): Fold number or label per sample.
        save_path (str): File path to save the plot (e.g. '/path/to/kde_plot.png').
    """
    df = pd.DataFrame({
        'Time': T,
        'Event': E,
        'Fold': fold_assignments
    })

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df[df['Event'] == 1], x='Time', hue='Fold', common_norm=False)
    plt.title("Time-to-Event Distribution (E == 1) by Fold")
    plt.xlabel("Time to Event (T)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def create_stratify_labels(events, times, n_bins=10):
    time_bins = pd.qcut(times, q=n_bins, duplicates='drop').codes  # handle duplicates
    return np.char.add(events.astype(str), np.char.add("_", time_bins.astype(str)))


def run_fusion_fold(
    trial, mode,
    X_ct_tr, X_ct_te,
    path_tr, path_te,
    T_tr, E_tr, S_tr, ids_tr,
    T_te, E_te, S_te, ids_te,
    hparams,
    tuning=False
):
    best_val_ci = 0.0
    best_val_auroc = 0.0
    best_epoch = 0
    no_improve = 0
    
    train_cis, val_cis = [], []
    train_aucs, val_aucs = [], []
    hparams["dropout"] = 0.25
    epochs = hparams["epochs"]
    device = hparams['device']
    
    base_train = compute_baseline_metrics(T_tr, E_tr, S_tr, hparams["device"], hparams["horizon"])
    base_val   = compute_baseline_metrics(T_te, E_te, S_te, hparams["device"], hparams["horizon"])
    
    hparams["base_tr_ci"]    = base_train["ci"]
    hparams["base_va_ci"]    = base_val["ci"]
    hparams["base_tr_auroc5"] = base_train["auroc"]
    hparams["base_va_auroc5"] = base_val["auroc"]
    
    scaler = StandardScaler()
    path_tr = scaler.fit_transform(path_tr)   
    path_te = scaler.transform(path_te) 
    path_tr = torch.as_tensor(path_tr, dtype=torch.float32)
    path_te = torch.as_tensor(path_te, dtype=torch.float32)
    train_ds = FusionDataset(X_ct_tr, path_tr, T_tr, E_tr, ids_tr)
    val_ds   = FusionDataset(X_ct_te, path_te, T_te, E_te, ids_te)
    if mode == "concat":
        l1_lambda = hparams["l1_lambda"]
        model = MultiModalCox3D(
            state_dict=hparams["state_dict"],
            sample_input_D=8, 
            sample_input_H=128, 
            sample_input_W=196,
            pathology_dim=path_tr.shape[1], 
            hidden_dim=hparams['hidden_dim'], 
            dropout=hparams['dropout'],
            resnet = hparams["resnet"] 
        )
        model.to(device)
        # optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )
        backbone_params = list(model.backbone.parameters())
        head_params     = [p for n, p in model.named_parameters() if "backbone" not in n]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hparams["backbone_lr"],"weight_decay": hparams["weight_decay"]},  # e.g. 1e-5
                {"params": head_params,     "lr": hparams["learning_rate"], "weight_decay": hparams["weight_decay"]} # e.g. 1e-4
            ]
        )
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        if hparams["resnet"]:         
            for name, param in model.backbone.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True
        else:
            for name, param in model.backbone.named_parameters():
                if "decoder" in name:
                    param.requires_grad = True
                    
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=hparams['eta_min'])
    elif mode == "corr":
        l1_lambda = hparams["l1_lambda"]
        model = CorrFusionCox(
            state_dict=hparams["state_dict"],
            sample_input_D=8, 
            sample_input_H=128, 
            sample_input_W=196,
            pathology_dim=path_tr.shape[1], 
            hidden_dim=hparams['hidden_dim'], 
            dropout=hparams['dropout'],
            resnet = hparams["resnet"] 
        )
        model.to(device)
        # optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )
        all_params = list(model.parameters())

        # Backbone params only
        backbone_params = list(model.backbone.parameters())

        # Everything else = model parameters not in backbone
        backbone_param_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in all_params if id(p) not in backbone_param_ids]


        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": hparams["backbone_lr"], "weight_decay": 0.0},
            {"params": head_params,     "lr": hparams["learning_rate"], "weight_decay": hparams["weight_decay"]},
        ])        
        
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=hparams['eta_min'])
    elif mode == "late":
        alpha = hparams["ct_alpha"]
        l1_lambda_ct = hparams["ct_l1_lambda"]   
        model_ct = MedicalNet3D(
            state_dict=hparams["state_dict"],
            sample_input_D=8,
            sample_input_H=128,
            sample_input_W=196,
            hidden_dim=hparams["ct_hidden_dim"],
            dropout=hparams["dropout"],
            resnet=hparams["resnet"]        
        )   
        
        l1_lambda_path = hparams["wsi_l1_lambda"]
        model_path = CoxModelFixed(
            input_dim=path_tr.shape[1],
            hidden_dim=hparams["wsi_hidden_dim"],
            dropout=hparams["dropout"]
        )

        backbone_params = list(model_ct.backbone.parameters())
        head_params     = [p for n, p in model_ct.named_parameters() if "backbone" not in n]
        optimizer_ct = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hparams["ct_backbone_lr"], "weight_decay": 0.0},  # e.g. 1e-5
                {"params": head_params,     "lr": hparams["ct_learning_rate"], "weight_decay": hparams["ct_weight_decay"]} # e.g. 1e-4
            ]
        )
        
        if hparams["resnet"]:         
            for name, param in model_ct.backbone.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True
        else:
            for name, param in model_ct.backbone.named_parameters():
                if "decoder" in name:
                    param.requires_grad = True

        optimizer_path = torch.optim.AdamW(
            model_path.parameters(),
            lr=hparams["wsi_learning_rate"],
            weight_decay=hparams["wsi_weight_decay"]
        )
        model_ct.to(device)
        model_path.to(device)
        scheduler_cos_ct = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ct, T_max=epochs, eta_min=hparams["ct_eta_min"])
        scheduler_cos_path = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_path, T_max=epochs, eta_min=hparams["wsi_eta_min"])

    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(1, epochs + 1):
        if mode == "concat" or mode == "corr":
            model.train()
        elif mode == "late":
            model_ct.train()
            model_path.train()
        for ct, path, time, event, _ in train_loader:
            ct, path = ct.to(device), path.to(device)
            if mode == "concat" or mode == "corr":
                risk = model(ct, path).flatten()
                base_loss = cox_loss(risk, time.to(device), event.to(device))
                # l1_penalty = sum(p.abs().sum() for p in model.parameters())
                # loss = base_loss + l1_lambda * l1_penalty 
                l1_penalty = sum(p.abs().sum() for p in head_params)
                loss = base_loss + l1_lambda * l1_penalty
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            elif mode == "late":
                risk_ct = model_ct(ct).flatten()                
                base_loss_ct = cox_loss(risk_ct, time.to(device), event.to(device))
                l1_penalty_ct = sum(p.abs().sum() for p in model_ct.parameters())
                loss_ct = base_loss_ct + l1_lambda_ct * l1_penalty_ct 
                loss_ct.backward() 
                torch.nn.utils.clip_grad_norm_(model_ct.parameters(), max_norm=1.0)
                optimizer_ct.step()  
                optimizer_ct.zero_grad()
                
                risk_path = model_path(path).flatten()
                base_loss_path = cox_loss(risk_path, time.to(device), event.to(device))
                l1_penalty_path = sum(p.abs().sum() for p in model_path.parameters())
                loss_path = base_loss_path + l1_lambda_path * l1_penalty_path 
                loss_path.backward() 
                torch.nn.utils.clip_grad_norm_(model_path.parameters(), max_norm=1.0)
                optimizer_path.step()
                optimizer_path.zero_grad()
                
            
        if mode == "concat" or mode == "corr":
            scheduler_cos.step()
        elif mode == "late":
            scheduler_cos_ct.step()
            scheduler_cos_path.step()
            
        all_outs, all_times, all_events = [], [], []        
        if mode == "concat" or mode == "corr":
            model.eval()
        elif mode == "late":
            model_ct.eval()
            model_path.eval()
        
        if not tuning:            
            all_r_tr, all_t_tr, all_e_tr = [], [], []
            with torch.no_grad():            
                for ct, path, time, event, _ in train_loader: 
                    ct, path = ct.to(device), path.to(device)
                    time = time.to(device)
                    event= event.to(device)
                    if mode == "concat" or mode == "corr":
                        risk = model(ct, path).flatten() 
                    elif mode == "late":
                        risk_ct = model_ct(ct).flatten() 
                        risk_path = model_path(path).flatten()
                        risk = alpha * risk_path + (1-alpha)*risk_ct
                    all_r_tr.append(risk.detach().cpu().numpy())
                    all_t_tr.append(time.detach().cpu().numpy())
                    all_e_tr.append(event.detach().cpu().numpy())
                    
            all_r_tr = np.concatenate(all_r_tr,   axis=0)
            all_t_tr = np.concatenate(all_t_tr,   axis=0)
            all_e_tr = np.concatenate(all_e_tr,   axis=0)  
            
            ci_tr = concordance_index(all_t_tr, -all_r_tr, all_e_tr)
            auroc_tr = auroc_at_horizon(
                all_t_tr, all_e_tr, all_r_tr, hparams["horizon"]
            )          
            train_cis.append(ci_tr)            
            train_aucs.append(auroc_tr)        
            
        with torch.no_grad():            
            for ct, path, time, event, _ in val_loader: 
                ct, path = ct.to(device), path.to(device)
                time = time.to(device)
                event= event.to(device)
                if mode == "concat" or mode == "corr":
                        risk = model(ct, path).flatten() 
                elif mode == "late":
                    risk_ct = model_ct(ct).flatten() 
                    risk_path = model_path(path).flatten()
                    risk = alpha * risk_path + (1-alpha)*alpha    
                
                all_outs.append(risk.detach().cpu().numpy())
                all_times.append(time.detach().cpu().numpy())
                all_events.append(event.detach().cpu().numpy())

        risks_np  = np.concatenate(all_outs,   axis=0)
        times_np  = np.concatenate(all_times,   axis=0)
        events_np = np.concatenate(all_events,  axis=0)
        
        ci_va   = concordance_index(times_np,  -risks_np,  events_np)
        auroc_va = auroc_at_horizon(
            times_np, events_np, risks_np, hparams["horizon"]
        )
        
        val_cis.append(ci_va)
        val_aucs.append(auroc_va)
        is_stable, slope = is_unstable(val_cis)
        if  is_stable and tuning:
            print(f"Trial stopped early due to unstable validation at epoch {epoch} with slope {slope}. Performance C-index: {ci_va} AUROC: {auroc_va} ")
            return 0, 0, train_cis, val_cis, train_aucs, val_aucs, epoch

        if epoch >= hparams["start_early_stop"] and tuning:
            if ci_va >= best_val_ci:
                best_val_ci = ci_va
                best_val_auroc = auroc_va
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

        if no_improve >= hparams["early_stop_patience"] and tuning and epoch >= hparams["start_early_stop"]:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch} best_val_ci={best_val_ci:.4f}, last_val_ci={ci_va:.4f}")
            break

    final_ci, final_auroc = ci_va, auroc_va

    if not tuning:
        print(f"→ Fold {hparams['fold']} best at epoch {epoch}:  CI={ci_va:.4f}, AUROC5={auroc_va:.4f}")
        plot(hparams["plot_dir_final"], hparams, epoch, train_cis, train_aucs, val_cis, val_aucs)
        if mode == "concat" or mode == "corr":
            ckpt = {
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ci":        best_val_ci,
                "hparams":            hparams,
            }
            ckpt_path = os.path.join(
                hparams["plot_dir_final"],
                f"checkpoint_fold{hparams['fold']}.pt"
            )
            torch.save(ckpt, ckpt_path)
            print(f"Saved outer‑fold checkpoint to {ckpt_path}")
       
            per_patient_df = pd.DataFrame([])        

            per_val = save_per_patient_csv_multi(
                model = model,
                device = device,
                out_dir=hparams["plot_dir_final"],
                fold=hparams["fold"],
                modality=hparams["modality"],
                split_label="val",
                full_loader=val_loader,
                S_np=S_te
            )
        elif mode == "late":
            ckpt_ct = {
                "epoch": epoch,
                "model_state_dict": model_ct.state_dict(),
                "optimizer_state_dict": optimizer_ct.state_dict(),
                "best_val_ci": best_val_ci,
                "hparams": hparams,
            }
            ckpt_path_ct = os.path.join(hparams["plot_dir_final"], f"checkpoint_ct_fold{hparams['fold']}.pt")
            torch.save(ckpt_ct, ckpt_path_ct)

            ckpt_path = {
                "epoch": epoch,
                "model_state_dict": model_path.state_dict(),
                "optimizer_state_dict": optimizer_path.state_dict(),
                "best_val_ci": best_val_ci,
                "hparams": hparams,
            }
            ckpt_path_path = os.path.join(hparams["plot_dir_final"], f"checkpoint_path_fold{hparams['fold']}.pt")
            torch.save(ckpt_path, ckpt_path_path)

            print(f"Saved outer-fold checkpoints to {ckpt_path_ct} and {ckpt_path_path}")

            # Create a wrapper model for prediction
            class LateFusionModel(torch.nn.Module):
                def __init__(self, model_ct, model_path, alpha):
                    super().__init__()
                    self.model_ct = model_ct
                    self.model_path = model_path
                    self.alpha = alpha
                def forward(self, ct, path):
                    r_ct = self.model_ct(ct).flatten()
                    r_path = self.model_path(path).flatten()
                    return self.alpha * r_path + (1 - self.alpha) * r_ct

            fused_model = LateFusionModel(model_ct, model_path, alpha).to(device)
            per_val = save_per_patient_csv_multi(
                model=fused_model,
                device=device,
                out_dir=hparams["plot_dir_final"],
                fold=hparams["fold"],
                modality=hparams["modality"],
                split_label="val",
                full_loader=val_loader,
                S_np=S_te
            )
    
    return final_ci, final_auroc, train_cis, val_cis, train_aucs, val_aucs, best_epoch     

def tune_fusion(
    X_ct, path_feats,
    times, events, scores, ids,
    make_hparams_fn, finalize_hparams_fn,
    outer_args, fold, mode
):
    """
    Performs inner cross-validation via Optuna to select best hyperparameters for fusion model.
    """
    def objective(trial):
        # Sample hyperparameters
        hparams = make_hparams_fn(trial, fold, outer_args)
        cis, aucs = [], []

        skf = StratifiedKFold(
            n_splits=outer_args['n_inner'],
            shuffle=True, random_state=outer_args['random_state']
        )
        start_time = timer.time()
        epochs = []
        failed = 0
        
        # strat_labels = create_stratify_labels(events, times)

         
              
        for tr_idx, va_idx in skf.split(X_ct, events):
            X_tr, X_va = X_ct[tr_idx], X_ct[va_idx]
            P_tr, P_va = path_feats[tr_idx], path_feats[va_idx]
            T_tr, T_va = times[tr_idx], times[va_idx]
            E_tr, E_va = events[tr_idx], events[va_idx]
            S_tr, S_va = scores[tr_idx], scores[va_idx]
            ids_tr_sub = [ids[i] for i in tr_idx]
            ids_va_sub = [ids[i] for i in va_idx]
            ci, auc, _, _, _, _, epoch = run_fusion_fold(
                trial, mode,
                X_tr, X_va,
                P_tr, P_va,
                T_tr, E_tr, S_tr, ids_tr_sub,
                T_va, E_va, S_va, ids_va_sub,
                hparams, tuning=True
            )
            epochs.append(epoch)
            cis.append(ci); aucs.append(auc)
        end_time = timer.time()
        elapsed = end_time - start_time   
        hparams["mean_inner_epochs"] = int(np.mean(epochs))
        trial.set_user_attr("mean_inner_epochs", hparams["mean_inner_epochs"])
        print(f" Trial completed in {elapsed:.2f} seconds") 
        # return float(np.min(cis)) if outer_args["tune_c_index"] else float(np.min(aucs))
        return float(np.mean(cis)) - float(np.std(cis))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=outer_args['n_trials'])

    best = study.best_trial.params
    best["final_epochs"] = study.best_trial.user_attrs["mean_inner_epochs"]
    
    return finalize_hparams_fn(best, outer_args, fold)

def nested_cross_validate_fusion(
    mode,
    X_ct, path_feats, ct_fixed, 
    times, events, scores, ids,
    modality_name, outer_args,
    make_hparams_fn, finalize_hparams_fn
):
    """
    Runs outer StratifiedKFold and inner Optuna tuning for fusion model.
    """


    # Stratified split using combined labels
    skf = StratifiedKFold(n_splits=outer_args['n_splits'], shuffle=True, random_state=outer_args['random_state'])
    outer_cis, outer_aucs = [], []
    base_cis, base_aucs = [], []
    all_train_cis, all_val_cis = [], []
    all_train_aucs, all_val_aucs = [], []

    # strat_labels = create_stratify_labels(events, times)

    # fold_assignments = np.empty(len(events), dtype=object)
    # for fold, (tr_idx, te_idx) in enumerate(skf.split(X_ct, strat_labels), start=1):
    #     fold_assignments[te_idx] = f"fold_{fold}"      
    # save_time_distribution_by_fold(
    #     T=times,
    #     E=events,
    #     fold_assignments=fold_assignments,
    #     save_path=os.path.join(outer_args["plot_dir_final"], "kde_time_by_fold.png")
    # )   
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_ct, events), start=1):
        X_tr, X_te = X_ct[tr_idx], X_ct[te_idx]
        P_tr, P_te = path_feats[tr_idx], path_feats[te_idx]
        T_tr, T_te = times[tr_idx], times[te_idx]
        E_tr, E_te = events[tr_idx], events[te_idx]
        S_tr, S_te = scores[tr_idx], scores[te_idx]
        ids_tr = [ids[i] for i in tr_idx]
        ids_te = [ids[i] for i in te_idx]

        base_tr = compute_baseline_metrics(T_tr, E_tr, S_tr,
                                           outer_args["device"],
                                           outer_args["horizon"])
        base_te = compute_baseline_metrics(T_te, E_te, S_te,
                                           outer_args["device"],
                                           outer_args["horizon"])
        print(f"  Baseline Test CI={base_te['ci']:.4f}, AUROC={base_te['auroc']:.4f}")
        base_cis.append(base_te['ci'])
        base_aucs.append(base_te['auroc'])
        
        # Tune hyperparameters on train split
        best_hparams = tune_fusion(
            X_tr, P_tr,
            T_tr, E_tr, S_tr, ids_tr,
            make_hparams_fn, finalize_hparams_fn,
            outer_args, fold, mode
        )
        start_time = timer.time()
        # Train & evaluate on outer test
        ci_outer, auc_outer, tr_cis, val_cis, tr_aucs, val_aucs, _ = run_fusion_fold(
            1, mode,
            X_tr, X_te,
            P_tr, P_te,
            T_tr, E_tr, S_tr, ids_tr,
            T_te, E_te, S_te, ids_te,
            best_hparams, tuning=False
        )
        end_time = timer.time()
        elapsed = end_time - start_time  
        print(f" Outer fold completed in {elapsed:.2f} seconds")
        print(f"→ Fold {fold} C-index: {ci_outer:.4f}, AUROC5: {auc_outer:.4f}")
        outer_cis.append(ci_outer)
        outer_aucs.append(auc_outer)
        all_train_cis.append(tr_cis)
        all_val_cis.append(val_cis)
        all_train_aucs.append(tr_aucs)
        all_val_aucs.append(val_aucs)

    # Stack + average metrics across folds for plotting
    all_train_cis_mat   = stack_pad_nan(all_train_cis)
    all_val_cis_mat     = stack_pad_nan(all_val_cis)
    all_train_aucs_mat  = stack_pad_nan(all_train_aucs)
    all_val_aucs_mat    = stack_pad_nan(all_val_aucs)

    # compute nan‐aware stats along axis=0 (across folds)
    mean_train_cis  = np.nanmean(all_train_cis_mat, axis=0)
    std_train_cis   = np.nanstd(all_train_cis_mat, axis=0)
    mean_val_cis    = np.nanmean(all_val_cis_mat, axis=0)
    std_val_cis     = np.nanstd(all_val_cis_mat, axis=0)
    mean_train_aucs = np.nanmean(all_train_aucs_mat, axis=0)
    std_train_aucs  = np.nanstd(all_train_aucs_mat, axis=0)
    mean_val_aucs   = np.nanmean(all_val_aucs_mat, axis=0)
    std_val_aucs    = np.nanstd(all_val_aucs_mat, axis=0)

    # base metrics remain the same
    mean_base_ci   = np.mean(base_cis)
    std_base_ci    = np.std(base_cis)
    mean_base_auc  = np.mean(base_aucs)
    std_base_auc   = np.std(base_aucs)

    plot_average_metrics(
        modality=modality_name,
        plot_dir=outer_args["plot_dir_final"],
        epochs=np.arange(1, len(mean_train_cis) + 1),
        mean_train_cis=mean_train_cis, std_train_cis=std_train_cis,
        mean_val_cis=mean_val_cis, std_val_cis=std_val_cis,
        mean_train_aucs=mean_train_aucs, std_train_aucs=std_train_aucs,
        mean_val_aucs=mean_val_aucs, std_val_aucs=std_val_aucs,
        mean_base_ci=mean_base_ci, std_base_ci=std_base_ci,
        mean_base_auc=mean_base_auc, std_base_auc=std_base_auc,
        n_splits=outer_args["n_splits"],
    )

    print(f"\nNested CV ({modality_name}) - C-index: {np.mean(outer_cis):.4f} ± {np.std(outer_cis):.4f}")
    print(f"Nested CV ({modality_name}) - AUROC5 : {np.mean(outer_aucs):.4f} ± {np.std(outer_aucs):.4f}")
    print(f"Leibovich baseline  - C-index: {mean_base_ci:.4f} ± {std_base_ci:.4f}")
    print(f"Leibovich baseline  - AUROC5 : {mean_base_auc:.4f} ± {std_base_auc:.4f}")
    y_true_list, y_pred_list = load_fold_predictions(
        plot_dir_final=outer_args["plot_dir_final"],
        modality=modality_name,
        n_splits=outer_args["n_splits"]
    )

    plot_multiple_roc_curves(
        y_true_list=y_true_list,
        y_pred_list=y_pred_list,
        plot_dir=outer_args["plot_dir_final"],
        modality=modality_name
    )
    return (np.mean(outer_cis), np.std(outer_cis)), (np.mean(outer_aucs), np.std(outer_aucs))
