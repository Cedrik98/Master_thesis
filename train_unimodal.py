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
    MedicalNet3D
)
from utils import (
    plot, 
    plot_average_metrics, 
    save_per_patient_csv, 
    auroc_at_horizon, 
    compute_baseline_metrics,
    stack_pad_nan,
    is_unstable,
    load_fold_predictions,
    plot_multiple_roc_curves
)

def run_fold_singlemodality(
    trial,
    mode,
    X_tr,   
    X_te,    
    T_tr, E_tr, S_tr, ids_tr,
    T_te, E_te, S_te, ids_te,
    hparams,
    tuning: bool
):
    best_epoch = 0
    hparams["dropout"] = 0.25
    device = hparams["device"]
    l1_lambda = hparams["l1_lambda"]
    base_train = compute_baseline_metrics(T_tr, E_tr, S_tr, hparams["device"], hparams["horizon"])
    base_val   = compute_baseline_metrics(T_te, E_te, S_te, hparams["device"], hparams["horizon"])

    hparams["base_tr_ci"]    = base_train["ci"]
    hparams["base_va_ci"]    = base_val["ci"]
    hparams["base_tr_auroc5"] = base_train["auroc"]
    hparams["base_va_auroc5"] = base_val["auroc"]
    if mode == "wsi":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)   
        X_te = scaler.transform(X_te)    
        
        X_tr = torch.as_tensor(X_tr, dtype=torch.float32)
        X_te = torch.as_tensor(X_te, dtype=torch.float32)
        
    T_tr_t = torch.tensor(T_tr, dtype=torch.float32)
    E_tr_t = torch.tensor(E_tr, dtype=torch.float32)
    T_te_t = torch.tensor(T_te, dtype=torch.float32)
    E_te_t = torch.tensor(E_te, dtype=torch.float32)

    if mode == "wsi":
        train_ds = WSIDataset(X_tr, T_tr, E_tr, ids_tr)
        val_ds   = WSIDataset(X_te, T_te, E_te, ids_te)
        model = CoxModelFixed(
            input_dim=X_tr.shape[1],
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams["dropout"]
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
    elif mode == "ct":
        train_ds = CTDataset(X_tr, T_tr, E_tr, ids_tr)
        val_ds   = CTDataset(X_te, T_te, E_te, ids_te)
        model = MedicalNet3D(
            state_dict=hparams["state_dict"],
            sample_input_D=8,
            sample_input_H=128,
            sample_input_W=196,
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams["dropout"],
            resnet= hparams["resnet"]           
        )              
        model.to(device)
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
        # for param in model.backbone.parameters():
        #         param.requires_grad = False
        
        # if hparams["resnet"]:         
        #     for name, param in model.backbone.named_parameters():
        #         if "layer4" in name:
        #             param.requires_grad = True
        # else:
        #     for name, param in model.named_parameters():
        #         if "decoder" in name:
        #             param.requires_grad = True
                    
    train_loader = DataLoader(
        train_ds,
        batch_size=hparams["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,             
        persistent_workers=True,    
        prefetch_factor=2           
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hparams["batch_size"],
        shuffle=False,              
        drop_last=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    
    epochs = hparams["epochs"]
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=hparams['eta_min'])
    
    best_val_ci = 0.0
    best_val_auroc = 0.0
    best_epoch = 0
    no_improve = 0

    train_cis, val_cis = [], []
    train_aucs, val_aucs = [], []
   
    for epoch in range(1, epochs + 1):
        model.train()        
        for batch in train_loader:

            x = batch[0].to(device, non_blocking=True)
            time = batch[1].to(device, non_blocking=True)
            event = batch[2].to(device, non_blocking=True)
        
            risk = model(x).flatten()
            base_loss = cox_loss(risk.float(), time, event)
            
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = base_loss + l1_lambda * l1_penalty
            # l1_penalty = sum(p.abs().sum() for p in head_params)
            # loss = base_loss + l1_lambda * l1_penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
      
        scheduler_cos.step()            
        all_outs, all_times, all_events =[], [], []        
        model.eval()       

        if not tuning:            
            all_r_tr, all_t_tr, all_e_tr = [], [], []
            with torch.no_grad():            
                for batch in train_loader:
                    x = batch[0].to(device, non_blocking=True)
                    time = batch[1].to(device, non_blocking=True)
                    event = batch[2].to(device, non_blocking=True)
                    risk = model(x).flatten()
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
            for batch in val_loader:
                x = batch[0].to(device)
                time = batch[1].to(device)
                event = batch[2].to(device)
                risk = model(x).flatten()
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
        # is_stable, slope = is_unstable(val_cis)
        # if  is_stable and tuning:
        #     print(f"Trial stopped early due to unstable validation at epoch {epoch} with slope {slope}. Performance C-index: {ci_va} AUROC: {auroc_va} ")
        #     return 0, 0, train_cis, val_cis, train_aucs, val_aucs, epoch

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

        per_val = save_per_patient_csv(
            model = model,
            device = device,
            out_dir=hparams["plot_dir_final"],
            fold=hparams["fold"],
            modality=hparams["modality"],
            split_label="val",
            full_loader=val_loader,
            S_np=S_te
        )
    
    return final_ci, final_auroc, train_cis, val_cis, train_aucs, val_aucs, best_epoch
        
def tune_singlemodality(
    outer_args, fold,
    X_list, T_arr, E_arr, S_arr, ids_list,
    make_hparams_fn, finalize_hparams_fn,
    mode
):   
    def objective(trial):
        hparams = make_hparams_fn(trial, fold, outer_args)
        cis, aucs = [], []
        skf = StratifiedKFold(
            n_splits=outer_args["n_inner"],
            shuffle=True,
            random_state=outer_args["random_state"]
        )

        start_time = timer.time()
        epochs = []
        failed = 0
        for tr_idx, va_idx in skf.split(X_list, E_arr):
            X_tr, X_va = X_list[tr_idx], X_list[va_idx]
            T_tr, T_va = T_arr[tr_idx],  T_arr[va_idx]
            E_tr, E_va = E_arr[tr_idx],  E_arr[va_idx]
            S_tr, S_va = S_arr[tr_idx],  S_arr[va_idx]
            ids_tr = [ids_list[i] for i in tr_idx]
            ids_va = [ids_list[i] for i in va_idx]
            ci, auc, _, _, _,_, epoch = run_fold_singlemodality(
                trial,
                mode,
                X_tr, X_va,
                T_tr, E_tr, S_tr, ids_tr,
                T_va, E_va, S_va, ids_va,
                hparams,
                tuning=True
            )
            # if ci == 0.0:
            #     end_time = timer.time()
            #     elapsed = end_time - start_time   
            #     print(f" Trial completed in {elapsed:.2f} seconds")
            #     trial.set_user_attr("mean_inner_epochs", 0) 
            #     return 0.4
            epochs.append(epoch)
            cis.append(ci); aucs.append(auc)
        end_time = timer.time()
        elapsed = end_time - start_time   
        hparams["mean_inner_epochs"] = int(np.mean(epochs))
        trial.set_user_attr("mean_inner_epochs", hparams["mean_inner_epochs"])
        print(f" Trial completed in {elapsed:.2f} seconds") 
        return float(np.mean(cis)) if outer_args["tune_c_index"] else float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=outer_args["n_trials"])
    best = study.best_trial.params
    best["final_epochs"] = study.best_trial.user_attrs["mean_inner_epochs"]

    return finalize_hparams_fn(best, outer_args, fold)

def nested_cross_validate_singlemodality(
    mode,            
    X_wsi,
    X_ct,  
    ct_fixed, 
    times, events, scores, ids,
    modality_name, outer_args,
    make_hparams_fn, finalize_hparams_fn
):
    outer = StratifiedKFold(
        n_splits=outer_args["n_splits"],
        shuffle=True,
        random_state=outer_args.get("random_state", 42)
    )
    outer_cis, outer_aucs = [], []
    base_cis, base_aucs = [], []
    all_train_cis, all_val_cis = [], []
    all_train_aucs, all_val_aucs = [], []
    if mode == "wsi":
        features_list = X_wsi
    elif mode == "ct":
        features_list = X_ct

    for fold, (tr_idx, te_idx) in enumerate(outer.split(features_list, events), start=1):
        print(f"\n>>> Outer fold {fold}/{outer_args['n_splits']} <<<")
        tr_feats, te_feats = features_list[tr_idx], features_list[te_idx]
        T_tr, T_te = times[tr_idx],  times[te_idx]
        E_tr, E_te = events[tr_idx], events[te_idx]
        S_tr, S_te = scores[tr_idx], scores[te_idx]
        ids_tr     = [ids[i] for i in tr_idx]
        ids_te     = [ids[i] for i in te_idx]
        print(f"  Outer Train: {E_tr.sum()}/{len(E_tr)} event-positive")
        print(f"  Outer Test : {E_te.sum()}/{len(E_te)} event-positive")
        
        base_tr = compute_baseline_metrics(T_tr, E_tr, S_tr,
                                           outer_args["device"],
                                           outer_args["horizon"])
        base_te = compute_baseline_metrics(T_te, E_te, S_te,
                                           outer_args["device"],
                                           outer_args["horizon"])
        
        print(f"  Baseline Test CI={base_te['ci']:.4f}, AUROC={base_te['auroc']:.4f}")
        base_cis.append(base_te['ci'])
        base_aucs.append(base_te['auroc'])
        

        x = tr_feats

            
        T_tr_sub = times[tr_idx]
        E_tr_sub = events[tr_idx]
        S_tr_sub = scores[tr_idx]
        ids_tr_sub = [ids[i] for i in tr_idx]

        best_hparams = tune_singlemodality(
            outer_args, fold,
            x,     # length = len(tr_idx)
            T_tr_sub,       # length = len(tr_idx)
            E_tr_sub,       # length = len(tr_idx)
            S_tr_sub,       # length = len(tr_idx)
            ids_tr_sub,     # length = len(tr_idx)
            make_hparams_fn,
            finalize_hparams_fn,
            mode
        )
        ci_outer, auc_outer, tr_cis, val_cis, tr_aucs, val_aucs, _ = run_fold_singlemodality(
            1,
            mode,
            tr_feats, te_feats,
            T_tr, E_tr, S_tr, ids_tr,
            T_te, E_te, S_te, ids_te,
            best_hparams,
            tuning=False
        )
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


