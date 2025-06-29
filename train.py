import os
import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model import CoxModel, cox_loss, SurvivalDataset, IntermediateFusionCox, MedicalNetCox3D, SurvivalNiftiDataset3D
from utils import plot, plot_average_metrics, save_per_patient_csv, concordance_index_gpu, compute_auroc_at_horizon_torch, compute_baseline_metrics
from hparams import (
    make_hparams_uni,
    finalize_hparams_uni,
    make_hparams_late,
    finalize_hparams_late,
    make_hparams_intermediate,
    finalize_hparams_intermediate,
    make_hparams_uni3d,
    finalize_hparams_uni3d
)
import pandas as pd

def train_and_evaluate(
    model, optimizer, scheduler_warm, scheduler_cos,
    train_loader, full_train_loader, full_val_loader,
    compute_risk_fn, cox_loss_fn,
    device, horizon, epochs, early_stop_patience,
    plot_dir=None, tuning=False
):
    best_val_ci    = 0.0
    best_val_auroc = 0.0
    best_epoch     = 0
    no_improve     = 0

    train_cis, val_cis   = [], []
    train_aucs, val_aucs = [], []

    is_late = isinstance(model, tuple)

    for epoch in range(1, epochs + 1):
        # ── (1) Set train mode ──
        if is_late:
            model[0].train()
            model[1].train()
        else:
            model.train()

        # ── (2) One epoch of optimization ──
        for batch in train_loader:
            if is_late:
                # batch = ((Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _))
                (Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _) = batch

                # Move CT branch inputs to device
                Xb_ct = Xb_ct.to(device, non_blocking=True)
                Tb_ct = Tb_ct.to(device, non_blocking=True)
                Eb_ct = Eb_ct.to(device, non_blocking=True)

                # Move WSI branch inputs to device
                Xb_wsi = Xb_wsi.to(device, non_blocking=True)
                Tb_wsi = Tb_wsi.to(device, non_blocking=True)
                Eb_wsi = Eb_wsi.to(device, non_blocking=True)

                # ---- CT step ----
                optimizer[0].zero_grad()
                r_ct = model[0](Xb_ct).squeeze(-1)            # <-- squeeze only last dim
                loss_ct = cox_loss_fn(r_ct, Tb_ct, Eb_ct)
                l1_ct = sum(p.abs().sum() for p in model[0].parameters())
                loss_ct = loss_ct + model[0].hparams.get("ct_l1_lambda", 0.0) * l1_ct
                loss_ct.backward()
                optimizer[0].step()

                # Step CT warmup / cosine schedulers
                if epoch < model[0].warmup_epochs:
                    scheduler_warm[0].step()
                else:
                    scheduler_cos[0].step()

                # ---- WSI step ----
                optimizer[1].zero_grad()
                r_wsi = model[1](Xb_wsi).squeeze(-1)           # <-- squeeze only last dim
                loss_wsi = cox_loss_fn(r_wsi, Tb_wsi, Eb_wsi)
                l1_wsi = sum(p.abs().sum() for p in model[1].parameters())
                loss_wsi = loss_wsi + model[1].hparams.get("wsi_l1_lambda", 0.0) * l1_wsi
                loss_wsi.backward()
                optimizer[1].step()

                # Step WSI warmup / cosine schedulers
                if epoch < model[1].warmup_epochs:
                    scheduler_warm[1].step()
                else:
                    scheduler_cos[1].step()

            else:
                # batch = (Xb, Tb, Eb, _)
                Xb, Tb, Eb, _ = batch

                # Move inputs to device
                Xb = Xb.to(device, non_blocking=True)
                Tb = Tb.to(device, non_blocking=True)
                Eb = Eb.to(device, non_blocking=True)

                optimizer.zero_grad()
                risk = compute_risk_fn(model, (Xb, Tb, Eb, None)).squeeze(-1)   # <-- squeeze only last dim
                loss = cox_loss_fn(risk, Tb, Eb)
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + getattr(model, "l1_lambda", 0.0) * l1_norm
                loss.backward()
                optimizer.step()

                # Step warmup / cosine schedulers
                if epoch < model.warmup_epochs:
                    scheduler_warm.step()
                else:
                    scheduler_cos.step()

        # ── (3) Compute train/val metrics in small sub‐batches ──
        if is_late:
            model[0].eval()
            model[1].eval()
        else:
            model.eval()

        # Accumulate risk‐scores, times, and events from each mini‐batch
        all_risk_tr = []
        all_T_tr    = []
        all_E_tr    = []

        with torch.no_grad():
            # — TRAIN metrics —
            for batch in full_train_loader:
                if is_late:
                    (Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _) = batch

                    Xb_ct = Xb_ct.to(device, non_blocking=True)
                    Tb_ct = Tb_ct.to(device, non_blocking=True)
                    Eb_ct = Eb_ct.to(device, non_blocking=True)
                    Xb_wsi = Xb_wsi.to(device, non_blocking=True)
                    # Tb_wsi / Eb_wsi == Tb_ct / Eb_ct anyway

                    r_ct  = model[0](Xb_ct).squeeze(-1)
                    r_wsi = model[1](Xb_wsi).squeeze(-1)
                    alpha = model[0].hparams.get("ct_alpha", 1.0)
                    r_tr_batch = alpha * r_ct + (1 - alpha) * r_wsi

                    all_risk_tr.append(r_tr_batch.cpu())
                    all_T_tr.append(Tb_ct.cpu())
                    all_E_tr.append(Eb_ct.cpu())

                else:
                    Xb_full, Tb_full, Eb_full, _ = batch
                    Xb_full = Xb_full.to(device, non_blocking=True)
                    Tb_full = Tb_full.to(device, non_blocking=True)
                    Eb_full = Eb_full.to(device, non_blocking=True)

                    r_tr_batch = compute_risk_fn(model, (Xb_full, Tb_full, Eb_full, None)).squeeze(-1)
                    all_risk_tr.append(r_tr_batch.cpu())
                    all_T_tr.append(Tb_full.cpu())
                    all_E_tr.append(Eb_full.cpu())

            # Concatenate all mini‐batches along dim=0
            all_risk_tr = torch.cat(all_risk_tr, dim=0).numpy()
            all_T_tr    = torch.cat(all_T_tr,    dim=0).numpy()
            all_E_tr    = torch.cat(all_E_tr,    dim=0).numpy()

            # Compute train‐CI and train‐AUROC
            ci_tr = concordance_index_gpu(
                risk=torch.from_numpy(all_risk_tr).to(device),
                time=torch.from_numpy(all_T_tr).to(device),
                event=torch.from_numpy(all_E_tr).to(device)
            ).item()
            auroc_tr = compute_auroc_at_horizon_torch(
                times   = torch.from_numpy(all_T_tr).to(device),
                events  = torch.from_numpy(all_E_tr).to(device),
                risks   = torch.from_numpy(all_risk_tr).to(device),
                horizon = horizon
            ).item()

            # — VAL metrics —
            all_risk_val = []
            all_T_val    = []
            all_E_val    = []

            for batch in full_val_loader:
                if is_late:
                    (Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _) = batch

                    Xb_ct = Xb_ct.to(device, non_blocking=True)
                    Tb_ct = Tb_ct.to(device, non_blocking=True)
                    Eb_ct = Eb_ct.to(device, non_blocking=True)
                    Xb_wsi = Xb_wsi.to(device, non_blocking=True)

                    r_ct  = model[0](Xb_ct).squeeze(-1)
                    r_wsi = model[1](Xb_wsi).squeeze(-1)
                    alpha = model[0].hparams.get("ct_alpha", 1.0)
                    r_va_batch = alpha * r_ct + (1 - alpha) * r_wsi

                    all_risk_val.append(r_va_batch.cpu())
                    all_T_val.append(Tb_ct.cpu())
                    all_E_val.append(Eb_ct.cpu())

                else:
                    Xb_val, Tb_val, Eb_val, _ = batch
                    Xb_val = Xb_val.to(device, non_blocking=True)
                    Tb_val = Tb_val.to(device, non_blocking=True)
                    Eb_val = Eb_val.to(device, non_blocking=True)

                    r_va_batch = compute_risk_fn(model, (Xb_val, Tb_val, Eb_val, None)).squeeze(-1)
                    all_risk_val.append(r_va_batch.cpu())
                    all_T_val.append(Tb_val.cpu())
                    all_E_val.append(Eb_val.cpu())

            # Concatenate and compute val‐CI / val‐AUROC
            all_risk_val = torch.cat(all_risk_val, dim=0).numpy()
            all_T_val    = torch.cat(all_T_val,    dim=0).numpy()
            all_E_val    = torch.cat(all_E_val,    dim=0).numpy()

            ci_va = concordance_index_gpu(
                risk  = torch.from_numpy(all_risk_val).to(device),
                time  = torch.from_numpy(all_T_val).to(device),
                event = torch.from_numpy(all_E_val).to(device)
            ).item()
            auroc_va = compute_auroc_at_horizon_torch(
                times   = torch.from_numpy(all_T_val).to(device),
                events  = torch.from_numpy(all_E_val).to(device),
                risks   = torch.from_numpy(all_risk_val).to(device),
                horizon = horizon
            ).item()

        # Record metrics for this epoch
        train_cis.append(ci_tr)
        val_cis.append(ci_va)
        train_aucs.append(auroc_tr)
        val_aucs.append(auroc_va)

        # ── (4) Early stopping & track “best” ──
        if ci_va > best_val_ci + 1e-4:
            best_val_ci    = ci_va
            best_val_auroc = auroc_va
            best_epoch     = epoch
            no_improve     = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(f"→ Early stopping at epoch {epoch}, best val CI={best_val_ci:.4f}")
            break

        # ── (5) Optional plotting ──
        if plot_dir and not tuning:
            hp = model[0].hparams if isinstance(model, tuple) else model.hparams
            plot(plot_dir, hp, epoch, train_cis, train_aucs, val_cis, val_aucs)

    # ── (6) Return everything ──
    return {
        "train_cis":       train_cis,
        "val_cis":         val_cis,
        "train_aucs":      train_aucs,
        "val_aucs":        val_aucs,
        "best_val_ci":     best_val_ci,
        "best_val_auroc":  best_val_auroc,
        "best_epoch":      best_epoch,
    }

def build_models_and_compute_risk_fn(mode, hparams):
    if mode == ["ct"] or mode == ["wsi"] or mode == ["intermediate"]:
        model = CoxModel(
            input_dim=hparams["input_dim_uni"],
            hidden_dim=hparams["hidden_dim"],
            dropout=hparams.get("dropout", 0.2),
        )
        model.hparams = hparams
        model.warmup_epochs = hparams["warmup_epochs"]
        def compute_risk(m, batch):
            Xb, Tb, Eb, _ = batch
            return m(Xb)
        return model, compute_risk

    elif mode == ["late"]:
        model_ct = CoxModel(
            input_dim=hparams["ct_input_dim"],
            hidden_dim=hparams["ct_hidden_dim"],
            dropout=hparams["ct_dropout"],
        )
        model_wsi = CoxModel(
            input_dim=hparams["wsi_input_dim"],
            hidden_dim=hparams["wsi_hidden_dim"],
            dropout=hparams["wsi_dropout"],
        )
        alpha = hparams["ct_alpha"]
        # Attach hyperparams so plot() can read them if needed:
        model_ct.hparams = hparams
        model_wsi.hparams = hparams
        model_ct.warmup_epochs = hparams["ct_warmup_epochs"]
        model_wsi.warmup_epochs = hparams["wsi_warmup_epochs"]

        def compute_risk_late(models, batch):
            (Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _) = batch
            r_ct = models[0](Xb_ct).squeeze()
            r_wsi = models[1](Xb_wsi).squeeze()
            return alpha * r_ct + (1 - alpha) * r_wsi

        return (model_ct, model_wsi), compute_risk_late

    elif mode == "specific-shared":  # ["intermediate"]
        model_int = IntermediateFusionCox(
            D_ct=hparams["ct_input_dim"],
            D_path=hparams["wsi_input_dim"],
            D_shared=hparams["D_shared"],
            D_specific=hparams["D_specific"],
            D_fusion=hparams["D_fusion"],
        )
        model_int.hparams = hparams
        model_int.warmup_epochs = hparams["warmup_epochs"]

        def compute_risk_intermediate(m, batch):
            Xb_cat, Tb, Eb, _ = batch
            D_ct = hparams["ct_input_dim"]
            ct_feat = Xb_cat[:, :D_ct]
            wsi_feat = Xb_cat[:, D_ct:]
            return m(ct_feat, wsi_feat)

        return model_int, compute_risk_intermediate

def run_fold_generic(
    modalities, tr_feats, va_feats,
    T_tr, E_tr, S_tr, ids_tr,
    T_va, E_va, S_va, ids_va,
    hparams, tuning, early_stop
):
    
    device = hparams["device"]
    horizon = hparams["horizon"]

    # ── Compute “Leibovich” baseline on TRAIN and VAL splits ──
    base_train = compute_baseline_metrics(T_tr, E_tr, S_tr, device, horizon)
    base_val   = compute_baseline_metrics(T_va, E_va, S_va, device, horizon)

    # Inject them into hparams so plot() can find them
    hparams["base_tr_ci"]    = base_train["ci"]
    hparams["base_va_ci"]    = base_val["ci"]
    hparams["base_tr_auroc5"] = base_train["auroc"]
    hparams["base_va_auroc5"] = base_val["auroc"]
   
    # 1) Standardize each modality on CPU
    scalers = [StandardScaler().fit(X) for X in tr_feats]
    tr_feats_t = [torch.tensor(s.transform(tr_feats[i]), dtype=torch.float32, device=hparams["device"])
                for i, s in enumerate(scalers)]
    va_feats_t = [torch.tensor(s.transform(va_feats[i]), dtype=torch.float32, device=hparams["device"])
                for i, s in enumerate(scalers)]

    # 2) Move T/E to GPU
    T_tr_t = torch.tensor(T_tr, dtype=torch.float32, device=hparams["device"])
    E_tr_t = torch.tensor(E_tr, dtype=torch.float32, device=hparams["device"])
    T_va_t = torch.tensor(T_va, dtype=torch.float32, device=hparams["device"])
    E_va_t = torch.tensor(E_va, dtype=torch.float32, device=hparams["device"])
    
    # 3) Build DataLoaders
    train_datasets = []
    val_datasets   = []
    for i, feat_t in enumerate(tr_feats_t):
        train_datasets.append(SurvivalDataset(feat_t, T_tr_t, E_tr_t, ids_tr))
        val_datasets.append(SurvivalDataset(va_feats_t[i], T_va_t, E_va_t, ids_va))

    if len(modalities) == 1:
        train_loader = DataLoader(train_datasets[0],
                                batch_size=hparams["batch_size"], shuffle=True, drop_last=True)
        full_train_loader = DataLoader(train_datasets[0],
                                    batch_size=len(ids_tr), shuffle=False)
        full_val_loader   = DataLoader(val_datasets[0],
                                    batch_size=len(ids_va), shuffle=False)
    else:
        train_loader = zip(
            DataLoader(train_datasets[0], batch_size=hparams["batch_size"], shuffle=True, drop_last=True),
            DataLoader(train_datasets[1], batch_size=hparams["batch_size"], shuffle=True, drop_last=True),
        )
        full_train_loader = zip(
            DataLoader(train_datasets[0], batch_size=len(ids_tr), shuffle=False),
            DataLoader(train_datasets[1], batch_size=len(ids_tr), shuffle=False),
        )
        full_val_loader = zip(
            DataLoader(val_datasets[0], batch_size=len(ids_va), shuffle=False),
            DataLoader(val_datasets[1], batch_size=len(ids_va), shuffle=False),
        )

    # 4) Build model + compute_risk_fn
    model, compute_risk_fn = build_models_and_compute_risk_fn(mode, hparams)
    if isinstance(model, tuple):
        # send each branch to GPU
        for m in model:
            m.to(hparams["device"])
    else:
        model.to(hparams["device"])
        
    # 5) Set up optimizer & schedulers (depends on “modalities”)
    if modalities == ["uni"] or modalities == ["intermediate"]:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=hparams["learning_rate"],
                                      weight_decay=hparams["weight_decay"])
        scheduler_warm = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=hparams["start_factor"],
            total_iters=hparams["warmup_epochs"]
        )
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, (hparams["epochs"] if tuning else hparams["final_epochs"]) - hparams["warmup_epochs"]),
            eta_min=hparams["eta_min"]
        )
    else:  # ["ct", "wsi"] has two separate optimizers
        optimizer_ct = torch.optim.AdamW(
            model[0].parameters(),
            lr=hparams["ct_learning_rate"],
            weight_decay=hparams["ct_weight_decay"]
        )
        scheduler_warm_ct = torch.optim.lr_scheduler.LinearLR(
            optimizer_ct,
            start_factor=hparams["ct_start_factor"],
            total_iters=hparams["ct_warmup_epochs"]
        )
        scheduler_cos_ct = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ct,
            T_max=max(1, (hparams["epochs"] if tuning else hparams["final_epochs"]) - hparams["ct_warmup_epochs"]),
            eta_min=hparams["ct_eta_min"]
        )

        optimizer_wsi = torch.optim.AdamW(
            model[1].parameters(),
            lr=hparams["wsi_learning_rate"],
            weight_decay=hparams["wsi_weight_decay"]
        )
        scheduler_warm_wsi = torch.optim.lr_scheduler.LinearLR(
            optimizer_wsi,
            start_factor=hparams["wsi_start_factor"],
            total_iters=hparams["wsi_warmup_epochs"]
        )
        scheduler_cos_wsi = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_wsi,
            T_max=max(1, (hparams["epochs"] if tuning else hparams["final_epochs"]) - hparams["wsi_warmup_epochs"]),
            eta_min=hparams["wsi_eta_min"]
        )

        # For late‐fusion, we’ll train CT and WSI branches alternately inside train_and_evaluate:
        def optimizer_step(mdl, opt_ct, opt_wsi, batch):
            (Xb_ct, Tb_ct, Eb_ct, _), (Xb_wsi, Tb_wsi, Eb_wsi, _) = batch
            # CT step
            opt_ct.zero_grad()
            r_ct = mdl[0](Xb_ct).squeeze()
            loss_ct = cox_loss(r_ct, Tb_ct, Eb_ct)
            l1_ct = sum(p.abs().sum() for p in mdl[0].parameters())
            loss_ct = loss_ct + hparams.get("ct_l1_lambda", 0.0) * l1_ct
            loss_ct.backward()
            opt_ct.step()
            # WSI step
            opt_wsi.zero_grad()
            r_wsi = mdl[1](Xb_wsi).squeeze()
            loss_wsi = cox_loss(r_wsi, Tb_wsi, Eb_wsi)
            l1_wsi = sum(p.abs().sum() for p in mdl[1].parameters())
            loss_wsi = loss_wsi + hparams.get("wsi_l1_lambda", 0.0) * l1_wsi
            loss_wsi.backward()
            opt_wsi.step()

        # Pass these four objects into train_and_evaluate instead of a single optimizer/scheduler
        optimizer = (optimizer_ct, optimizer_wsi)
        scheduler_warm = (scheduler_warm_ct, scheduler_warm_wsi)
        scheduler_cos = (scheduler_cos_ct, scheduler_cos_wsi)

    # 6) Call train_and_evaluate. For late-fusion, train_and_evaluate has to know how to “step” both optimizers.
    result = train_and_evaluate(
        model=model,
        optimizer=optimizer,
        scheduler_warm=scheduler_warm,
        scheduler_cos=scheduler_cos,
        train_loader=train_loader,
        full_train_loader=full_train_loader,
        full_val_loader=full_val_loader,
        compute_risk_fn=compute_risk_fn,
        cox_loss_fn=cox_loss,
        device=hparams["device"],
        horizon=hparams["horizon"],
        epochs=(hparams["epochs"] if tuning else hparams["final_epochs"]),
        early_stop_patience=hparams.get("early_stop_patience", 10),
        plot_dir=(hparams["plot_dir_tuning"] if tuning else hparams["plot_dir_final"]),
        tuning=tuning
    )

    # ── (7) Instead of “last‐epoch,” grab the best‐epoch metrics ──
    ci_va      = result["best_val_ci"]
    auroc_va   = result["best_val_auroc"]
    best_ep    = result["best_epoch"]
    
    # ── (8) Build per‐patient DataFrame just as before ──
    per_patient_df = pd.DataFrame([])
    if not tuning:
        print(f"→ Fold {hparams['fold']} best at epoch {best_ep}:  CI={ci_va:.4f}, AUROC5={auroc_va:.4f}")
        per_train = save_per_patient_csv(
            out_dir=hparams["plot_dir_final"],
            fold=hparams["fold"],
            modality=hparams["modality"],
            split_label="train",
            full_loader=full_train_loader,
            compute_risk_fn=lambda batch: compute_risk_fn(model, batch).squeeze(),
            S_np=S_tr
        )
        per_val = save_per_patient_csv(
            out_dir=hparams["plot_dir_final"],
            fold=hparams["fold"],
            modality=hparams["modality"],
            split_label="val",
            full_loader=full_val_loader,
            compute_risk_fn=lambda batch: compute_risk_fn(model, batch).squeeze(),
            S_np=S_va
        )
        per_patient_df = pd.concat([per_train, per_val], ignore_index=True)

    return ci_va, auroc_va, per_patient_df, result["train_cis"], result["val_cis"], result["train_aucs"], result["val_aucs"]

def tune_generic(
    outer_args, fold, inner_data, make_hparams_fn, make_final_hparams_fn
):
    def inner_objective(trial):
        
        hparams = make_hparams_fn(trial, fold, outer_args)
        cis_inner, auroc_inner = [], []
        X_list, T_arr, E_arr, S_arr, ids_list = inner_data

        inner_cv = StratifiedKFold(
            n_splits=outer_args["n_inner"],
            shuffle=True,
            random_state=outer_args.get("random_state", 42)
        )

        for i_tr, i_va in inner_cv.split(X_list[0], E_arr):
            tr_feats = [X[i_tr] for X in X_list]
            va_feats = [X[i_va] for X in X_list]
            T_tr_sub, E_tr_sub, S_tr_sub = T_arr[i_tr], E_arr[i_tr], S_arr[i_tr]
            T_va_sub, E_va_sub, S_va_sub = T_arr[i_va], E_arr[i_va], S_arr[i_va]
            ids_tr_sub = [ids_list[i] for i in i_tr]
            ids_va_sub = [ids_list[i] for i in i_va]

            if len(X_list) == 1:
                # unimodal
                ci, auroc, *_ = run_fold_generic(
                    ["uni3d"],                 # modalities
                    [tr_feats[0]],           # list of one feature‐matrix
                    [va_feats[0]],
                    T_tr_sub, E_tr_sub, S_tr_sub, ids_tr_sub,
                    T_va_sub, E_va_sub, S_va_sub, ids_va_sub,
                    hparams,                 # single‐dict from make_hparams_uni
                    True,                    # tuning
                    False                    # early_stop
                )

            elif len(X_list) == 2 and make_hparams_fn.__name__.endswith("late"):
                # late fusion
                ci, auroc, *_ = run_fold_generic(
                    modalities,
                    [tr_feats[0], tr_feats[1]],           # CT + WSI features
                    [va_feats[0], va_feats[1]],
                    T_tr_sub, E_tr_sub, S_tr_sub, ids_tr_sub,
                    T_va_sub, E_va_sub, S_va_sub, ids_va_sub,
                    hparams,                             # ← pass the entire flat dict
                    True, early_stop=False
                )

            else:
                tr_cat = np.concatenate([tr_feats[0], tr_feats[1]], axis=1)  # (n_train_sub, D_ct + D_wsi)
                va_cat = np.concatenate([va_feats[0], va_feats[1]], axis=1)
                # intermediate
                ci, auroc, *_ = run_fold_generic(
                    modalities,
                    [tr_cat],                       # a single array of shape (n_train_sub, D_ct + D_wsi)
                    [va_cat],                       # ditto for validation
                    T_tr_sub, E_tr_sub, S_tr_sub, ids_tr_sub,
                    T_va_sub, E_va_sub, S_va_sub, ids_va_sub,
                    hparams,                        # flat dict still contains "ct_input_dim" etc.
                    True,                           # tuning=True
                    early_stop=False
                )

            cis_inner.append(ci)
            auroc_inner.append(auroc)
        print(f"→ [Inner trial {trial.number}] finished at {time.strftime('%H:%M:%S')}, mean CI={mean_ci:.4f}")
        return (
            float(np.mean(cis_inner))
            if outer_args["tune_c_index"]
            else float(np.mean(auroc_inner))
        )

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(inner_objective, n_trials=outer_args["n_trials"])
    best_params = study.best_trial.params
    print(f"  Best params (fold {fold}): {best_params}")

    return make_final_hparams_fn(best_params, outer_args, fold)

def nested_cross_validate_generic(
    mode, modalities, features_list, times, events, scores, ids,
    modality_name, outer_args
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

    device = outer_args.get("device", torch.device("cpu"))
    horizon = outer_args["horizon"]

    for fold, (tr_idx, te_idx) in enumerate(outer.split(features_list[0], events), start=1):
        tr_feats = [F[tr_idx] for F in features_list]
        te_feats = [F[te_idx] for F in features_list]
        T_tr, T_te = times[tr_idx], times[te_idx]
        E_tr, E_te = events[tr_idx], events[te_idx]
        S_tr, S_te = scores[tr_idx], scores[te_idx]
        ids_tr = [ids[i] for i in tr_idx]
        ids_te = [ids[i] for i in te_idx]

        print(f"\n>>> Outer fold {fold}/{outer_args['n_splits']} <<<")
        print(f"  Outer Train: {E_tr.sum()}/{len(E_tr)} event-positive")
        print(f"  Outer Test : {E_te.sum()}/{len(E_te)} event-positive")

        baseline = compute_baseline_metrics(T_te, E_te, S_te, device, horizon)
        base_cis.append(baseline["ci"])
        base_aucs.append(baseline["auroc"])
        print(f"  Leibovich Test CI: {baseline['ci']:.4f}, AUROC5: {baseline['auroc']:.4f}")

        # Inner tuning
        print("Tuning bro2")
        inner_data = (tr_feats, T_tr, E_tr, S_tr, ids_tr)
        print("Tuning bro3")
        if mode == ["ct"]:
            make_hparams_fn = make_hparams_ct
            make_final_hparams_fn = finalize_hparams_ct
        elif mode == ["wsi"]:
            make_hparams_fn = make_hparams_wsi
            make_final_hparams_fn = finalize_hparams_wsi
        elif mode == ["late"]:
            make_hparams_fn = make_hparams_late
            make_final_hparams_fn = finalize_hparams_late
        elif mode == ["intermediate"]:
            make_hparams_fn = make_hparams_intermediate
            make_final_hparams_fn = finalize_hparams_intermediate
        elif mode == ["shared_specific"]:
            make_hparams_fn = make_hparams_shared_specific
            make_final_hparams_fn = finalize_hparams_shared_specific
        print("Tuning bro")
        best_hparams = tune_generic(outer_args, fold, inner_data, make_hparams_fn, make_final_hparams_fn)

        # Final evaluation
        if modalities == ["uni"]:
            ci_outer, auc_outer, pdf, tr_cis, val_cis, tr_aucs, val_aucs = run_fold_generic(
                modalities,                # = ["uni"]
                [tr_feats[0]],
                [te_feats[0]],
                T_tr, E_tr, S_tr, ids_tr,
                T_te, E_te, S_te, ids_te,
                best_hparams,              # single‐dict
                False,                     # tuning = False
                True                       # early_stop = True
            )
        elif modalities == ["ct", "wsi"]:
            ci_outer, auc_outer, pdf, tr_cis, val_cis, tr_aucs, val_aucs = run_fold_generic(
                modalities,                # = ["ct","wsi"]
                [tr_feats[0], tr_feats[1]],
                [te_feats[0], te_feats[1]],
                T_tr, E_tr, S_tr, ids_tr,
                T_te, E_te, S_te, ids_te,
                best_hparams,
                False,
                True
            )
        elif modalities == ["intermediate"]:
            # 1) Concatenate CT and WSI features for this fold:
            tr_cat = np.concatenate([tr_feats[0], tr_feats[1]], axis=1)
            te_cat = np.concatenate([te_feats[0], te_feats[1]], axis=1)

            # 2) Pass a single list entry to run_fold_generic:
            ci_outer, auc_outer, pdf, tr_cis, val_cis, tr_aucs, val_aucs = run_fold_generic(
                ["intermediate"],
                [tr_cat],             # one element: (n_train, D_ct + D_wsi)
                [te_cat],             # one element: (n_test,  D_ct + D_wsi)
                T_tr, E_tr, S_tr, ids_tr,
                T_te, E_te, S_te, ids_te,
                best_hparams,         # contains "ct_input_dim" and "wsi_input_dim"
                False,                # tuning=False
                True                  # early_stop=True
            )

        print(f"→ Fold {fold} C-index: {ci_outer:.4f}, AUROC5: {auc_outer:.4f}")
        outer_cis.append(ci_outer); outer_aucs.append(auc_outer)
        all_train_cis.append(tr_cis)
        all_val_cis.append(val_cis)
        all_train_aucs.append(tr_aucs)
        all_val_aucs.append(val_aucs)

    all_train_cis = np.stack(all_train_cis)
    all_val_cis   = np.stack(all_val_cis)
    all_train_aucs= np.stack(all_train_aucs)
    all_val_aucs  = np.stack(all_val_aucs)

    mean_train_cis = all_train_cis.mean(axis=0)
    std_train_cis  = all_train_cis.std(axis=0)
    mean_val_cis   = all_val_cis.mean(axis=0)
    std_val_cis    = all_val_cis.std(axis=0)
    mean_train_aucs= all_train_aucs.mean(axis=0)
    std_train_aucs = all_train_aucs.std(axis=0)
    mean_val_aucs  = all_val_aucs.mean(axis=0)
    std_val_aucs   = all_val_aucs.std(axis=0)

    mean_base_ci  = np.mean(base_cis)
    std_base_ci   = np.std(base_cis)
    mean_base_auc = np.mean(base_aucs)
    std_base_auc  = np.std(base_aucs)

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

    return (np.mean(outer_cis), np.std(outer_cis)), (np.mean(outer_aucs), np.std(outer_aucs))