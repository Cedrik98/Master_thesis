def make_hparams_wsi(trial, fold, outer_args):
    return {
        "hidden_dim":     trial.suggest_categorical("hidden_dim", [16, 32, 64,128,256]),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "l1_lambda":      trial.suggest_float("l1_lambda", 1e-7, 1e-2, log=True),     
        "eta_min":        trial.suggest_float("eta_min", 1e-8, 1e-6, log=True),  
        # carry over any fixed keys from outer_args:
        "start_early_stop": outer_args["start_early_stop"],
        "device":         outer_args["device"],
        "epochs":         outer_args["epochs"],
        "batch_size":     outer_args["batch_size"],
        "early_stop_patience": outer_args.get("early_stop_patience", 10),
        "plot_dir_tuning":    outer_args["plot_dir_tuning"],
        "plot_dir_final":     outer_args["plot_dir_final"],
        "horizon":       outer_args["horizon"],
        "modality":      outer_args["modality"],
        "state_dict":    outer_args["state_dict"],
        "resnet":        outer_args["resnet"],
        "fold":          fold,
    }


def make_hparams_uni(trial, fold, outer_args):
    return {
        "hidden_dim":     trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
        "l1_lambda":      trial.suggest_float("l1_lambda", 1e-3, 1e-2, log=True),
        "backbone_lr":    trial.suggest_float("backbone_lr", 1e-6, 1e-3),             
        "eta_min":        trial.suggest_float("eta_min", 1e-8, 1e-7, log=True),  
        # carry over any fixed keys from outer_args:
        "start_early_stop": outer_args["start_early_stop"],
        "device":         outer_args["device"],
        "epochs":         outer_args["epochs"],
        "batch_size":     outer_args["batch_size"],
        "early_stop_patience": outer_args.get("early_stop_patience", 10),
        "plot_dir_tuning":    outer_args["plot_dir_tuning"],
        "plot_dir_final":     outer_args["plot_dir_final"],
        "horizon":       outer_args["horizon"],
        "modality":      outer_args["modality"],
        "state_dict":    outer_args["state_dict"],
        "resnet":        outer_args["resnet"],
        "fold":          fold,
    }

def make_hparams_concat(trial, fold, outer_args):
    return {
        "hidden_dim":     trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "l1_lambda":      trial.suggest_float("l1_lambda", 1e-7, 1e-4, log=True),
        "eta_min":        trial.suggest_float("eta_min", 1e-8, 1e-5, log=True),  
        "backbone_lr":    trial.suggest_float("backbone_lr", 1e-6, 1e-3, log=True),
        "dropout":        trial.suggest_float("dropout", 0.2, 0.5, log=True),
        # carry over any fixed keys from outer_args:
        "start_early_stop": outer_args["start_early_stop"],
        "device":         outer_args["device"],
        "epochs":         outer_args["epochs"],
        "batch_size":     outer_args["batch_size"],
        "early_stop_patience": outer_args.get("early_stop_patience", 10),
        "plot_dir_tuning":    outer_args["plot_dir_tuning"],
        "plot_dir_final":     outer_args["plot_dir_final"],
        "horizon":       outer_args["horizon"],
        "modality":      outer_args["modality"],
        "state_dict":    outer_args["state_dict"],
        "resnet":        outer_args["resnet"],
        "fold":          fold,
    }

def finalize_hparams_uni(best_params, outer_args, fold):
    # best_params is just the subset you suggested; combine with the fixed outer_args
    final = {
        **outer_args,
        **best_params,
        "fold": fold,
        # ensure we still carry the “final_epochs” key, etc.
    }
    return final

def make_hparams_late(trial, fold, outer_args):
    # sample all “ct_…” and “wsi_…” hyperparameters in one flat dict
    return {
        # CT side
        "ct_hidden_dim":     trial.suggest_categorical("ct_hidden_dim", [32, 64, 128, 256]),
        "ct_learning_rate":  trial.suggest_float("ct_learning_rate",  1e-5, 1e-3, log=True),
        "ct_weight_decay":   trial.suggest_float("ct_weight_decay",   1e-5, 1e-2, log=True),        
        "ct_l1_lambda":      trial.suggest_float("ct_l1_lambda", 1e-5, 1e-2, log=True),
        "ct_alpha":          trial.suggest_float("ct_alpha",          0.0,  1.0),
        "ct_eta_min":        trial.suggest_float("ct_eta_min", 1e-8, 1e-5, log=True),  
        "ct_backbone_lr":    trial.suggest_float("ct_backbone_lr", 1e-6, 1e-3),    
        # WSI side
        "wsi_hidden_dim":    trial.suggest_categorical("wsi_hidden_dim", [32, 64, 128, 256]),
        "wsi_learning_rate": trial.suggest_float("wsi_learning_rate", 1e-5, 1e-3, log=True),
        "wsi_weight_decay":  trial.suggest_float("wsi_weight_decay",  1e-5, 1e-2, log=True),
        "wsi_eta_min":        trial.suggest_float("wsi_eta_min", 1e-8, 1e-5, log=True), 
        "wsi_l1_lambda":      trial.suggest_float("wsi_l1_lambda", 1e-5, 1e-2, log=True),
        # everything from outer_args that run_fold_generic (and plot) will need:
        "start_early_stop": outer_args["start_early_stop"],
        "device":             outer_args["device"],
        "epochs":             outer_args["epochs"],
        "batch_size":         outer_args["batch_size"],
        "early_stop_patience": outer_args.get("early_stop_patience", 10),
        "plot_dir_tuning":    outer_args["plot_dir_tuning"],
        "plot_dir_final":     outer_args["plot_dir_final"],
        "horizon":            outer_args["horizon"],
        "modality":           outer_args["modality"],
        "state_dict":         outer_args["state_dict"],
        "resnet":             outer_args["resnet"],
        "fold":               fold,
    }


def finalize_hparams_late(best_params, outer_args, fold):

    final = {}

    # 1) Copy over every key that starts with "ct_":
    for k, v in best_params.items():
        if k.startswith("ct_"):
            final[k] = v

    # 2) Copy over every key that starts with "wsi_":
    for k, v in best_params.items():
        if k.startswith("wsi_"):
            final[k] = v

    # 3) Also ensure that run_fold_generic (and plot()) sees the fields it expects:
    final["device"]   = outer_args["device"]
    final["fold"]     = fold
    final["modality"] = outer_args["modality"]

    # 5) Make sure any other fixed outer_args fields (like epochs, batch_size) are present:
    final["start_early_stop"]   = outer_args["start_early_stop"]
    final["epochs"]             = outer_args["epochs"]
    final["batch_size"]         = outer_args["batch_size"]
    final["early_stop_patience"] = outer_args.get("early_stop_patience", 10)
    final["plot_dir_tuning"]    = outer_args["plot_dir_tuning"]
    final["plot_dir_final"]     = outer_args["plot_dir_final"]
    final["horizon"]            = outer_args["horizon"]
    final["state_dict"]         = outer_args["state_dict"]
    final["resnet"]             = outer_args["resnet"]

    return final

