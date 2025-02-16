import optuna
import os
import pandas as pd
import re


def get_db_storage_path(args):
    db_storage_path = f"sqlite:///{args.db_dir}/{args.dataset}-{args.model}.db"
    return db_storage_path


def match_study_name(db_storage_path):
    match = re.search(r"_(.+?)-", db_storage_path)
    if match:
        model = match.group(1)
    else:
        raise ValueError(f"Cannot parse model name from {db_storage_path}")
    return model


def load_best_hparams(model_name, db_storage_path, config):
    best_trial = get_best_trial(model_name, db_storage_path)
    best_params = best_trial.params

    for key, value in best_params.items():
        setattr(config, key, value)

    return config


def get_best_trial(model_name, db_storage_path):
    study = optuna.load_study(study_name=model_name, storage=db_storage_path)
    best_trial = study.best_trial
    return best_trial


def summarize_db_result(log_dir):
    """
    Summarize the results of all .db files in the log_dir
    """
    db_paths = []
    db_fnames = []

    # Step 1: Locate all .db files
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".db"):
                db_path = os.path.join(root, file)
                db_paths.append(db_path)
                db_fnames.append(os.path.splitext(file)[0])

    results = []
    for db_path, db_fname in zip(db_paths, db_fnames):
        storage_path = f"sqlite:///{db_path}"
        study_name = db_fname.split("-")[1]
        if "attnnet-nopf" in db_fname:
            study_name = "attnnet-nopf"
        # load the best trial
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_path)
            trials_df = study.trials_dataframe()
            trials_df = trials_df[trials_df["state"] == "COMPLETE"]
            best_trial = get_best_trial(study_name, storage_path)
            params = best_trial.params
            best_trial_idx = best_trial.number
            best_trial = trials_df.loc[best_trial_idx]
        except Exception as e:
            print(f"Error in {db_path}: {e}")
            continue
       
        result = best_trial[[idx for idx in best_trial.index if idx.startswith("user_attrs_val")]]
        result["best_trial"] = best_trial["number"] + 1
        result["fname_name"] = os.path.splitext(db_fname)[0]
        result["completed_trials"] = len(study.trials)
        for key, value in params.items():
            result[key] = value

        results.append(result)
    results = pd.concat(results, axis=1).T

    return results


if __name__ == "__main__":
    log_dir = "/home/jasonz/Code/SharpVIC-image/optuna_db"
    results = summarize_db_result(log_dir)
    results.to_csv("optuna_results.csv", index=False)

    # model_name = "adnet"
    # db_storage_path = f"sqlite:///optuna_db/ham-sex_{model_name}-resnet18-attn.db"
    # config = {}
    # load_best_hparams(model_name=model_name, db_storage_path=db_storage_path, config=config)
