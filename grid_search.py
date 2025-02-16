import optuna
from functools import partial
import argparse
from pathlib import Path

from algo_fn.objectives import cls_objective, seg_objective
from utils.misc import set_random_seed
from utils.hparams import get_db_storage_path
from configs.grid_search_configs import ClsSearchSpaceConfigs, SegSearchSpaceConfigs


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna Search')
    parser.add_argument('--dataset', type=str, default='ham', choices=["ham", "chestx"], help='dataset')
    parser.add_argument("--task", type=str, default="seg", choices=["cls", "seg"])
    parser.add_argument("--model", type=str, default="unet",choices=["resnet", "densenet", "unet"])
    parser.add_argument("--pretrained", action="store_true", help="whether to use pretrained model")
    parser.add_argument('--db_dir', type=str, default='optuna_db', help='db_save_dir')
    parser.add_argument("--n_trials", type=int, default=10, help="number of trials")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    Path(args.db_dir).mkdir(exist_ok=True, parents=True)
    db_path = get_db_storage_path(args)

    # load and set search space
    if args.task == "cls":
        search_space_configs = ClsSearchSpaceConfigs()
    elif args.task == "seg":
        search_space_configs = SegSearchSpaceConfigs()
    search_space = getattr(search_space_configs, f"search_space")
    grid_search_sampler = optuna.samplers.GridSampler(search_space)

    # Define the objective function to be maximized.
    if args.task == "cls":
        objective_with_param = partial(cls_objective, args=args)
    elif args.task == "seg":
        objective_with_param = partial(seg_objective, args=args)

    # create study
    study = optuna.create_study(direction="minimize",storage=db_path,
                                sampler=grid_search_sampler, study_name=args.model, load_if_exists=True)
    
    # optimize
    study.optimize(objective_with_param, n_trials=args.n_trials)
