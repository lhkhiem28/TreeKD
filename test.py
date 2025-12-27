import os
import tqdm
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings("ignore")

from source.config import parse_args_llm
from source.utils.help_funcs import seed_everything
from source.utils.help_funcs import collate_fn
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.datasets import *
from source.models import *
from source.utils.evaluation import *

import wandb

tasks = [
    ("Caco-2", "regression"),
    ("HIA", "classification"),
    ("Pgp", "classification"),
    ("Bioavailability", "classification"),
    ("Lipophilicity", "regression"),
    ("Solubility", "regression"),
    ("BBB", "classification"),
    ("PPBR", "regression"),
    ("VDss", "regression"),
    ("CYP2D6 inhibition", "classification"),
    ("CYP3A4 inhibition", "classification"),
    ("CYP2C9 inhibition", "classification"),
    ("CYP2D6 substrate", "classification"),
    ("CYP3A4 substrate", "classification"),
    ("CYP2C9 substrate", "classification"),
    ("Half life", "regression"),
    ("Clearance microsome", "regression"),
    ("Clearance hepatocyte", "regression"),
    ("hERG", "classification"),
    ("Ames", "classification"),
    ("DILI", "classification"),
    ("LD50", "regression"),
]

def main(args):
    seed = args.seed
    seed_everything(seed=seed)

    for task, tasktype in tasks:
        # Step 1: Build dataset
        test_dataset = load_dataset[args.dataset](path = args.path, task = task, split = "test", use_rule = args.use_rule, rule_index = args.rule_index)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)

        # Step 2: Build model
        args.llm_path = get_llm_path[args.llm_name]
        model = load_model[args.model_name](args=args)
        if args.checkpoint_path is not None:
            model = _reload_model(model, args.checkpoint_path)

        # Step 3: Evaluating
        model.eval()
        eval_outputs = []
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                eval_outputs.append(output)

        # Step 4: Post-processing & report
        scores = eval_funcs[args.dataset](eval_outputs, tasktype, args.checkpoint_path)
        if tasktype == "regression":
            print("MAE: {:.3f} | Spearman: {:.3f} (Validity: {:.2f}%)".format(
                *scores
            ))
        elif tasktype == "classification":
            print("AUROC: {:.3f} | AUPRC: {:.3f} (Validity: {:.2f}%)".format(
                *scores
            ))

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    main(args)