import argparse

def parse_args_llm():
    parser = argparse.ArgumentParser(description="TreeKD")
    parser.add_argument("--project", type=str, default="TreeKD")
    parser.add_argument("--seed", type=int, default=0)

    # Model related
    parser.add_argument("--model_name", type=str, default='llm')
    parser.add_argument("--llm_name", type=str)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--n_gpus", type=int, default=4)

    # Model Training
    parser.add_argument("--dataset", type=str, default='generation')
    parser.add_argument("--path", type=str, default='../TreeKD-datasets')
    parser.add_argument("--task", type=str, default='DrugADMET')
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--use_rule", action='store_true', default=False)
    parser.add_argument("--rule_index", type=int, default=None)

    # Inference
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Checkpoint
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser