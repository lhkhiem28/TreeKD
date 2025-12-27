# python train_sft.py --llm_name 'gemma-2-2b' --batch_size 2 --run_name "sft" --gamma 1.0
# python test.py      --llm_name 'gemma-2-2b' --batch_size 2 --run_name "sft" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_sft.pth
# python train_sft.py --llm_name 'granite-3.3-2b' --batch_size 2 --run_name "sft" --gamma 1.0
# python test.py      --llm_name 'granite-3.3-2b' --batch_size 2 --run_name "sft" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_sft.pth
python train_sft.py --llm_name 'gemma-2-2b' --use_rule --rule_index 0 --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0
python test.py      --llm_name 'gemma-2-2b' --use_rule --rule_index 0 --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_sft_fgs_rule.pth
python train_sft.py --llm_name 'granite-3.3-2b' --use_rule --rule_index 0 --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0
python test.py      --llm_name 'granite-3.3-2b' --use_rule --rule_index 0 --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_sft_fgs_rule.pth