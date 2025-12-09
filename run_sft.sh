python train_sft.py --llm_name 'gemma-2-2b' --use_rule --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0
for i in {0..99}; do
    echo "Running gemma-2-2b, rule_index = $i"
    python test.py      --llm_name 'gemma-2-2b' --use_rule --rule_index $i --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_sft_fgs_rule.pth
done
python train_sft.py --llm_name 'granite-3.3-2b' --use_rule --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0
for i in {0..99}; do
    echo "Running granite-3.3-2b, rule_index = $i"
    python test.py      --llm_name 'granite-3.3-2b' --use_rule --rule_index $i --batch_size 2 --run_name "sft_fgs_rule" --gamma 1.0 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_sft_fgs_rule.pth
done