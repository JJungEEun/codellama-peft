CUDA_VISIBLE_DEVICES=3 python main.py \
  --model_name_or_path codellama/CodeLlama-7b-hf \
  --adapter_path runs/checkpoints/conala/CodeLlama-7b-hf_lora \
  --tuning_method lora \
  --dataset conala \
  --do_test