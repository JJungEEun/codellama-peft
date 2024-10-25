CUDA_VISIBLE_DEVICES=3 python infer.py \
    --model_path codellama/CodeLlama-7b-hf\
    --adapter_path runs/checkpoints/conala/CodeLlama-7b-hf_lora \
    --tuning_method lora \
    --num_icl_examples -1
