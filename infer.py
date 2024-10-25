import logging
import re
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
from utils import *
from accelerate import Accelerator
import time

logger = logging.getLogger(__name__)
    

EOF_STRINGS_CONALA = ["<|endoftext|>", "</s>", "\n"]
EOF_STRINGS_CODEALPACA = ["<|endoftext|>", "</s>"]

def load_model_and_tokenizer(args):
    model_cls = AutoModelForSeq2SeqLM if "codet5" in args.model_path else AutoModelForCausalLM
    model_kwargs = {"trust_remote_code": True}

    if args.tuning_method != "ft" or args.num_icl_examples > -1:
        model_kwargs["torch_dtype"] = torch.float16

    if args.tuning_method == "qlora-8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconfig
    elif args.tuning_method == "qlora-4bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs["quantization_config"] = qconfig

    model = model_cls.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = True
    if args.tuning_method != "ft":
        model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
        model.print_trainable_parameters()
    else:
        model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)

def generate_code(args):
    model, tokenizer = load_model_and_tokenizer(args)
    
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    # intent_text = "Write a SQL query function to find students who match a major"
    intent_text = "# User study task #Encrypt the user resident registration numbers # Write your program requirements: "

    prompt = "\n### Instruction:\n" + intent_text + "\n### Response:\n"
    model_inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    eof_string = EOF_STRINGS_CONALA if args.dataset == "conala" else EOF_STRINGS_CODEALPACA
    
    start_time = time.time()

    
    with torch.no_grad():
        generated_sequences = model.generate(
                    input_ids=model_inputs["input_ids"].to(args.device),
                    num_beams=10,
                    num_return_sequences=1,
                    max_new_tokens=args.max_target_length,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [EndOfFunctionCriteria(model_inputs["input_ids"].shape[1], eof_string, tokenizer)]
                    )
                )

        end_time = time.time()

        generated_sequences = generated_sequences.detach().cpu().numpy()
        generated_sequences = generated_sequences[:, model_inputs["input_ids"].shape[1]:][0]
        
        new_tokens_decoded = tokenizer.decode(generated_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        new_tokens_decoded = re.split("(%s)" % "|".join(eof_string), new_tokens_decoded.strip())[0]
        new_tokens_decoded = new_tokens_decoded.replace("\n", " ").replace("\t", " ")

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        return new_tokens_decoded
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate code using a pre-trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the adapter checkpoint')
    parser.add_argument('--tuning_method', type=str, required=True, choices=['ft', 'lora', 'qlora-4bit', 'qlora-8bit'], help='Tuning method to use')
    parser.add_argument('--num_icl_examples', type=int, default=-1, help='Number of in-context learning examples')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset", default="conala", type=str, help="Dataset on which to fine-tune the model.")
    
    args = parser.parse_args()

    if args.dataset == "conala":
        args.max_input_length = 64
        args.max_target_length = 64
    else:
        args.max_input_length = 64
        args.max_target_length = 128

    
    generated_code = generate_code(args)
    print("Generated Code:\n", generated_code)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_path = "codellama/CodeLlama-7b-hf"
    # args.adapter_path = "runs/checkpoints/conala/CodeLlama-7b-hf_lora"
    # args.tuning_method = "lora"
    # args.num_icl_examples = -1

