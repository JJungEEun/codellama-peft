import logging
import os
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria

logger = logging.getLogger(__name__)

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
    
def load_model_and_tokenizer(model_path, adapter_path, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval() 
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate_code(prompt, model, tokenizer, device='cuda'):
    """
    Generate a single piece of code from the prompt using the specified model and tokenizer.
    """

    eof_string = ["<|endoftext|>", "</s>"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_length = inputs["input_ids"].shape[1] + 100  # Adding 100 tokens space for generation

    with torch.no_grad():
        generated_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            num_beams=2,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([EndOfFunctionCriteria(inputs["input_ids"].shape[1], eof_string, tokenizer)])
        )

    generated_sequences = generated_sequences[:, inputs["input_ids"].shape[1]:][0]
    generated_code = tokenizer.decode(generated_sequences, skip_special_tokens=True)
    
    return generated_code

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "codellama/CodeLlama-7b-hf"
    adapter_path = "runs/checkpoints/conala/CodeLlama-7b-hf_lora"
    model, tokenizer = load_model_and_tokenizer(model_path, adapter_path, device)
    prompt = "Write a function to calculate the standard deviation of data points in Python"
    generated_code = generate_code(prompt, model, tokenizer, device)
    
    print("Generated Code:\n", generated_code)
