import logging
import os
import random
import re

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import \
    AutoModelForCausalLM, \
    AutoModelForSeq2SeqLM, \
    AutoTokenizer, \
    default_data_collator, \
    StoppingCriteriaList, \
    StoppingCriteria, BitsAndBytesConfig

from utils import *

logger = logging.getLogger(__name__)
EOF_STRINGS_CONALA = ["<|endoftext|>", "</s>", "\n"]
EOF_STRINGS_CODEALPACA = ["<|endoftext|>", "</s>"]


def load_model_and_tokenizer(args):
    model_cls = AutoModelForSeq2SeqLM if "codet5" in args.model_name_or_path else AutoModelForCausalLM
    model_kwargs = {"trust_remote_code": True}

    if args.tuning_method != "ft" or args.num_icl_examples > -1:
        model_kwargs["torch_dtype"] = torch.float16

    if args.tuning_method == "qlora-8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconfig
    elif args.tuning_method == "qlora-4bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs["quantization_config"] = qconfig

    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = True
    if args.tuning_method != "ft":
        model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
        model.print_trainable_parameters()
    else:
        model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

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


def run_test(args):
    dataset_loading_func = globals().get(f"load_{args.dataset}_test_dataset")
    test_dataset = dataset_loading_func()
    intent_column = "nl"
    code_column = "cmd"

    model, tokenizer = load_model_and_tokenizer(args)

    if args.num_icl_examples >= 0:
        # zero-shot learning
        icl_prompt = "Generate Python code given a natural language instruction."
        if args.num_icl_examples > 0:
            train_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
            train_dataset = train_loading_func()["train"]
            random_indices = random.sample(range(len(train_dataset)), args.num_icl_examples)
            icl_examples = train_dataset.select(random_indices)
            for n in icl_examples:
                icl_prompt += f"\n### Instruction:\n{n[intent_column]}\
                                \n### Response:\n{n[code_column]}\n"

    def preprocess_function(example):
        prompt = "\n### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        if args.num_icl_examples >= 0:
            prompt = icl_prompt + prompt
        # no need to pad/truncate, we do not do batched generation
        if "codet5" in args.model_name_or_path:
            prompt += "<extra_id_0>"
        model_inputs = tokenizer(prompt)

        labels = tokenizer(example[code_column])["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs

    test_dataset = test_dataset.map(preprocess_function,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "labels"]],
                                    desc="Generating samples features.")
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=default_data_collator, pin_memory=True)

    eof_string = EOF_STRINGS_CONALA if args.dataset == "conala" else EOF_STRINGS_CODEALPACA
    predictions = [[] for _ in range(len(test_dataset))]
    references = []
    for step, sample in tqdm(enumerate(dataloader), total=len(test_dataset)):
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=sample["input_ids"].to(args.device),
                num_beams=10,
                num_return_sequences=10,
                max_new_tokens=args.max_target_length,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], eof_string, tokenizer)]
                )
            )
            generated_sequences = generated_sequences.detach().cpu().numpy()
            if "codet5" not in args.model_name_or_path:
                generated_sequences = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, new_tokens in zip([step] * args.num_return_sequences, generated_sequences):
                new_tokens_decoded = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                new_tokens_decoded = re.split("(%s)" % "|".join(eof_string), new_tokens_decoded.strip())[0]
                new_tokens_decoded = new_tokens_decoded.replace("\n", " ").replace("\t", " ")
                predictions[task].append(new_tokens_decoded)

            reference_decoded = tokenizer.decode(sample["labels"][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            reference_decoded = reference_decoded.replace("\n", " ").replace("\t", " ")
            references.append(reference_decoded)

    # export top-10 predictions
    jsonl_data = []
    for preds, refs in zip(predictions, references):
        jsonl_data.append({
            "predictions": preds,
            "references": refs
        })
    logger.info(f"Exporting test predictions in directory {args.run_dir}.")
    base_fname = f"output_{args.dataset}"
    if args.num_icl_examples > -1:
        base_fname += f"_{args.num_icl_examples}shot"
    with open(os.path.join(args.run_dir, f"{base_fname}.jsonl"), "w", encoding="utf-8") as fout:
        for entry in jsonl_data:
            json.dump(entry, fout)
            fout.write("\n")

    # export top-1 predictions for CodeBLEU
    predictions = [p[0] for p in predictions]
    base_pred_fname = f"predictions_{args.dataset}"
    base_ref_fname = f"references_{args.dataset}"
    if args.num_icl_examples > -1:
        base_pred_fname += f"_{args.num_icl_examples}shot"
        base_ref_fname += f"_{args.num_icl_examples}shot"
    with open(os.path.join(args.run_dir, f"{base_pred_fname}.txt"), "w", encoding="utf-8") as fpred, \
            open(os.path.join(args.run_dir, f"{base_ref_fname}.txt"), "w", encoding="utf-8") as fref:
        for prediction, reference in zip(predictions, references):
            fpred.write(prediction + "\n")
            fref.write(reference + "\n")
