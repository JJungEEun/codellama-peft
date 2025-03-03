import logging
import math

import torch
import wandb
from datasets import concatenate_datasets
from peft import (get_peft_model, TaskType, LoraConfig, IA3Config, PromptTuningConfig, PrefixTuningConfig)
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, 
                          TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, BitsAndBytesConfig)

from utils import *

logger = logging.getLogger(__name__)


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.best_loss = float("inf")
        self.saved_models_dir = []

    def on_step_end(self, args, state, control, model, tokenizer, **kwargs):
        if state.global_step % self.eval_steps == 0:
            evaluation_results = self.trainer.evaluate()
            eval_loss = evaluation_results["eval_loss"]

            if eval_loss < self.best_loss:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                self.best_loss = eval_loss

                if "wandb" in args.report_to:
                    wandb.run.summary["best_evaluation_loss"] = eval_loss


def load_model_and_tokenizer(args):
    model_cls = AutoModelForSeq2SeqLM if "codet5" in args.model_name_or_path else AutoModelForCausalLM
    task_type = TaskType.SEQ_2_SEQ_LM if "codet5" in args.model_name_or_path else TaskType.CAUSAL_LM

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    if args.tuning_method != "ft":
        model_kwargs["torch_dtype"] = torch.float16

    if args.tuning_method == "qlora-8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconfig
    elif args.tuning_method == "qlora-4bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)
        model_kwargs["quantization_config"] = qconfig

    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.tuning_method in ["lora", "qlora-8bit", "qlora-4bit"]:
        peft_config = LoraConfig(task_type=task_type,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_lora"],
                                 lora_dropout=args.lora_dropout,
                                 bias="none")
    elif args.tuning_method == "ia3":
        peft_config = IA3Config(task_type=task_type,
                                target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_ia3"],
                                feedforward_modules=LORA_IA3_TARGET_MODULES[args.model_name]["ff_modules"])
    elif args.tuning_method == "prompt-tuning":
        peft_config = PromptTuningConfig(task_type=task_type,
                                        num_virtual_tokens=args.prompt_num_virtual_tokens,
                                        prompt_tuning_init="TEXT",
                                        prompt_tuning_init_text="Generate Python code given a natural language "
                                                                "instruction.",
                                        tokenizer_name_or_path=args.model_name_or_path)
    elif args.tuning_method == "prefix-tuning":
        peft_config = PrefixTuningConfig(task_type=task_type,
                                        num_virtual_tokens=args.prefix_num_virtual_tokens)

    if args.tuning_method != "ft":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if "codet5" not in args.model_name_or_path:
        tokenizer.padding_side = "left"

    return model, tokenizer


def run_train(args):
    if args.dataset == "joint":
        dataset = load_conala_train_dataset()
        codealpaca = load_codealpaca_train_dataset()
        dataset["train"] = concatenate_datasets([dataset["train"], codealpaca["train"]])
        dataset["validation"] = concatenate_datasets([dataset["validation"], codealpaca["validation"]])
        dataset.shuffle(args.seed)
    else:
        dataset_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
        dataset = dataset_loading_func()
    code_column = "cmd"
    intent_column = "nl"

    model, tokenizer = load_model_and_tokenizer(args)

    def preprocess_function(example):
        """
        # we tokenize, pad and truncate the samples in the following way:
        #   <pad><pad>...### Instruction:\n<intent>\n### Answer:\n<snippet><eos>
        #
        #   - prompt tokens `<pad><pad>...<intent + \n>` are ignored in the computation of the loss (-100 labels)
        #   - `<eos>` delimits the snippet and allows the model to have more focused predictions at inference
        """
        tokenized_target = tokenizer(example[code_column],
                                    max_length=args.max_target_length - 1,
                                    truncation=True,
                                    # incoder adds eos token before the start of a sequence -> ignore
                                    add_special_tokens=False)
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]
        
        prompt = "### Instruction:\n" + example[intent_column] + "\n### Response:\n"

        max_prompt_len = (args.max_input_length + args.max_target_length) - len(tokenized_target["input_ids"])
        model_inputs = tokenizer(prompt, max_length=max_prompt_len,  padding="max_length", truncation=True)

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

        return model_inputs

    def preprocess_function_seq2seq(example):
        prompt = "Generate Python code: ### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        model_inputs = tokenizer(prompt, max_length=args.max_input_length, padding="max_length", truncation=True)
        labels = tokenizer(example[code_column], max_length=args.max_target_length, padding="max_length", truncation=True)

        labels["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenize_fn = preprocess_function_seq2seq if "codet5" in args.model_name_or_path else preprocess_function
    dataset = dataset.map(tokenize_fn,
                        num_proc=args.num_workers,
                        remove_columns=dataset["train"].column_names,
                        desc="Generating samples features.")

    n_samples = len(dataset["train"])
    n_samples_per_step = args.batch_size * args.num_gpus * args.gradient_accumulation_steps
    eval_steps = math.ceil((n_samples // n_samples_per_step) * args.ratio_samples_per_eval_step)

    training_args_cls = Seq2SeqTrainingArguments if "codet5" in args.model_name_or_path else TrainingArguments
    training_args = training_args_cls(
        output_dir=args.run_dir,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adafactor",
        logging_strategy="steps",
        logging_steps=10,
        fp16=True,
        report_to=["wandb"] if args.use_wandb else ["none"]
    )
    trainer_cls = Seq2SeqTrainer if "codet5" in args.model_name_or_path else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.add_callback(SaveBestModelCallback(trainer, eval_steps))
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss before training: {round(eval_results['eval_loss'], 4)}")
    trainer.train()
