import json
import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import PeftConfig, get_peft_model
from .utils import get_optimal_process_count

# Needs floats apparently for BCELoss to work correctly
TOKEN_FALSE, TOKEN_TRUE = 0.0, 1.0


@dataclass
class ExtraArguments:
    triples: str = field(
        default=None,
        metadata={
            "help": "Triples file containing query, positive and negative examples (TSV format)."
        },
    )
    pairs: str = field(
        default=None,
        metadata={
            "help": "File containing pairs of query, passage and label (positive/negative) in TSV format."
        },
    )
    base_model: str = field(
        default="answerdotai/ModernBERT-base",
        metadata={
            "help": "Base model to fine-tune.",
            "choices": ["answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"],
        },
    )
    max_doc_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )
    num_processes: int = field(
        default=get_optimal_process_count(),
        metadata={
            "help": "Number of processes to use for data preprocessing. By default will try to use all available CPUs."
        },
    )

    peft_config_path: str | None = field(
        default=None,
        metadata={"help": "Path to a JSON file containing arguments for PEFT."},
    )


# def split_triples(triples):
#     examples = {
#         "label": [],
#         "query": [],
#         "text": [],
#     }
#     for i in range(len(triples["query"])):
#         examples["query"].append(triples["query"][i])
#         examples["text"].append(triples["positive"][i])
#         examples["label"].append(TOKEN_TRUE)
#         examples["query"].append(triples["query"][i])
#         examples["text"].append(triples["negative"][i])
#         examples["label"].append(TOKEN_FALSE)
#     return examples


def split_triples(triples):
    if not isinstance(triples["query"], list):
        raise ValueError(f"Expected a list of dicts but got: {type(triples)}.")
    examples = {
        "text": [],
        "label": [],
    }
    # Go over the whole batch
    for i in range(len(triples["query"])):
        examples["text"].append((triples["query"][i], triples["positive"][i]))
        # Add as a
        examples["label"].append([TOKEN_TRUE])
        examples["text"].append((triples["query"][i], triples["negative"][i]))
        examples["label"].append([TOKEN_FALSE])
    return examples


def split_pairs(pairs):
    examples = {
        "label": [],
        "query": [],
        "text": [],
    }
    for i in range(len(pairs["query"])):
        examples["query"].append(pairs["query"][i])
        examples["text"].append(pairs["passage"][i])
        examples["label"].append((TOKEN_FALSE, TOKEN_TRUE)[int(pairs["label"][i])])
    return examples


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
    # training_args.evaluation_strategy = "no"
    # training_args.do_eval = False
    training_args.eval_strategy = "steps"
    training_args.do_eval = True
    training_args.eval_steps = 1000
    training_args.prediction_loss_only = True
    training_args.bf16_full_eval = True
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config.num_labels = 1
    config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
    )

    if args.peft_config_path:
        with open(args.peft_config_path) as f:
            peft_config = PeftConfig.from_peft_type(**json.load(f))
        model = get_peft_model(
            model,
            peft_config,
        )
        # print("Patched model:")
        # print(model)
        model.print_trainable_parameters()

    trainer_cls = Trainer
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=256)

    # if args.triples:
    #     dataset = load_dataset(
    #         "csv",
    #         data_files=args.triples,
    #         sep="\t",
    #         names=("query", "positive", "negative"),
    #     )
    #     dataset = dataset.map(
    #         split_triples,
    #         remove_columns=("query", "positive", "negative"),
    #         batched=True,
    #         num_proc=args.num_processes,
    #     )
    # elif args.pairs:
    #     dataset = load_dataset(
    #         "csv",
    #         data_files=args.pairs,
    #         sep="\t",
    #         names=("query", "passage", "label"),
    #     )
    #     dataset = dataset.map(
    #         split_pairs,
    #         remove_columns=("query", "passage", "passage"),
    #         batched=True,
    #         num_proc=args.num_processes,
    #     )
    # else:
    #     raise Exception("We must define a triples or a pairs file.")

    dataset_path = os.path.join(
        training_args.output_dir,
        "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
        "dataset",
    )
    if not os.path.exists(dataset_path):
        dataset = load_dataset(
            "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
            "triplet-hard",
            split="train",
        )
        dataset = dataset.map(
            split_triples,
            remove_columns=("query", "positive", "negative"),
            batched=True,
            num_proc=args.num_processes,
        )

        total_examples = None
        if training_args.max_steps > 0:
            total_examples = (
                training_args.gradient_accumulation_steps
                * training_args.per_device_train_batch_size
                * training_args.max_steps
                * torch.cuda.device_count()
            )

        if training_args.num_train_epochs > 0:
            total_examples = None

        if total_examples:
            dataset = dataset.shuffle().select(range(total_examples))

        dataset = dataset.map(
            lambda x: tokenizer(x["text"], return_tensors="np"),
            remove_columns=("text",),
            batched=True,
            desc="Tokenizing",
            num_proc=args.num_processes,
        )
        dataset.save_to_disk(dataset_path)
    else:
        dataset = load_from_disk(dataset_path, keep_in_memory=True)
    print(dataset.features)

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset.select(range(len(dataset) - 4096)),
        eval_dataset=dataset.select(range(len(dataset) - 4096 - 1, len(dataset)))
        if training_args.do_eval
        else None,
        data_collator=data_collator,
    )

    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_metrics.metrics)

    if hasattr(model, "peft_config"):
        model.save_pretrained(
            training_args.output_dir,
            path_initial_model_for_weight_conversion="pissa_init"
            if "pissa" in model.peft_config.init_lora_weights
            else None,
        )

    torch.cuda.empty_cache()
