
import torch
from functools import partial
from datasets import load_dataset
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
        default="ModernBERT-base",
        choices=["ModernBERT-base", "ModernBERT-large"],
        metadata={"help": "Base model to fine-tune."},
    )
    max_doc_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )


def split_triples(triples):
    examples = {
        "label": [],
        "query": [],
        "text": [],
    }
    for i in range(len(triples["query"])):
        examples["query"].append(triples["query"][i])
        examples["text"].append(triples["positive"][i])
        examples["label"].append(token_true)
        examples["query"].append(triples["query"][i])
        examples["text"].append(triples["negative"][i])
        examples["label"].append(token_false)
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
        examples["label"].append((token_false, token_true)[int(pairs["label"][i])])
    return examples

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "no"
    training_args.do_eval = False
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config.num_labels = 2
    config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
    )
    trainer_cls = Trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    token_false, token_true = [0, 1]

    if args.triples:
        dataset = load_dataset(
            "csv",
            data_files=args.triples,
            sep="\t",
            names=("query", "positive", "negative"),
        )
        dataset = dataset.map(
            split_triples,
            remove_columns=("query", "positive", "negative"),
            batched=True,
        )
    elif args.pairs:
        dataset = load_dataset(
            "csv",
            data_files=args.pairs,
            sep="\t",
            names=("query", "passage", "label"),
        )
        dataset = dataset.map(
            split_pairs,
            remove_columns=("query", "passage", "passage"),
            batched=True,
        )
    else:
        raise Exception("We must define a triples or a pairs file.")

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
        dataset["train"] = dataset["train"].shuffle().select(range(total_examples))

    dataset = dataset.map(
        tokenizer,
        remove_columns=("text", "label"),
        batched=True,
        desc="Tokenizing",
    )

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_metrics.metrics)

    torch.cuda.empty_cache()
