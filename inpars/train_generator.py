import os
import numpy as np
import torch
import json
from pathlib import Path
from functools import partial
from datasets import load_dataset
from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed
)
import dspy

from ALMA.utils.cpo_trainer import CPOTrainer
from ALMA.utils.cpo_config import CPOConfig

@dataclass
class TrainGeneratorArguments:
    model: str = field(
        metadata={"help": "Model to fine-tune."},
    )
    save_dir: Path = field(
        metadata={"help": """
                  Directory to save model checkpoints and metadata to.
                    The following will be written:
                    - checkpoints/<model_name>_<step_num>.pt
                    - checkpoints/<model_name>_latest.pt
                    - train_samples.json
                  """},
    )
    use_peft: bool = field(
        metadata={"help": "Use PEFT for training."},
    )
    use_dspy: bool = field(
        metadata={"help": "Use DSPy for training."},
    )
    model_checkpoint: str = field(
        metadata={"help": "Path to checkpoint to load from."},
    )

    teacher_queries_path: str = field(
        metadata={"help": "Path to teacher queries. Expecting JSON format."},
    )
    dataset_name: str = field(
        default="msmarco",
        metadata={"help": "Dataset name."},
    )
    data_sample_size: int = field(
        metadata={"help": "Number of data samples to use for training. By default, we use all samples."},
    )
    data_samples_path: str = field(
        metadata={"help": "Path to data samples (list of doc_ids)."},
    )

    max_doc_length: int = field(
        metadata={"help": "Maximum document length."},
    )

    def __post_init__(self):
        if not self.save_dir.exists():
            raise ValueError(f"Save directory {self.save_dir} does not exist.")
        self.save_dir.mkdir(parents=True, exist_ok=True)

def peft_finetune(args: TrainGeneratorArguments, 
                  s2s_args: TrainingArguments,
                  peft_config: PeftConfig):
    # load model from checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, # TODO: **model_kwargs
        )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # TODO initialise peft. I'm not sure how to integrate dspy into this
    if not peft_config:
        raise ValueError("PEFT config not provided.")
    model = PeftModel(model, peft_config)

    # load dataset
    # with fieds ["doc_id", "text"] (would be nice to have some gt queries as well.)
    dataset = load_dataset(args.dataset_name)

    # load data sample indices
    if args.data_samples_path and os.path.exists(args.data_samples_path):
        with open(args.data_samples_path, "r") as f:
            data_samples = json.load(f)
    else:
        raise ValueError("Data samples not provided.")

    # TODO: prepare data
    # load teacher queries.
    # TODO: fix format, right now I'm assuming
    #    [
    #       {
    #           "query": str,
    #           "doc_id": str,
    #           "score": float
    #       }
    #    ]
    with open(args.teacher_queries_path, "r") as f:
        teacher_queries = json.load(f)

    # TODO: prepare training args

    # TODO: prepare trainer

    # train

    # save model

    # save metadata

@dataclass
class DSPyConfig:
    # TODO
    pass


def dspy_finetune(args: TrainGeneratorArguments,
                  dspy_config: DSPyConfig):
    # TODO
    pass
