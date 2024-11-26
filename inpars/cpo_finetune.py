import os
import numpy as np
import torch
import json
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_llmt_dataset
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
from inpars.cpo_dataset import load_llmt_dataset

@dataclass
class ExtraArguments:
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
        default="msmarco-document",
        metadata={"help": "Dataset name."},
    )
    dataset_location: str = field(
        metadata={"help": "Dataset location."},
    )

    max_doc_length: int = field(
        metadata={"help": "Maximum document length."},
    )

    def __post_init__(self):
        if not self.save_dir.exists():
            raise ValueError(f"Save directory {self.save_dir} does not exist.")
        self.save_dir.mkdir(parents=True, exist_ok=True)

def peft_finetune(args: ExtraArguments, 
                  cpo_config: CPOConfig,
                  peft_config: PeftConfig):
    # load model from checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        PeftModel.from_pretrained(args.model_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, # TODO: **model_kwargs
        )
        model = PeftModel(model, peft_config)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load dataset
    # with fieds ["doc_id", "text"] (would be nice to have some gt queries as well.)
    dataset = load_llmt_dataset(args.dataset_location, args.dataset_name)

    # prepare data collator

    # TODO: prepare trainer

    # train

    # save model

    # save metadata

@dataclass
class DSPyConfig:
    # TODO
    pass


def dspy_finetune(args: ExtraArguments,
                  dspy_config: DSPyConfig):
    # TODO
    pass
