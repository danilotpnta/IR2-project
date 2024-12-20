from typing import Literal, Optional
import os
import sys
import json
import logging
import torch
from dataclasses import dataclass, field
from peft import PeftConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainerCallback,
    HfArgumentParser,
    set_seed,
)
from trl import CPOTrainer, CPOConfig
import bitsandbytes as bnb
from unsloth import FastLanguageModel

from inpars.cpo_dataset import load_cpo_dataset, DataConfig

from .utils import count_parameters


# Custom TrainerCallback to open the model, save merged model and push to hub
class CustomCallback(TrainerCallback):
    def __init__(self, model_args, model, tokenizer):
        self.model_args = model_args
        self.model = model
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, **kwargs):
        if args.do_train and args.local_rank in [-1, 0] and args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
            model_path = os.path.join(args.output_dir, "model" if self.model_args.save_merged else "adapter_model")
            save_method = f"merged_{self.model_args.save_precision}" if self.model_args.save_merged else "lora"
            if self.model_args.save_merged and self.model_args.save_precision == '4bit':
                save_method += '_forced'
            self.model.save_pretrained_merged(
                model_path,
                self.tokenizer,
                save_method=save_method,
            )
            logger.info(f"Model saved at {model_path}")
            if self.model_args.repo_id:
                self.model.push_to_hub_merged(
                    self.model_args.repo_id,
                    self.tokenizer,
                    save_method=save_method,
                )
                logger.info(f"Model pushed to hub as {self.model_args.repo_id}")                
            exit()

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_kwargs: dict = field(
        metadata={"help": "keyword arguments to pass to the model when loading"}
    )
    use_wandb: bool = field(default=False, metadata={"help": "Use wandb for logging"})
    peft_config_path: str = field(
        default=None, metadata={"help": "Path to peft config file"}
    )
    save_merged: bool = field(
        default=False,
        metadata={"help": "Save merged model. If False, only the adapter model is saved"}
    )
    save_precision: Literal["4bit", "16bit"] = field(
        default="fp16",
        metadata={"help": "Precision to save the model in"}
    )
    repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface repo ID to use. If None, we don't push"}
    )

    def __post_init__(self):
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        _, self.tokenizer = FastLanguageModel.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # To change the embedding matrix directly
            # self.tokenizer.pad_token_id = self.tokenizer.vocab_size + 1

        if self.use_wandb:
            self.wandb_api_key = os.getenv("WANDB_API_KEY")
            if self.wandb_api_key is None:
                raise ValueError("WANDB_API_KEY is not set")

def train(
    model_args: ModelArguments,
    data_config: DataConfig,
    cpo_config: CPOConfig,
    peft_config: PeftConfig,
):
    # set seed
    set_seed(cpo_config.seed)
    # init wandb
    if model_args.use_wandb:
        import wandb

        wandb.login(key=model_args.wandb_api_key)
        wandb.init(project="cpo_finetune")
        wandb.config.update(model_args)
        wandb.config.update(data_config)
        wandb.config.update(cpo_config)
        wandb.config.update(peft_config)
        cpo_config.report_to = ["wandb"]

    # load dataset
    train_dataset, eval_dataset, test_dataset = load_cpo_dataset(
        data_config, cpo_config, model_args.tokenizer
    )

    # avoid passive-agressive warning message
    cpo_config.remove_unused_columns = False

    # load model
    model, _ = FastLanguageModel.from_pretrained(
        model_args.model_name_or_path,
        **model_args.model_kwargs,
    )

    # Patch the model to have a pad token
    model.config.pad_token_id = model_args.tokenizer.pad_token_id

    # To modify the embedding matrix directly
    # input_embed = model.get_input_embeddings()
    # input_embed.weight = torch.nn.Parameter(torch.cat([input_embed.weight, torch.zeros(1, model_args.config.hidden_size).to(input_embed.weight)], dim=0))
    # input_embed.padding_idx = model_args.tokenizer.pad_token_id

    # Convert to dict
    peft_config = peft_config.to_dict()
    # Default arguments for Unsloth
    peft_config["target_modules"] = peft_config["target_modules"] or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # TODO: Figure out a better patch than this.
    del peft_config["task_type"]

    model = FastLanguageModel.get_peft_model(model, **peft_config)
    # 8-bit optimizer to deal with strained memory requirements
    optimizer = bnb.optim.Lion8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cpo_config.learning_rate / 3,
        betas=(cpo_config.adam_beta1, cpo_config.adam_beta2),
        weight_decay=cpo_config.weight_decay,
    )

    # Initialize a LearningRate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=cpo_config.learning_rate / 300,
        T_max=(
            cpo_config.num_train_epochs
            * len(train_dataset)
            // cpo_config.train_batch_size
        ),
    )

    # TODO: prepare trainer
    cpo_config.auto_find_batch_size = True
    cpo_config.dataset_num_proc = data_config.preprocessing_num_workers
    trainer = CPOTrainer(
        model=model,
        tokenizer=model_args.tokenizer,
        # peft_config=peft_config,
        args=cpo_config,
        optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        # callbacks=[CustomCallback(model_args, model, model_args.tokenizer)],
    )
    logger.info(
        f"Number of total parameters: {count_parameters(model, lambda _: True)/10**6:0.1f}M"
    )
    logger.info(
        f"Number of trainable parameters: {count_parameters(model)/10**6:0.1f}M"
    )
    # train
    trainer.train(resume_from_checkpoint=cpo_config.resume_from_checkpoint)

    # save model
    model_path = os.path.join(cpo_config.output_dir, "model" if model_args.save_merged else "adapter_model")
    save_method = f"merged_{model_args.save_precision}" if model_args.save_merged else "lora"
    if model_args.save_merged and model_args.save_precision == '4bit':
        save_method += '_forced'
    model.save_pretrained_merged(
        model_path,
        model_args.tokenizer,
        save_method=save_method,
    )
    logger.info(f"Model saved at {model_path}")
    if model_args.repo_id:
        model.push_to_hub_merged(
            model_args.repo_id,
            model_args.tokenizer,
            save_method=save_method,
        )
        logger.info(f"Model pushed to hub as {model_args.repo_id}")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, CPOConfig, DataConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # NOTE: stole it from ALMA/run_cpo_llmmt.py
        model_args, training_config, data_config = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_config, data_config = parser.parse_args_into_dataclasses()

    with open(model_args.peft_config_path) as f:
        peft_config = PeftConfig.from_peft_type(**json.load(f))

    # logging
    logging.basicConfig(level=logging.INFO)

    train(
        model_args=model_args,
        data_config=data_config,
        cpo_config=training_config,
        peft_config=peft_config,
    )
