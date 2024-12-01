import os
import sys
import json
import logging
import torch
from dataclasses import dataclass, field
from peft import PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainerCallback,
    HfArgumentParser,
    set_seed
)

from ALMA.utils.cpo_trainer import CPOTrainer
from ALMA.utils.cpo_config import CPOConfig
from inpars.cpo_dataset import load_cpo_dataset, DataConfig

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use wandb for logging"}
    )
    peft_config_path: str = field(
        default=None,
        metadata={"help": "Path to peft config file"}
    )

    def __post_init__(self):
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.use_wandb:
            self.wandb_api_key = os.getenv("WANDB_API_KEY")
            if self.wandb_api_key is None:
                raise ValueError("WANDB_API_KEY is not set")



class SavePeftModelCallback(TrainerCallback):
    """
    Copy-paste from ALMA/utils/utils.py
    """
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(
            checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control

def train(
    model_args: ModelArguments,
    data_config: DataConfig, 
    cpo_config: CPOConfig,
    peft_config: PeftConfig
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
        cpo_config.report_to = "wandb"

    # load dataset
    # with fieds ["doc_id", "text"] (would be nice to have some gt queries as well.)
    # dataset = load_cpo_dataset(data_config, cpo_config, model_args.tokenizer)

    # avoid passive-agressive warning message
    cpo_config.remove_unused_columns = False

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # TODO: prepare trainer
    trainer = CPOTrainer(
        model=model,
        tokenizer=model_args.tokenizer,
        peft_config=peft_config,
        args=cpo_config,
        train_dataset=None,
        callbacks=[SavePeftModelCallback]
    )
    # train
    trainer.train()
    # save model
    trainer.save_model()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, CPOConfig, DataConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # NOTE: stole it from ALMA/run_cpo_llmmt.py
        model_args, training_config, data_config = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
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
        peft_config=peft_config
    )
