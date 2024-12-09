import torch
from transformers import AutoModel
from peft import PeftModel, merge_lora_adapters
import json
import os

def merge_and_quanzize(
    model_path: str,
    model_name: str,
    output_dir: str,
    quantization_config_path: str,
    quantization_config: dict,
    use_wandb: bool,
):
    """
    Given a path to a peft model, where the base model is already quantized,
    infer the quantization parameters for the adapter model and quantize it.
    """
    # Load the base model
    base_model = AutoModel.from_pretrained(model_name)
    
    # Load the LoRA adapters
    adapter_model = PeftModel.from_pretrained(model_path, base_model=base_model)
    
    # Merge the LoRA adapters into the base model
    merged_model = merge_lora_adapters(base_model, adapter_model)
    
    # Load quantization configuration
    with open(quantization_config_path, 'r') as f:
        quantization_config = json.load(f)
    
    # Apply quantization to the merged model
    quantized_model = torch.quantization.quantize_dynamic(
        merged_model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8,
        **quantization_config
    )

    # Save the quantized model
    quantized_model.save_pretrained(output_dir)

    if use_wandb:
        import wandb
        wandb.init(project="quantized_model_project")
        wandb.save(os.path.join(output_dir, "*"))

    print(f"Quantized model saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quantization_config_path", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    merge_and_quanzize(
        args.model_path,
        args.model_name,
        args.output_dir,
        args.quantization_config_path,
        {},
        args.use_wandb,
    )