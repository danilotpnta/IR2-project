"""
Credits to Claude 3.5 Sonnet/Perplexity
"""
import vllm
import torch
import numpy as np

def compare_model_weights(base_name, ft_name, dtype):
    # Load the models
    base = vllm.LLM(base_name, dtype=dtype)
    ft = vllm.LLM(ft_name, dtype=dtype)
    
    # Get the state dictionaries
    base_state_dict = base.model.state_dict()
    ft_state_dict = ft.model.state_dict()
    
    differences = {}
    
    # Compare each weight matrix
    for key in base_state_dict.keys():
        if key not in ft_state_dict:
            differences[key] = "Missing in model2"
            continue

        base_tensor = base_state_dict[key]
        ft_tensor = ft_state_dict[key]

        if base_tensor.shape != ft_tensor.shape:
            differences[key] = f"Shape mismatch: {base_tensor.shape} vs {ft_tensor.shape}"
            continue

        # Calculate differences
        diff = torch.abs(base_tensor - ft_tensor)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        if max_diff > 1e-7:  # Threshold for floating point differences
            differences[key] = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }

    return differences

def print_differences(differences):
    if not differences:
        print("Models are identical")
        return

    print("Differences found in the following layers:")
    for key, diff in differences.items():
        print(f"\nLayer: {key}")
        print(f"Difference: {diff}")

if __name__ == "__main__":
    base_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ft_name = "mjhamar/Meta-Llama-3.1-8B-Instruct-cpo-beir"
    dtype = torch.bfloat16

    differences = compare_model_weights(base_name, ft_name, dtype)
    print_differences(differences)

