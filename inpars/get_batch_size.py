import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


def estimate_best_batch_size(
    model_name="EleutherAI/gpt-j-6B",
    max_length=512,
    start_batch=1,
    fp16=False,
    safety_margin=0.8,
):
    print(f"Testing batch sizes for {model_name}")

    if not torch.cuda.is_available():
        print("No GPU available")
        return 1

    gpu = torch.cuda.get_device_properties(0)
    total_memory = gpu.total_memory / 1024**3
    print(f"Total GPU Memory: {total_memory:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model_kwargs = {}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.to("cuda")

        base_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Base memory usage with model loaded: {base_memory:.2f} GB")

        batch_size = start_batch
        last_successful = 0

        while True:
            try:
                dummy_input = torch.randint(
                    0, 1000, (batch_size, max_length), device="cuda"
                )

                with torch.no_grad():
                    outputs = model.generate(
                        dummy_input,
                        max_new_tokens=64,
                        min_length=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_free = total_memory - memory_used

                print(f"\nBatch size {batch_size}:")
                print(f"Memory used: {memory_used:.2f} GB")
                print(f"Memory free: {memory_free:.2f} GB")

                if memory_used / total_memory > safety_margin:
                    print(f"\nReached {safety_margin*100}% of GPU memory")
                    break

                last_successful = batch_size
                batch_size *= 2

                del outputs, dummy_input
                torch.cuda.empty_cache()
                gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOut of memory at batch size {batch_size}")
                    break
                else:
                    raise e

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        recommended = max(1, last_successful)
        print(f"\nRecommended batch size: {recommended}")
        return recommended

    except Exception as e:
        print(f"Error during testing: {e}")
        return 1


best_batch_size = estimate_best_batch_size(
    model_name="EleutherAI/gpt-j-6B", max_length=512, fp16=True, safety_margin=0.8
)


"""
Some rules of thumb:

For GPT-J-6B:

8GB GPU: batch_size=1 (or 2 with fp16)
16GB GPU: batch_size=2-4 (or 4-8 with fp16)
24GB GPU: batch_size=4-6 (or 8-12 with fp16)
32GB GPU: batch_size=6-8 (or 12-16 with fp16)
40GB+ GPU: batch_size=8+ (or 16+ with fp16)


Factors that affect maximum batch size:

Model size (GPT-J-6B is 6 billion parameters)
Input sequence length (longer sequences need more memory)
FP16 vs FP32 (FP16 uses half the memory)
Other running processes on GPU
Generation parameters (max_new_tokens, etc.)


Memory optimization tips:

Use fp16=True to halve memory usage
Reduce max_prompt_length and max_new_tokens if possible
Clear GPU cache between runs: torch.cuda.empty_cache()
Monitor memory usage during generation
Leave some memory headroom (don't use 100% of available memory)
"""
