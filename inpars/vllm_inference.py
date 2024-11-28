import os
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

GPUS_AVAILABLE = int(os.environ["GPU_COUNT"])


def generate_queries(
    prompts: list[str],
    doc_ids: list[str],
    model_name="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
    save_folder="cache",
    batch_size=512,
    use_tqdm_inner=True,
    top_k=500,
    top_p=0.9,
    temperature=0.8,
    max_tokens=256,
    logprobs=3,
    stop=["\n", "Example", "Document:"],
    **kwargs,
):
    save_folder = os.path.join(save_folder, model_name)
    os.makedirs(save_folder, exist_ok=True)
    save_file = f"{save_folder}/results_recovery.json"

    try:
        with open(save_file, "r") as f:
            generations = json.load(f)
    except:
        generations = {}

    # Create a sampling params object.
    sampling_params = SamplingParams(
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        stop=stop,
        max_tokens=max_tokens,
        logprobs=logprobs,
        **kwargs,
    )

    # Create an LLM.
    llm = LLM(
        model=model_name,
        enable_prefix_caching=True,
        seed=42,
        gpu_memory_utilization=0.975,
        max_model_len=8192,
        enable_chunked_prefill=True,
        dtype="auto",
        tensor_parallel_size=GPUS_AVAILABLE,
    )

    loader_docid = DataLoader(doc_ids[len(generations) :], batch_size=batch_size)
    loader_prompts = DataLoader(prompts[len(generations) :], batch_size=batch_size)

    for d_ids, p in tqdm(
        zip(loader_docid, loader_prompts),
        desc="Generation",
        unit="batch",
        total=len(loader_docid),
    ):
        # Get the outputs.
        outputs = llm.generate(p, sampling_params, use_tqdm=use_tqdm_inner)

        # Save the outputs.
        generations |= {
            d_id: (
                output.outputs[0].text,
                repr(output.outputs[0].logprobs), # this is hard to show generally
                output.outputs[0].cumulative_logprob,
            )
            for d_id, output in zip(d_ids, outputs)
        }

        with open(save_file, "w") as f:
            json.dump(generations, f)

    return generations
