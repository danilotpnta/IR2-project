import os
import pickle

import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

GPUS_AVAILABLE = int(os.environ["GPU_COUNT"])


def generate_queries(
    prompts: list[str],
    doc_ids: list[str],
    model_name="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
    save_folder_root="cache",
    batch_size=512,
    dataset="scifact",
    use_tqdm_inner=True,
    top_k=500,
    top_p=0.9,
    temperature=0.8,
    max_tokens=256,
    logprobs=3,
    stop=["\n", "Example", "Document:"],
    **kwargs,
):

    save_folder = os.path.join(save_folder_root, dataset, model_name)
    os.makedirs(save_folder, exist_ok=True)
    save_file = f"{save_folder}/prompts_cache_results.pkl"

    generations = []

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
        max_model_len=4096,
        enable_chunked_prefill=True,
        dtype="auto",
        tensor_parallel_size=GPUS_AVAILABLE,
    )

    loader_docid = DataLoader(doc_ids, batch_size=batch_size)
    loader_prompts = DataLoader(prompts, batch_size=batch_size)

    for i, (p, d_ids) in tqdm(
        enumerate(zip(loader_docid, loader_prompts)),
        desc="Generation",
        unit="queries",
    ):
        # Get the outputs.
        outputs = llm.generate(p, sampling_params, use_tqdm=use_tqdm_inner)

        # Save the outputs.
        generations += [
            (d_id, str(output.outputs[0].text)) for d_id, output in zip(d_ids, outputs)
        ]

        if i % 100 == 0:
            try:
                with open(save_file, "wb") as f:
                    pickle.dump(generations, f)
                save = pl.DataFrame(
                    generations,
                    schema={"doc_id": str, "query": str},
                    strict=False,
                    orient="row",
                )
                save.write_ndjson(f"{save_folder}/results.jsonl")
            except Exception as e:
                print("Oopsie.", e)

    with open(save_file, "wb") as f:
        pickle.dump(generations, f)
    save = pl.DataFrame(
        generations, schema={"doc_id": str, "query": str}, strict=False, orient="row"
    )
    save.write_ndjson(f"{save_folder}/results.jsonl")
    return generations
