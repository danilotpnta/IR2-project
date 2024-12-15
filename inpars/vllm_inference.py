import logging
import os
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

GPUS_AVAILABLE = int(os.environ.get("GPU_COUNT", 1))
SEED = int(os.environ.get("SEED", 42))

logger = logging.getLogger()

def _serialize_logprobs(logprobs, num_tokens):
    """
    LLM.generate() returns a list of dictionaries of the form:
    [
        { token_id: Logprob(logprob=..., rank=...) ...}
    ]
    The size of these dictionaries of <= num_tokens+1.
    If length is >num_tokens, the lowest ranked token is the one in the returned sequence.
    If length is num_tokens, the sampled token is in top-num_tokens.
    We only need num_tokens number of logprobs. (otherwise numpy will complain)

    Return:
    - If num_tokens is None, return an empty list.
    - If num_tokens=1, return a list of logprobs.
    - Otherwise, return a list of lists of logprobs.
    """
    ret = []
    if not num_tokens:
        return ret
    # iterate over the list of outputs
    for item in logprobs:
        lp_list = list(sorted(item.values(), key=lambda x: x.rank)) # ascending order; highest rank first (1, 2, ...)
        if len(lp_list) > num_tokens:
            _ = lp_list.pop(-2) # remove the lowest ranked token that was not sampled.
        ret.append([lp.logprob for lp in lp_list]
                   if num_tokens > 1 else lp_list[0].logprob)
    return ret

class VLLMQueryGenerator:
    def __init__(self):
        self.model = None
        self.model_name = ""

    def __call__(self,
        prompts: list[str],
        doc_ids: list[str],
        model_name="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        lora_repo=None,
        save_folder="cache",
        max_prompt_length=8192,
        batch_size=256,
        use_tqdm_inner=True,
        top_k=500,
        top_p=0.9,
        temperature=0.8,
        max_tokens=256,
        logprobs=None,
        stop=["\n", "Example", "Document:"],
        dtype="auto",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        force=True,
        seed=SEED,
        **kwargs,
    ):
        save_folder = os.path.join(save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        save_file = f"{save_folder}/results_recovery.json"

        if force is True:
            generations = {}
        else:
            try:
                with open(save_file, "r") as f:
                    generations = json.load(f)
                logger.info(f"Found {len(generations)} saved generations.")
                if len(generations) == len(prompts):
                    logger.info("All generations have already been recovered.")
                    return generations
            except Exception:
                logger.info("No generated queries were recovered.")
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
        lora_kwargs = {}
        if lora_repo is not None:
            lora_kwargs = {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
                "qlora_adapter_name_or_path": lora_repo,
                "enable_lora": True,
                "max_lora_rank": 64,
            }

        loader_docid = DataLoader(doc_ids[len(generations) :], batch_size=batch_size)
        loader_prompts = DataLoader(prompts[len(generations) :], batch_size=batch_size)

        try:
            # avoid re-loading the model all the time
            if self.model_name != model_name:
                # Create an LLM.
                llm = LLM(
                    model=model_name,
                    enable_prefix_caching=enable_prefix_caching,
                    seed=seed,
                    gpu_memory_utilization=0.95,
                    max_model_len=max_prompt_length,
                    enable_chunked_prefill=enable_chunked_prefill,
                    dtype=dtype,
                    tensor_parallel_size=GPUS_AVAILABLE,
                    max_num_batched_tokens=max_prompt_length,
                    **lora_kwargs,
                )
                self.model = llm
                self.model_name = model_name
            else:
                llm = self.model

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
                        _serialize_logprobs(output.outputs[0].logprobs, logprobs),
                        output.outputs[0].cumulative_logprob,
                    )
                    for d_id, output in zip(d_ids, outputs)
                }

                with open(save_file, "w") as f:
                    json.dump(generations, f)
        except (ValueError, RuntimeError) as e:
            # Catch-all mainly targeting the bug with prefix caching
            # TODO: Identify this issue eventually.
            logging.error(f"RuntimeError detected: {e}", stack_info=True)
            logging.warning(
                "Retrying again with prefix caching disabled. If this issue persists."
            )

            llm = LLM(
                model=model_name,
                enable_prefix_caching=False,
                seed=seed,
                gpu_memory_utilization=0.95,
                max_model_len=max_prompt_length,
                enable_chunked_prefill=False,
                dtype=dtype,
                tensor_parallel_size=GPUS_AVAILABLE,
                max_num_batched_tokens=max_prompt_length,
                **lora_kwargs,
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
                        _serialize_logprobs(output.outputs[0].logprobs, logprobs),
                        output.outputs[0].cumulative_logprob,
                    )
                    for d_id, output in zip(d_ids, outputs)
                }

                with open(save_file, "w") as f:
                    json.dump(generations, f)

        return generations

# singleton
generate_queries = VLLMQueryGenerator()
