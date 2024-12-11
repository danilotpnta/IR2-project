import logging
import os
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

GPUS_AVAILABLE = int(os.environ.get("GPU_COUNT", 1))
SEED = int(os.environ.get("SEED", 42))

logger = logging.getLogger()


class VLLMQueryGenerator:
    def __init__(self):
        self.model = None
        self.model_name = ""

    def __call__(self,
        prompts: list[str],
        doc_ids: list[str],
        model_name="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
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
            )
            self.model = llm
            self.model_name = model_name
        else:
            llm = self.model

        loader_docid = DataLoader(doc_ids[len(generations) :], batch_size=batch_size)
        loader_prompts = DataLoader(prompts[len(generations) :], batch_size=batch_size)

        try:
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
                        repr(
                            output.outputs[0].logprobs
                        ),  # this is a bit hard to serialize trivially
                        output.outputs[0].cumulative_logprob,
                    )
                    for d_id, output in zip(d_ids, outputs)
                }

                with open(save_file, "w") as f:
                    json.dump(generations, f)
        except RuntimeError as e:
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
                        repr(
                            output.outputs[0].logprobs
                        ),  # this is a bit hard to serialize trivially
                        output.outputs[0].cumulative_logprob,
                    )
                    for d_id, output in zip(d_ids, outputs)
                }

                with open(save_file, "w") as f:
                    json.dump(generations, f)

        return generations

# singleton
generate_queries = VLLMQueryGenerator()
