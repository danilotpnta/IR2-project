from math import ceil
import pickle
from tqdm import tqdm, trange
from vllm import LLM, SamplingParams
import os
import polars as pl

GPUS_AVAILABLE=int(os.environ["GPU_COUNT"])

dataset = "scifact"
file = f"cache/{dataset}/prompts_cache_prompts.csv"
df = pl.read_csv(file)

model_name = "llama70b_fp8"
save_folder = os.path.join("cache", dataset, model_name)
os.makedirs(save_folder, exist_ok=True)
save_file = f"{save_folder}/prompts_cache_results.pkl"
save_file_full = f"{save_folder}/prompts_cache_results_full.pkl"

prompts = df['prompt']
doc_ids = df['doc_id']
generations = []

# Create a sampling params object.
# TODO: Tweaking the sampling params can be essential
sampling_params = SamplingParams(top_k=500, top_p=0.9, temperature=0.8, stop=["\n", "Example", "Document:"], max_tokens=256, logprobs=3)

# Create an LLM.
llm = LLM(model="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic", enable_prefix_caching=True, seed=42, gpu_memory_utilization=0.975, max_model_len=4096, enable_chunked_prefill=True,dtype="auto", tensor_parallel_size=GPUS_AVAILABLE)

BATCH_SIZE = 512
# from itertools import batched
# loader_docid = batched(doc_ids, BATCH_SIZE)
# loader_prompts = batched(prompts, BATCH_SIZE)

# for p, d_ids in tqdm(zip(loader_docid, loader_prompts), desc="Generation", unit="queries", total=ceil(doc_ids / BATCH_SIZE)):
#     llm.generate(p, sampling_params, use_tqdm=False)

outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
# Print the outputs.
generations += [(d_id, str(output.outputs[0].text)) for d_id, output in zip(doc_ids, outputs)]

with open(save_file_full, 'wb') as f:
    pickle.dump(outputs)
# if i % 100 == 0:
#     with open(save_file, 'wb') as f:
#         pickle.dump(generations, f)
#     try:
#         save = pl.DataFrame(generations, schema={"doc_id": str, "query": str}, strict=False, orient="row")
#         save.write_ndjson(f"{save_folder}/results.jsonl")
#     except Exception as e:
#         print("Oopsie.", e)
        
with open(save_file, 'wb') as f:
    pickle.dump(generations, f)
save = pl.DataFrame(generations, schema={"doc_id": str, "query": str}, strict=False, orient="row")
save.write_ndjson(f"{save_folder}/results.jsonl")