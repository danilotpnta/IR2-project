from llama_cpp import Completion, CompletionChoice, Llama
import polars as pl
import pickle
from tqdm import tqdm, trange
import os


dataset = "scifact"
file = f"cache/{dataset}/prompts_cache_prompts.csv"
df = pl.read_csv(file)

model_name = "llamacpp_70b"
save_folder = os.path.join("cache", dataset, model_name)
os.makedirs(save_folder, exist_ok=True)
save_file = f"{save_folder}/prompts_cache_results.pkl"

prompts = df['prompt']
outputs = []

# TODO: This ain't working for some reason, guess I'll just need to load in my own queries.
# prompt_fill = """Generate 5 RELEVANT QUERIES for the final document based on the following examples. Be brief and simple. 
# The generated RELEVANT QUERIES must follow the following template: 
# ```
# 1.
# 2.
# 3.
# 4.
# 5.
# ```
# Rest:
# {}
# """

# Load the previous prompts
with open(save_file, 'rb') as f:
    outputs = pickle.load(f)
print(outputs[0])
print(f"Found {len(outputs)} generated prompts already.")

MODEL_PATH = os.path.join(os.environ["HF_HOME"], "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_L" , "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_L-00001-of-00002.gguf")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # use full GPU acceleration
    seed=42,
    flash_attn=True,
    n_ctx=4096,  # Uncomment to increase the context window
    offload_kqv=True, n_threads=len(os.sched_getaffinity(0))-1,
    verbose=False,
    last_n_tokens_size=128
)

out = llm.create_completion("How many r's are there in the word \"strawberry\"?", max_tokens=128, stop=["\n", "Q:"], echo="True")
print(out["choices"])

start = len(outputs) if len(outputs) > 1 else 0
for i in trange(start, len(prompts), total=len(prompts), initial=start, desc="Generation", unit="queries"):
    prompt = prompts[i]
    # prompt = prompt_fill.format(prompt)
    generated_text = llm.create_completion(prompt, max_tokens=128, stop=["\n", "Example", "Document:"])
    # print(generated_text["choices"])
    out = generated_text["choices"]
    if len(out) == 1:
        out = out[0]
        out["doc_id"] = df["doc_id"][i]
        outputs.append(out)
    else:
        print("oops", len(out))
        for _ in out:
            _["doc_id"] = df["doc_id"][i]
            outputs.append(_)

    if i % 100 == 0:
        with open(save_file, 'wb') as f:
            pickle.dump(outputs, f)
        try:
            save = pl.DataFrame(outputs)
            save.write_ndjson(f"{save_folder}/results.jsonl")
        except TypeError:
            bad_idxs = [(i, len(el)) for i, el in enumerate(outputs) if len(el)>=1]
            for bad_idx, _ in bad_idxs:
                outputs[bad_idx] = outputs[0]
            print(bad_idxs)
            save = pl.DataFrame(outputs)
            save.write_ndjson(f"{save_folder}/results.jsonl")

save = pl.from_records(outputs)
save.write_ndjson(f"{save_folder}/results.jsonl")
with open(save_file, 'wb') as f:
    pickle.dump(outputs, f)
    