from llama_cpp import Llama
import polars as pl
import pickle
from tqdm import tqdm
import os

MODEL_PATH = os.path.join(os.environ["HF_HOME"], "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_L" , "Llama-3.1-Nemotron-70B-Instruct-HF-Q5_K_L-00001-of-00002.gguf")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Uncomment to use GPU acceleration
    seed=42,
    flash_attn=True,
    n_ctx=4096,  # Uncomment to increase the context window
)

out = llm.create_completion("Q: Name the planets in the solar system? A: ", max_tokens=128, stop=["\n", "Q:"], echo="True")
print(out["choices"])

file = "cache/prompts_cache_prompts.csv"
df = pl.read_csv(file)

def save_intermediate(_list):
    with open("cache/prompts_cache_results.pkl", 'wb') as f:
        pickle.dump(_list, f)

prompts = df['prompt']
outputs = []

# TODO: This ain't working for some reason, guess I'll just need to load in my own queries.
prompt_fill = """Generate 5 RELEVANT QUERIES for the final document based on the following examples. Be brief and simple. 
The generated RELEVANT QUERIES must follow the following template: 
```
1.
2.
3.
4.
5.
```
Rest:

{}
"""

for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="queries"):
    
    prompt = prompt_fill.format(prompt)
    generated_text = llm.create_completion(prompt, max_tokens=128, stop=["\n", "Example"])
    # print(generated_text["choices"])
    outputs.append(generated_text["choices"])
    if i % 100 == 0:
        save_intermediate(outputs)

save_intermediate(outputs)