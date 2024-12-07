import polars as pl
from tqdm import tqdm
import pickle
from itertools import batched

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in strawberry?"

def encode_message(prompts: str | list[str]):
    # TODO: Actually figure out if there is a nice way to do batching.
    if isinstance(prompts, str):
        prompts = [prompts]
    message_template = {"role": "user", "content": ""}
    messages = [message_template | {"content": p} for p in prompts]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

# tokenized_message = encode_message(prompt) 
# response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=512, pad_token_id = tokenizer.eos_token_id)
# generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
# generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# print(generated_text)

# See response at top of model card

file = "cache/prompts_cache_prompts.csv"
df = pl.read_csv(file)

def save_intermediate(_list):
    with open("cache/prompts_cache_results.pkl", 'wb') as f:
        pickle.dump(_list, f)

prompts = df['prompt']
outputs = []
for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="queries"):
    
    tokenized_message = encode_message(prompt) 
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=512, pad_token_id = tokenizer.eos_token_id)
    # input_lengths = (tokenized_message['input_ids'] != tokenizer.pad_token_id).sum(dim=-1)
    # generated_tokens = [response_token_ids[i, input_lengths[i]:].tolist() for i in range(response_token_ids.size(0))]
    generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print(generated_text)
    outputs.append(generated_text)
    if i % 100 == 0:
        save_intermediate(outputs)

save_intermediate(outputs)
# df_2 = pl.DataFrame({"doc_id": df["doc_id"], "prompt"})