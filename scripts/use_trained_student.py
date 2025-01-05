from inpars.cpo_dataset import score_queries
from inpars.vllm_inference import generate_queries
from inpars.query_eval import QueryEval
import json

output_path = "/scratch-shared/scur2577/cpo_dataset/msmarco_document/llmt_preference_Meta-Llama-3.1-8B_100000.json"
output_dir = "/scratch-shared/scur2577/cpo_dataset/msmarco_document"

# Load the documents and doc_ids
with open(output_path, "r") as f:
    output = json.load(f)

prompts = [doc["prompt"] for doc in output["data"].values()]
doc_ids = [doc_id for doc_id in output["data"].keys()]

queries = generate_queries(prompts, doc_ids, model_name="inpars-plus/Meta-Llama-3.1-8B_merged-16bit_CPO_MSMARCO", force=False, temperature=0.3)

# Save the queries
for doc_id, query in queries.items():
    output["data"][doc_id]["taught_model_query"] = query
    
with open(output_path, "w") as f:
    json.dump(output, f)
    
query_eval = QueryEval.load_from_cache(output_dir)
if query_eval is None:
    print("Error!")
    

ref_doc_query_pairs = {
    doc_id: data["taught_model_query"] for doc_id, data in output["data"].items()
}

scores = score_queries(ref_doc_query_pairs, query_eval)

for doc_id, score in scores.items():
    output["data"][doc_id]["taught_model_score"] = score

# checkpoint
with open(output_path, "w") as f:
    json.dump(output, f)