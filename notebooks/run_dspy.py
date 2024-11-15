import dspy
import litellm
# litellm.set_verbose=True

# lm = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8000, url="http://localhost")
# dspy.configure(lm=lm)

# # Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
# qa = dspy.ChainOfThought('question -> answer')

# # Run with the default LM configured with `dspy.configure` above.
# response = qa(question="Where is Bahrain located?")
# print(response.answer)

# response = lm._generate(prompt='What is the capital of Ecuador?')
# print(response)


# python -m vllm.entrypoints.openai.api_server --model "EleutherAI/gpt-j-6B" --port 8000
# python -m vllm.entrypoints.openai.api_server --model "mosaicml/mpt-7b" --port 8001



# config = {
#     "model_name": ["EleutherAI/gpt-j-6B", "mosaicml/mpt-7b"],
#     "ports": [8000, 8001],
# }

# for i in range(len(config)):
#     print(f"Running model: {config['model_name'][i]}")
#     print(f"Port: {config['ports'][i]}")

#     lm = dspy.LM(
#         model=f"openai/{config['model_name'][i]}",
#         api_base=f"http://localhost:{config['ports'][i]}/v1", 
#         api_key="fake_key",
#         model_type="completion",
#     )
#     dspy.configure(lm=lm)

#     qa = dspy.ChainOfThought("question -> answer")

#     response = qa(
#         question=f"Which color can you indetidy?\nText: The sky is turning yellow"
#     )
#     print(response)

#     response = qa(question="If I mixed two color red and blue what do I get?")
#     print(response)

#     response = qa(question="How many floors are in the castle David Gregory inherited?")
#     print(response)

model_names = ["EleutherAI/gpt-j-6B", "mosaicml/mpt-7b"]

lm = dspy.LM(
    model=f"openai/{model_names[0]}",
    api_base="http://localhost:8000/v1",  # ensure this points to your port
    api_key="fake_key",
    model_type="completion",
    cache = False
)
# lm = dspy.HFClientVLLM(model=model_names[0], port=8000, url="http://localhost")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")

# response = qa(question=f"Which color can you indetidy?\nText: The sky is turning yellow")
# response = qa(question=f"What is the capital of Ecuador?")
# print(response)

response = qa(question="If I mixed two colors red and blue which color do I get?")
# response = lm._generate(prompt="If I mixed two colors red and blue which color do I get?")
print(response)

# response = qa(question="How many floors are in the castle David Gregory inherited?")
# print(response)


# vllm_mpt = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8001, url="http://localhost")
# dspy.configure(lm=vllm_mpt)

# #Example DSPy CoT QA program
# qa = dspy.ChainOfThought('question -> answer')

# response = qa(question="If I mixed two colors red and blue what do I get?") 
# print(response.answer)

# response = vllm_mpt._generate(prompt='If I mixed two colors red and blue what do I get?')
# print(response)

document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
# response = summarize(document=document)
# response = lm._generate(prompt=f"Make a summary of: {document}")

# print(response)