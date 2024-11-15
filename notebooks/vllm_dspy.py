import dspy
import litellm

import os
os.environ['LITELLM_LOG'] = 'DEBUG'

# litellm.set_verbose=True

def ask_questions():
    qa = dspy.Predict('question: str -> response: str')
    response = qa(question="What is the capital of France?")
    # response = lm._generate("What is the capital of France?")
    print("\n** New response **")
    print(response.response)


    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question="What is the capital of France?")
    print("\n** New response **")
    print(response.answer)

    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

    summarize = dspy.ChainOfThought('document -> summary')
    response = summarize(document=document)
    # response = lm._generate(prompt=f"Make a summary of: {document}")

    print("\n** New response **")
    print(response.summary)

def main():
    lm = dspy.LM(
        model='openai/EleutherAI/gpt-j-6B',  
        api_base='http://localhost:8000/v1',  
        api_key="fake_key",                  
        model_type="completion",
        # cache={'no-cache': True, 'no-store': True},
    )
    dspy.configure(lm=lm)
    print("-- Questions using new dspy.LM -- ")
    ask_questions()

    lm_HF = dspy.HFClientVLLM(model="EleutherAI/gpt-j-6B", port=8000, url="http://localhost")
    dspy.configure(lm=lm_HF)
    print("-- Questions using HFClientVLLM -- ")
    ask_questions()

if __name__ == "__main__":
    main()