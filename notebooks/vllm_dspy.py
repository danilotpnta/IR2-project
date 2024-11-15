import dspy
import litellm

# import os
# os.environ["LITELLM_LOG"] = "DEBUG"

# litellm.set_verbose = True


def ask_questions(lm_HF):
    qa = dspy.Predict("question: str -> response: str")
    response = qa(question="What is the capital of France?")
    # response = lm._generate("What is the capital of France?")
    print("\n** New response **")
    print(response.response)

    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question="What is the capital of France?")
    print("\n** New response **")
    print(response.answer)

    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

    summarize = dspy.ChainOfThought("document -> summary")
    response = summarize(document=document)
    # response = lm._generate(prompt=f"Make a summary of: {document}")

    print("\n** New response **")
    print(response.summary)

    ans = "How did the Canadian COVID-19 outbreak compare to other countries?"
    question = """
    Example 1:
    Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.
    Good Question: How much caffeine is ok for a pregnant woman to have?
    Bad Question: Is a little caffeine ok during pregnancy?

    Example 2:
    Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.
    Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?
    Bad Question: What fruit is native to Australia?

    Example 3:
    Document: The Canadian Armed Forces. 1  The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2  There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3  In Canada, August 9 is designated as National Peacekeepers' Day.
    Good Question: Information on the Canadian Armed Forces size and history.
    Bad Question: How large is the Canadian military?

    Example 4:
    Document: The Coronavirus 2019 pandemic in Canada: the impact of public health interventions on the course of the outbreak in Alberta and other provinces Background: The SARS-CoV-2 disease 2019 (COVID-19) pandemic has spread across the world with varying impact on health systems and outcomes. We assessed how the type and timing of public- health interventions impacted the course of the outbreak in Alberta and other Canadian provinces. Methods: We used publicly-available data to summarize rates of laboratory data and mortality in relation to measures implemented to contain the outbreak and testing strategy. We estimated the transmission potential of SARS-CoV-2 before the state of emergency declaration for each province (R0) and at the study end date (Rt). Results: The first cases were confirmed in Ontario (January 25) and British Columbia (January 28). All provinces implemented the same health-policy measures between March 12 and March 30. Alberta had a higher percentage of the population tested (3.8%) and a lower mortality rate (3/100,000) than Ontario (2.6%; 11/100,000) or Quebec (3.1%; 31/100,000). British Columbia tested fewer people (1.7%) and had similar mortality as Alberta
    Good Question:
    Bad Question:
    """

    # qa = dspy.Predict("question: str -> response: str")
    # response = qa(question=question)
    # print("\n** New response **")
    # print(response.response)

    response = lm_HF(prompt=question, cache=False)
    print("\n** New response **")
    print(response)

def main():
    model_name = "EleutherAI/gpt-j-6B"
    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base="http://localhost:8000/v1",
        api_key="fake_key",
        model_type="completion",
        cache=False,
    )
    dspy.configure(lm=lm)
    print("-- Questions using new dspy.LM -- ")
    ask_questions(lm)

    lm_HF = dspy.HFClientVLLM(
        model=model_name, port=8000, url="http://localhost"
    )
    dspy.configure(lm=lm_HF)
    print("-- Questions using HFClientVLLM -- ")
    ask_questions(lm_HF)


if __name__ == "__main__":
    main()
