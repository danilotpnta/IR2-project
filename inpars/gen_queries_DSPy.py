import os
import sys
import dspy
import litellm
# os.environ["LITELLM_LOG"] = "DEBUG"
# litellm.set_verbose = True

STOP_WORDS = ["\n", "\n\n", "Bad Question:", "Example", "Document:"]

def ask_simple_questions(lm):
    qa = dspy.Predict("question: str -> response: str")
    response = qa(question="What is the capital of France?")
    print("\n** New response dspy.Predict **")
    print(response.response)

    qa_cot = dspy.ChainOfThought("question -> answer")
    response = qa_cot(question="What is the capital of France?")
    print("\n** New response dspy.ChainOfThought **")
    print(response.answer)

    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

    summarize = dspy.ChainOfThought("document -> summary")
    response = summarize(document=document)
    print("\n** New response Summary v1 **")
    print(response.summary)

    response = lm(prompt=document, use_cache=False)
    print("\n** New response Summary v2 **")
    print(response)


def ask_Trec_Covid_questions(lm):
    qgen_by_InPars_0 = (
        "How did the Canadian COVID-19 outbreak compare to other countries?"
    )
    prompt_0 = "Example 1:\nDocument: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1\u00bd 8-ounce cups of coffee or one 12-ounce cup of coffee.\nGood Question: How much caffeine is ok for a pregnant woman to have?\nBad Question: Is a little caffeine ok during pregnancy?\n\nExample 2:\nDocument: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\nGood Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\nBad Question: What fruit is native to Australia?\n\nExample 3:\nDocument: The Canadian Armed Forces. 1  The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2  There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3  In Canada, August 9 is designated as National Peacekeepers' Day.\nGood Question: Information on the Canadian Armed Forces size and history.\nBad Question: How large is the Canadian military?\n\nExample 4:\nDocument: The Coronavirus 2019 pandemic in Canada: the impact of public health interventions on the course of the outbreak in Alberta and other provinces Background: The SARS-CoV-2 disease 2019 (COVID-19) pandemic has spread across the world with varying impact on health systems and outcomes. We assessed how the type and timing of public- health interventions impacted the course of the outbreak in Alberta and other Canadian provinces. Methods: We used publicly-available data to summarize rates of laboratory data and mortality in relation to measures implemented to contain the outbreak and testing strategy. We estimated the transmission potential of SARS-CoV-2 before the state of emergency declaration for each province (R0) and at the study end date (Rt). Results: The first cases were confirmed in Ontario (January 25) and British Columbia (January 28). All provinces implemented the same health-policy measures between March 12 and March 30. Alberta had a higher percentage of the population tested (3.8%) and a lower mortality rate (3\/100,000) than Ontario (2.6%; 11\/100,000) or Quebec (3.1%; 31\/100,000). British Columbia tested fewer people (1.7%) and had similar mortality as Alberta. Data on provincial testing strategies were insufficient to inform further analyses. Mortality rates increased with increasing rates of lab- confirmed cases in Ontario and Quebec, but not in Alberta. R0 was similar across all provinces, but varied widely from 2.6 (95% confidence intervals 1.9-3.4) to 6.4 (4.3-8.5), depending on the assumed time interval between onset of symptoms in a primary and a secondary case (serial interval). The outbreak is currently under control in Alberta, British Columbia and Nova Scotia (Rt <1). Interpretation: COVID-19-related health outcomes varied by province despite rapid implementation of similar health-policy interventions across Canada. Insufficient information about provincial testing strategies and a lack of primary data on serial interval are major limitations of existing data on the Canadian COVID-19 outbreak.\nGood Question:"
    base_0 = "Document: The Coronavirus 2019 pandemic in Canada: the impact of public health interventions on the course of the outbreak in Alberta and other provinces Background: The SARS-CoV-2 disease 2019 (COVID-19) pandemic has spread across the world with varying impact on health systems and outcomes. We assessed how the type and timing of public- health interventions impacted the course of the outbreak in Alberta and other Canadian provinces. Methods: We used publicly-available data to summarize rates of laboratory data and mortality in relation to measures implemented to contain the outbreak and testing strategy. We estimated the transmission potential of SARS-CoV-2 before the state of emergency declaration for each province (R0) and at the study end date (Rt). Results: The first cases were confirmed in Ontario (January 25) and British Columbia (January 28). All provinces implemented the same health-policy measures between March 12 and March 30. Alberta had a higher percentage of the population tested (3.8%) and a lower mortality rate (3\/100,000) than Ontario (2.6%; 11\/100,000) or Quebec (3.1%; 31\/100,000). British Columbia tested fewer people (1.7%) and had similar mortality as Alberta. Data on provincial testing strategies were insufficient to inform further analyses. Mortality rates increased with increasing rates of lab- confirmed cases in Ontario and Quebec, but not in Alberta. R0 was similar across all provinces, but varied widely from 2.6 (95 confidence intervals 1.9-3.4) to 6.4 (4.3-8.5), depending on the assumed time interval between onset of symptoms in a primary and a secondary case (serial interval). The outbreak is currently under control in Alberta, British Columbia and Nova Scotia (Rt <1). Interpretation: COVID-19-related health outcomes varied by province despite rapid implementation of similar health-policy interventions across Canada. Insufficient information about provincial testing strategies and a lack of primary data on serial interval are major limitations of existing data on the Canadian COVID-19 outbreak.\n"

    qgen_by_InPars_1 = "What are the common pitfalls in telemedicine consultations?"
    prompt_1 = "Example 1:\nDocument: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1\u00bd 8-ounce cups of coffee or one 12-ounce cup of coffee.\nGood Question: How much caffeine is ok for a pregnant woman to have?\nBad Question: Is a little caffeine ok during pregnancy?\n\nExample 2:\nDocument: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\nGood Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\nBad Question: What fruit is native to Australia?\n\nExample 3:\nDocument: The Canadian Armed Forces. 1  The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2  There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3  In Canada, August 9 is designated as National Peacekeepers' Day.\nGood Question: Information on the Canadian Armed Forces size and history.\nBad Question: How large is the Canadian military?\n\nExample 4:\nDocument: Pitfalls in telemedicine consultations in the era of COVID 19 and how to avoid them BACKGROUND AND AIMS: With restrictions on face to face clinical consultations in the COVID-19 pandemic, Telemedicine has become an essential tool in providing continuity of care to patients. We explore the common pitfalls in remote consultations and strategies that can be adopted to avoid them. METHODS: We have done a comprehensive review of the literature using suitable keywords on the search engines of PubMed, SCOPUS, Google Scholar and Research Gate in the first week of May 2020 including 'COVID-19', 'telemedicine' and'remote consultations'. RESULTS: Telemedicine has become an integral part to support patient's clinical care in the current COVID-19 pandemic now and will be in the future for both primary and secondary care. Common pitfalls can be identified and steps can be taken to prevent them. CONCLUSION: Telemedicine it is going to play a key role in future of health medicine, however, telemedicine technology should be applied in appropriate settings and situations. Suitable training, enhanced documentations, communication and observing information governance guidelines will go a long way in avoiding pitfalls associated with remote consultations.\nGood Question:"
    base_1 = "Document: Pitfalls in telemedicine consultations in the era of COVID 19 and how to avoid them BACKGROUND AND AIMS: With restrictions on face to face clinical consultations in the COVID-19 pandemic, Telemedicine has become an essential tool in providing continuity of care to patients. We explore the common pitfalls in remote consultations and strategies that can be adopted to avoid them. METHODS: We have done a comprehensive review of the literature using suitable keywords on the search engines of PubMed, SCOPUS, Google Scholar and Research Gate in the first week of May 2020 including 'COVID-19', 'telemedicine' and'remote consultations'. RESULTS: Telemedicine has become an integral part to support patient's clinical care in the current COVID-19 pandemic now and will be in the future for both primary and secondary care. Common pitfalls can be identified and steps can be taken to prevent them. CONCLUSION: Telemedicine it is going to play a key role in future of health medicine, however, telemedicine technology should be applied in appropriate settings and situations. Suitable training, enhanced documentations, communication and observing information governance guidelines will go a long way in avoiding pitfalls associated with remote consultations.\n"

    prefix = "Here are some examples of documents with their respective good and bad questions\n\n"
    prompt_1_1 = prefix + prompt_1

    # Comparing outputs from dspy.Modules vs using only lm(..)
    # Doc 0: Canadian COVID-19 outbreak
    qaDocQ = dspy.Predict("document -> query")
    response = qaDocQ(
        document=prompt_0,
        config={
            "max_tokens": 64,
            "stop": ["\n", "\n\n", "Bad Question:", "Example", "Document:"],
        },
    )
    print("\n** New response dspy.Predict Doc 0 **")
    print(f"## InPars Query: {qgen_by_InPars_0}")
    print(response.query)

    response = lm(prompt=prompt_0, use_cache=False, max_tokens=64, stop=["\n", "\n\n"])
    print("\n** New response lm Doc 0 **")
    print(f"## InPars Query: {qgen_by_InPars_0}")
    print(response)

    # response = lm._generate(
    #     prompt=prompt_0, use_cache=False, max_tokens=64, stop=["\n", "\n\n"]
    # )
    # print("\n** New response lm Doc 0  Generate**")
    # print(f"## InPars Query: {qgen_by_InPars_0}")
    # print(response)

    # print("\n** Only query **")
    # print(f"## InPars Query: {qgen_by_InPars_0}")
    # print(response["choices"][0]["text"])

    # sys.exit(0)
    response = qaDocQ(document=prompt_1)
    print("\n** New response qaDocQ Doc 1 **")
    print(response.query)

    # Doc 1: Pitfalls in telemedicine consultations
    response = lm(prompt=prompt_1, use_cache=False, max_tokens=64)
    print("\n** New response lm Doc 1 **")
    print(response)

    response = qaDocQ(document=prompt_1_1, max_tokens=64)
    print("\n** New response qaDocQ Doc 1 + Prefix **")
    print(response.query)

    common_prompts = {
        "good_question": "Good Question:",
        "relevant_question": "Relevant Question:",
        "possible_questions": "These are possible questions for this Document: 1.",
        "find_document": "To find this document, use the following URL: ",
        "keywords": "Keywords: ",
    }

    documents = {
        "Doc 0: Canadian COVID-19 outbreak": {
            "base": base_0,
            "query": qgen_by_InPars_0,
        },
        "Doc 1: Pitfalls in telemedicine consultations": {
            "base": base_1,
            "query": qgen_by_InPars_1,
        },
    }

    follow_up_Q = {}
    for doc_key, doc_data in documents.items():
        base = doc_data["base"]
        for key, text in common_prompts.items():
            prompt_key = f"prompt{len(follow_up_Q)}"
            follow_up_Q[prompt_key] = [base + text, text]

    # Generate and display responses
    responses = ""
    for k, v in follow_up_Q.items():
        doc_name = (
            "Doc 0: Canadian COVID-19 outbreak"
            if "_0" in k
            else "Doc 1: Pitfalls in telemedicine consultations"
        )
        inpars_query = documents[doc_name]["query"]

        print(f"\n** New response {k} **")
        print(f"** Asks: {v[1]} **")
        print(f"## {doc_name}")
        print(f"## InPars Query: {inpars_query}")

        # Generate response using lm
        response = lm(prompt=v[0], use_cache=False, max_tokens=64)
        responses += response[0]  # Append first token of the response
        print(response)

    # TODO: Filter responses to only obtain question ie "..?" otherwhise GPT-J
    # is a token-by-token text generation and do not follow up with instruction-tuned tasks
    # print("\n** New response Redefined **")
    # redefine_response = base + responses + "\nMost relevant question:"
    # print("\nRedefined response: ", redefine_response)
    # response = lm(prompt=redefine_response, max_tokens=64)
    # print("\nRefined Query: ",response)


def main():
    model_name = "meta-llama/Llama-3.1-8B"

    lm_HF = dspy.HFClientVLLM(model=model_name, port=8000, url="http://localhost", stop=STOP_WORDS)
    dspy.configure(lm=lm_HF, stop=STOP_WORDS)
    print("-- Questions using HFClientVLLM -- ")
    # ask_simple_questions(lm_HF)
    ask_Trec_Covid_questions(lm_HF)

    # TODO: not compatible w/ ask_Trec_Covid_questions
    # see this: https://discord.com/channels/1161519468141355160/1209871299854336060/1287887683976433705
    # lm = dspy.LM(
    #     model=f"openai/{model_name}",
    #     api_base="http://localhost:8000/v1",
    #     api_key="fake_key",
    #     model_type="completion",
    #     cache=False,
    # )
    # dspy.configure(lm=lm)
    # print("-- Questions using new dspy.LM -- ")
    # # ask_simple_questions(lm_HF)


if __name__ == "__main__":
    main()
