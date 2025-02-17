### Access the data set

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
import pickle
import numpy as np

NAME = "xsum"

np.random.seed(45)

data = load_dataset("EdinburghNLP/xsum")

train_idx = np.random.choice(len(data['train']), 300, replace=False)
test_idx = np.random.choice(len(data['test']), 1000, replace=False)

data_train = [data['train'][int(i)] for i in train_idx]
data_test = [data['test'][int(i)] for i in test_idx]

# with open(f"./benchmarks/data/{NAME}/{NAME}_train.pkl", "wb") as file:
#     pickle.dump(data_train, file)
# with open(f"./benchmarks/data/{NAME}/{NAME}_test.pkl", "wb") as file:
#     pickle.dump(data_test, file)


from niagara import Model, AnthropicClient, FireworksClient, OpenAIClient
import json

xsum_zeroshot_prompt = """Summarize the given source document. Write a concise summary that is coherent, consistent, fluent, and relevant, as judged by the following criteria:

Coherence - collective quality of all sentences
Consistency - factual alignment between the summary and the source
Fluency - quality of individual sentences
Relevance - selection of important content from the source

Source document: {source_document}

Summary: """

xsum_system_prompt = """Summarize the given document. Output only the summarize, and nothing else. Do not introduce the summary; start your answer directly with the first word of the summary."""

def make_xsum_zeroshot_example(example):
    return xsum_zeroshot_prompt.format(source_document=example['document'])

def evaluate_xsum_answer(example, model_answer):

    source_document = example['document']

    eval_user_prompt = (f"""Consider a proposed summary of the following source document: {source_document}. Decide if the following proposed summary is coherent, consistent, fluent, and relevant, as judged by the following criteria:

    Coherence - collective quality of all sentences
    Consistency - factual alignment between the summary and the source
    Fluency - quality of individual sentences
    Relevance - selection of important content from the source
     
    Score each criterion (coherence, consistency, fluency, and relevance) on a scale from 1-5, where 5 is best. Return a JSON of the form """
    + '{"coherence": a, "consistency": b, "fluency": c, "relevance": d}' + 
    f", where a, b, c, d are the scores for the criteria (1-5). Only return this JSON.\n\nProposed summary: {model_answer}\n\nJSON containing the scores for all criteria:")
    eval_model_name = "claude-3-5-sonnet"
    eval_system_prompt = '''You are a helpful assistant who evaluates the quality of text summaries based on coherence, consistency, fluency, and relevance, as judged by the following criteria:
    
    Coherence - collective quality of all sentences
    Consistency - factual alignment between the summary and the source
    Fluency - quality of individual sentences
    Relevance - selection of important content from the source
    
    Score each criterion on a scale from 1-5 (5 is best). Only respond with a JSON. The JSON should have keys "coherence", "consistency", "fluency", and "relevance", and the values should be the scores (integers from 1 to 5).'''
    eval_temp = 0.0
    
    eval_response, _, _, _, _ = Model(
        model_name=eval_model_name,
        thresholds={"reject": 0.0, "accept": 0.0},
        client=AnthropicClient(),
    ).query_raw_model(
        system_prompt=eval_system_prompt,
        user_prompt=eval_user_prompt,
        temperature=eval_temp
    )

    # assert conformity with the schema
    dict_from_str = json.loads(eval_response.replace("'", '"'))
    ALLOWED_KEYS = { "consistency", "coherence", "fluency", "relevance" }
    ALLOWED_VALS = { 1,2,3,4,5 }
    assert np.all([ k in ALLOWED_KEYS and v in ALLOWED_VALS for k, v in dict_from_str.items() ])

    audit_data = {
        'eval_user_prompt': eval_user_prompt,
        'eval_system_prompt': eval_system_prompt, 
        'eval_response': eval_response,
        'eval_model_name': eval_model_name,
        'eval_temp': eval_temp,
    }

    return dict_from_str, audit_data


from hcma import Chain, ModelIntrinsicLogProb, AskModelConfidence, TwoSidedAsymptoticLog, NullTransformation, LogisticRegressionCalibrator
import pickle

ASK_MODEL_SYSPROMPT = """You are a helpful assistant who evaluates the quality of summaries by considering the following criteria:

Coherence - collective quality of all sentences
Consistency - factual alignment between the summary and the source
Fluency - quality of individual sentences
Relevance - selection of important content from the source
"""
ASK_MODEL_USERPROMPT_TEMPLATE = """Consider a proposed answer to the following query: {task_user_prompt}. The instructions were: {task_system_prompt}. Decide if the following proposed answer correctly answers the query. Only evaluate the final answer. For reference, the correct answer is provided below. Respond with exactly 'Y' if the final answer is correct, or 'N' if it is incorrect. Only output Y or N.\n\nProposed answer: {proposed_response}\n\nIs the given final answer correct? Respond with exactly Y or N:"""

llama_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=AskModelConfidence(eval_system_prompt=ASK_MODEL_SYSPROMPT, eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator()
        )
        for name in ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b"]
    ]
)

qwen_oai_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=AskModelConfidence(eval_system_prompt=ASK_MODEL_SYSPROMPT, eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator(),
            client=client
        )
        for name, client in [("gpt-4o-mini", OpenAIClient()), ("qwen2.5-32b-coder-instruct", FireworksClient()), ("qwen2.5-72b-instruct", FireworksClient()), ("gpt-4o", OpenAIClient())]
    ]
)

### Run evaluation

from benchmark_utils import run_evaluation_with_restarts

if __name__ == '__main__':

    chain, chain_name = llama_chain, "llama_chain"
    chain, chain_name = qwen_oai_chain, "qwen_oai_chain"

    train_results = run_evaluation_with_restarts(
        chain, 
        data_train, 
        system_prompt=xsum_system_prompt,
        make_example_fun=make_xsum_zeroshot_example,
        evaluate_answer_fun=evaluate_xsum_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_train.pkl",
        chunk_size=5
    )
    test_results = run_evaluation_with_restarts(
        chain, 
        data_test, 
        system_prompt=xsum_system_prompt,
        make_example_fun=make_xsum_zeroshot_example,
        evaluate_answer_fun=evaluate_xsum_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_test.pkl",
        chunk_size=5
    )