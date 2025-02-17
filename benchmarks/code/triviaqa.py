from dotenv import load_dotenv

load_dotenv()

import pickle
import json
import numpy as np
from datasets import load_dataset

from niagara import Model, AnthropicClient, FireworksClient, OpenAIClient
from niagara import Chain, ModelIntrinsicLogProb, AskModelConfidence, TwoSidedAsymptoticLog, NullTransformation, LogisticRegressionCalibrator

### Access the data set

NAME = "triviaqa"

np.random.seed(45)

data = load_dataset("mandarjoshi/trivia_qa", "rc")

train_idx = np.random.choice(len(data['train']), 300, replace=False)
test_idx = np.random.choice(len(data['test']), 1000, replace=False)

data_train = [data['train'][int(i)] for i in train_idx]
data_test = [data['test'][int(i)] for i in test_idx]

# with open(f"./benchmarks/data/{NAME}/{NAME}_train.pkl", "wb") as file:
#     pickle.dump(data_train, file)
# with open(f"./benchmarks/data/{NAME}/{NAME}_test.pkl", "wb") as file:
#     pickle.dump(data_test, file)


triviaqa_zeroshot_prompt = """Correctly answer the question below. Give the answer directly, without writing a complete sentence.\n\nQuestion: {question}\n\nAnswer: """

triviaqa_system_prompt = """Correctly answer the given question. Answer the question directly without writing a complete sentence. Output just the answer, nothing else."""

def make_triviaqa_zeroshot_example(example):
    return triviaqa_zeroshot_prompt.format(question=example['question'])

def evaluate_triviaqa_answer(example, model_answer):

    question = example['question']

    eval_user_prompt = f"""Consider a proposed answer to the following trivia question: {question}. The proposed answer is {model_answer}. Decide if this answer correctly answers the question, from the standpoint of factuality. Output \"Y\" if the answer is factually correct, and \"N\" otherwise. Do not say anything else."""
    eval_model_name = "claude-3-5-sonnet"
    eval_system_prompt = """You are a helpful assistant who judges answers to trivia questions. Given a trivia question and a proposed answer, output \"Y\" if the proposed answers correctly answers the question. Otherwise, if the answer is not factually correct, output \"N\". Only output "\Y\" or \"N\". Do not say anything else."""
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

    assert eval_response.strip().upper() in {"Y", "N"}

    audit_data = {
        'eval_user_prompt': eval_user_prompt,
        'eval_system_prompt': eval_system_prompt, 
        'eval_response': eval_response,
        'eval_model_name': eval_model_name,
        'eval_temp': eval_temp,
    }

    return eval_response.strip().upper() == "Y", audit_data


ASK_MODEL_SYSPROMPT = """Your task is to determine if a proposed answer correctly answers a given trivia question, from the standpoint of factuality. If the answer is factually correct, output \"Y\". Otherwise, if the answer is not factually correct, output \"N\". Only respond with \"Y\" or \"N\". Do not say anything else."""
ASK_MODEL_USERPROMPT_TEMPLATE = """Consider a proposed answer to the following trivia question: 

<question>
{task_user_prompt}.
</question>

Does the following proposed answer correctly answer this question? 

Proposed answer: {proposed_response}

Respond with exactly \"Y\" if the proposed answer correctly answers the question. Otherwise, if the answer is not correct, respond with \"N\". Output only \"Y\" or \"N\", nothing else."""


llama_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=AskModelConfidence(
                eval_system_prompt=ASK_MODEL_SYSPROMPT, 
                eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE,
                allowed_outputs={"Y", "N", "y", "n"},
                max_new_tokens=1
            ),
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
            conf_signal=AskModelConfidence(
                eval_system_prompt=ASK_MODEL_SYSPROMPT, 
                eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE,
                allowed_outputs={"Y", "N", "y", "n"},
                max_new_tokens=1
            ),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator(),
            client=client
        )
        for name, client in [("gpt-4o-mini", OpenAIClient()), ("qwen2.5-32b-coder-instruct", FireworksClient()), ("qwen2.5-72b-instruct", FireworksClient()), ("gpt-4o", OpenAIClient())]
    ]
)


### Run the evaluation

from benchmark_utils import run_evaluation_with_restarts

chain, chain_name = llama_chain, "llama_chain"
chain, chain_name = qwen_oai_chain, "qwen_oai_chain"

if __name__ == '__main__':
    print("Running train evaluation...")
    train_results = run_evaluation_with_restarts(
        chain, 
        data_train, 
        system_prompt=triviaqa_system_prompt,
        make_example_fun=make_triviaqa_zeroshot_example,
        evaluate_answer_fun=evaluate_triviaqa_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_train.pkl",
        chunk_size=5,
    )
    print("Running test evaluation...")
    test_results = run_evaluation_with_restarts(
        chain, 
        data_test,
        system_prompt=triviaqa_system_prompt,
        make_example_fun=make_triviaqa_zeroshot_example,
        evaluate_answer_fun=evaluate_triviaqa_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_test.pkl",
        chunk_size=5,
    )



