from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
import pickle
import numpy as np
from niagara import Model, AnthropicClient, FireworksClient, OpenAIClient, \
    Chain, ModelIntrinsicLogProb, AskModelConfidence, TwoSidedAsymptoticLog, \
        NullTransformation, LogisticRegressionCalibrator
from tqdm import tqdm
import time
import pickle
import os
import re

### Access the data set

NAME = "gsm8k"

np.random.seed(45)

data = load_dataset("openai/gsm8k", "main")

train_idx = np.random.choice(len(data['train']), 300, replace=False)
test_idx = np.random.choice(len(data['test']), 1000, replace=False)

data_train = [data['train'][int(i)] for i in train_idx]
data_test = [data['test'][int(i)] for i in test_idx]


### Write the prompt formatting functions

gsm8k_zeroshot_prompt = """Reason step-by-step through the following math problem. End your answer by outputting the final numerical answer prefixed by '#### '. For example, if your final answer is x, end your answer on '#### x'.\n\nProblem: {question}\n\nStep-By-Step Reasoning: """
gsm8k_system_prompt = "Consider the given math problem, think step-by-step, then report the final numerical answer x as follows '#### x'. Do not say anything after that."

def make_gsm8k_zeroshot_example(example):
    return gsm8k_zeroshot_prompt.format(question=example["question"])

def evaluate_gsm8k_answer(example, model_answer):

    question = example['question']
    correct_answer = example['answer']

    eval_user_prompt = f"""Consider a proposed answer to the following math problem: {question}. Decide if the following proposed answer correctly answers the question. Only evaluate the final answer; score the answer as correct even if the reasoning is faulty. For reference, the correct answer is provided below. Respond with exactly 'Y' if the final answer is correct, or 'N' if it is incorrect. Only output Y or N.\n\nProposed answer: {model_answer}\n\nCorrect answer:\n{correct_answer}\n\nIs the given final answer correct? Respond with exactly Y or N:"""
    eval_model_name = "claude-3-5-sonnet"
    eval_system_prompt = "You are a helpful assistant who evaluates answers. Only respond with Y or N."
    eval_temp = 0.0
    
    eval_response, _, _, _, _ = Model(
        model_name=eval_model_name,
        thresholds={"reject": 0.0, "accept": 0.0},
        client=AnthropicClient(),
    ).query_raw_model(
        system_prompt=eval_system_prompt,
        user_prompt=eval_user_prompt,
        max_new_tokens=1,
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


### Define the chain

# ASK_MODEL_SYSPROMPT = """Your job is to carefully evaluate solutions to math problems. Think step-by-step, then end your response on '#### Correct' or '#### Incorrect' depending on whether the provided solution is correct. Do not say anything after that."""
# ASK_MODEL_USERPROMPT_TEMPLATE = """Consider a proposed answer to the following math problem:

# <question>
# {task_user_prompt}
# </question>

# Decide if the following proposed solution is correct. 

# Proposed solution: {proposed_response}

# Analyze the reasoning in the proposed solution step-by-step. End your answer on \"#### Correct\" or \"#### Incorrect\", depending on whether the proposed solution is correct. Do not say anything after that.

# Step-by-step analysis: """



# ZeroShot Prompts:

ASK_MODEL_SYSPROMPT = """Your job is to judge if a solution to a math problem is correct. If the solution is correct, output "Y". Otherwise, output "N". Only output "Y" or "N", nothing else."""
ASK_MODEL_USERPROMPT_TEMPLATE = """Consider a proposed solution to the following math problem:

<question>
{task_user_prompt}
</question>

Proposed solution: {proposed_response}

Decide if the proposed solution is correct. Only say "Y" or "N", nothing else.

Correct? """


def correctness_extractor(response, answer):
    # Step 0: check for "#### correct" or "#### incorrect"
    match = re.search(r'####\s+(Correct|correct|Incorrect|incorrect)', response)
    if match:
        match = match.group(1) 
        return "Y" if match.lower() == "correct" else "N"

    # Step 1: search for "incorrect" anywhere in the output
    if "incorrect" in response:
        return "N"

    # Step 2: see if the answer contains "#### x" where x is a number. 
    #   If so, extract x and compare against the "#### x" from the proposed solution.
    match_eval = re.search(r'####\s*\$?(-?\d*\.?\d+)', response)
    match_orig = re.search(r'####\s*\$?(-?\d*\.?\d+)', answer)
    if match_eval and match_orig:
        float_ans = float(match_eval.group(1))
        original_ans = float(match_orig.group(1))
        return "Y" if float_ans == original_ans else "N"
    
    # Step 3: see if the answer contains "correct"
    if "correct" in response:
        return "Y"

    return response

llama_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=AskModelConfidence(eval_system_prompt=ASK_MODEL_SYSPROMPT, eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE, allowed_outputs={"Y", "N", "y", "n"}, max_new_tokens=1, extract_fun=None),
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
            conf_signal=AskModelConfidence(eval_system_prompt=ASK_MODEL_SYSPROMPT, eval_user_prompt_template=ASK_MODEL_USERPROMPT_TEMPLATE, allowed_outputs={"Y", "N", "y", "n"}, max_new_tokens=1, extract_fun=None),
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
    print("Running on training data...")
    train_results = run_evaluation_with_restarts(
        chain, 
        data_train, 
        system_prompt=gsm8k_system_prompt,
        make_example_fun=make_gsm8k_zeroshot_example,
        evaluate_answer_fun=evaluate_gsm8k_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_train.pkl",
        chunk_size=5
    )
    print("Running on testing data...")
    test_results = run_evaluation_with_restarts(
        chain, 
        data_test, 
        system_prompt=gsm8k_system_prompt,
        make_example_fun=make_gsm8k_zeroshot_example,
        evaluate_answer_fun=evaluate_gsm8k_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_test.pkl",
        chunk_size=5
    )