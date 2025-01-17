import os
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

def run_evaluation(
        chain, 
        examples, 
        system_prompt,
        make_example_fun,
        evaluate_answer_fun,
        filename, 
        start_from_chunk_index=0, 
        chunk_size=100
    ):
    """
    Evaluate a chain. 

    make_example_fun(example)
    evaluate_answer_fun(example, answer)
    """

    def make_chunks(num_examples, chunk_size):
        main_output = [ 
            list(range(i*chunk_size, (i+1)*chunk_size)) for i in range(num_examples // chunk_size)
        ]
        extra_output = list(range(num_examples // chunk_size * chunk_size, num_examples))
        if len(extra_output) > 0:
            return main_output + [extra_output]
        else:
            return main_output
    
    if os.path.exists(filename): # retrieve existing progress
        with open(filename, "rb") as file:
            retrieved_results = pickle.load(file)
        system_prompts = retrieved_results['system_prompts']
        user_prompts = retrieved_results['user_prompts']
        model_answers = retrieved_results['model_answers']
        model_correctness = retrieved_results['model_correctness']
        raw_confidences = retrieved_results['raw_confidences']
        eval_audits = retrieved_results['eval_audits']
        # the new items:
        chain_responses = retrieved_results['chain_responses']
        model_costs = retrieved_results['model_costs']
        model_latencies = retrieved_results['model_latencies']
        model_tokens = retrieved_results['model_tokens']


    else: # initialize from scratch
        Path(filename).touch()
        system_prompts = []
        user_prompts = []

        chain_responses = []
        model_costs = { model.model_name: [] for model in chain.models }
        model_latencies = { model.model_name: [] for model in chain.models }
        model_tokens = { model.model_name: [] for model in chain.models }

        model_answers = { model.model_name: [] for model in chain.models }
        model_correctness = { model.model_name: [] for model in chain.models }
        raw_confidences = { model.model_name: [] for model in chain.models }
        eval_audits = { model.model_name: [] for model in chain.models }

    chunks = make_chunks(len(examples), chunk_size)
    chunks = chunks[start_from_chunk_index:]

    for chunk_idx, chunk in tqdm(enumerate(chunks)):
        for example in tqdm([ examples[idx] for idx in chunk ]):
            response = chain.answer_query(
                system_prompt=system_prompt,
                user_prompt=make_example_fun(example),
                do_not_calibrate=True,
                run_all_models=True,
                temperature=0.0
            )

            verdicts = [ 
                evaluate_answer_fun(example, ans) for ans in response.all_answers
            ]

            # Record prompts
            system_prompts.append(system_prompt)
            user_prompts.append(make_example_fun(example))

            # Add chain response only once
            chain_responses.append(response)

            # Record model specific data
            for i, model in enumerate(chain.models):
                model_answers[model.model_name].append(response.all_answers[i])
                model_correctness[model.model_name].append(verdicts[i][0])
                raw_confidences[model.model_name].append(response.all_confidences[i])
                eval_audits[model.model_name].append(verdicts[i][1])

                # add the new elements
                model_costs[model.model_name].append(response.all_costs[i])
                model_latencies[model.model_name].append(response.all_latencies[i])
                model_tokens[model.model_name].append(response.all_tokens[i])



        # Save interim results for this chunk

        results = {
            "chain": str(chain),
            "chain_responses": chain_responses,
            "system_prompts": system_prompts,
            "user_prompts": user_prompts,
            "model_answers": model_answers,
            "model_costs": model_costs,
            "model_latencies": model_latencies,
            "model_tokens": model_tokens,
            "model_correctness": model_correctness,
            "raw_confidences": raw_confidences,
            "eval_audits": eval_audits,
        }

        with open(filename, "wb") as file:
            pickle.dump(results, file)

        print(f"Saved chunk {chunk_idx+start_from_chunk_index+1}/{start_from_chunk_index+len(chunks)} to {filename}")

        # Relax the API
        time.sleep(1)

    return results


def run_evaluation_with_restarts(
        chain, examples, system_prompt, make_example_fun, evaluate_answer_fun,
        filename, chunk_size=50, start_from_chunk_index=0
    ):

    def get_restart_chunk_idx(fname, examples, chunk_size):
        with open(fname, "rb") as file:
            results = pickle.load(file)
            examples_saved = len(results['user_prompts'])
            chunks_saved = int(examples_saved/chunk_size)
        return chunks_saved

    max_chunk_idx = int(np.ceil(float(len(examples))/float(chunk_size)))
    start_chunk_idx = start_from_chunk_index
    while start_chunk_idx < max_chunk_idx:
        try:
            run_evaluation(
                chain, examples, system_prompt, make_example_fun, evaluate_answer_fun, 
                filename, chunk_size=chunk_size, start_from_chunk_index=start_chunk_idx
            )
            break
        except Exception as e:
            time.sleep(0.5)
            print(e)
            start_chunk_idx = get_restart_chunk_idx(filename, examples, chunk_size)
            print(f"Encountered error. Restarting from chunk {start_chunk_idx}...")
            time.sleep(0.5) # sleep to chill out the API