from datasets import load_dataset, concatenate_datasets, Dataset
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
from multiprocessing import Pool
import math

def get_hash(example):
    """Get hash of question field."""
    return {"q_hash": hash(example["question"])}
    
def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["q_hash"] in uniques:
        uniques.remove(example["q_hash"])
        return True
    else:
        return False

def remove_duplicates(ds):
    ds = ds.map(get_hash, num_proc=32)
    uniques = set(ds.unique("q_hash"))
    return ds.filter(check_uniques, fn_kwargs={"uniques": uniques})

def get_ds(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    MAX_LEN_MARGIN = 512
    MAX_LEN = 8192 - MAX_LEN_MARGIN

    ds = load_dataset("lightblue/rag_datasets_selected", split="train")
    ds = ds.add_column("row_id", list(range(len(ds))))
    ds = ds.shuffle()

    ### ADDED TO 32B ONLY
    print("Deduplicating")
    ds = remove_duplicates(ds)

    selected_columns = ['question', 'answer', 'dataset_name', 'language', 'added_neg', 'doc_id', 'added_doc_id', 'row_id']
    added_columns = ['context', 'label']

    ds = ds.map(lambda x: {
        "positives": [p for p in x["positives"] if len(tokenizer.encode(p)) < MAX_LEN],
        "negatives": [n for n in x["negatives"] if len(tokenizer.encode(n)) < MAX_LEN],
    }, num_proc=32)

    ds = ds.filter(lambda x: bool(len(x["positives"]) > 0) and bool(len(x["negatives"]) > 0), num_proc=32)

    pos_ds = ds.select_columns(selected_columns + ["positives"]).map(lambda x: {
        "context": random.sample(x["positives"], k=1)[0],
        "label": True,
    }, num_proc=32).select_columns(selected_columns + added_columns)

    neg_ds = ds.select_columns(selected_columns + ["negatives"]).map(lambda x: {
        "context": random.sample(x["negatives"], k=1)[0],
        "label": False,
    }, num_proc=32).select_columns(selected_columns + added_columns)

    new_ds = concatenate_datasets([pos_ds, neg_ds])

    return new_ds

def generate_responses(inputs):
    text_list, model_name, gpu_id = inputs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(model=model_name)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    system_message = "You are a relatedness rating assistant. Given a piece of text and a query, output the level that the query relates to the text. Your output should be single number between 1-5, with 1 meaning completely unrelated, 2 meaning mostly unrelated, 3 being unsure as to whether it is related or not, 4 being mostly related, and 5 being completely related."

    chats = [[
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{c}\n\n<<<Query>>>{q}"},
    ] for c, q in text_list]

    responses = llm.chat(chats, sampling_params)

    return [x.outputs[0].text.strip() for x in responses]

flatten = lambda xss: [x for xs in xss for x in xs]

if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    new_ds = get_ds(model_name)
    all_texts = list(zip(new_ds["context"], new_ds["question"]))

    num_gpus = 8
    batch_size = int(math.ceil(len(all_texts) / num_gpus))

    split_texts_w_idx = []

    for i in range(num_gpus):
        start_idx = i*batch_size
        end_idx = start_idx + batch_size
        split_texts_w_idx.append((all_texts[start_idx:end_idx], model_name, i))

    with Pool(num_gpus) as p:
        scores_split = p.map(generate_responses, split_texts_w_idx)

    scores = []

    for score_split in scores_split:
        scores.extend(score_split)

    new_ds = new_ds.add_column("32B_score", scores)

    new_ds = new_ds.sort("row_id")
    new_ds.to_parquet("lightblue__rag_datasets_selected_32B4scored.parquet")
    new_ds.push_to_hub("lightblue/rag_datasets_selected_32B4scored")