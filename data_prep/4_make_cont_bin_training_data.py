from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

score_system_message = "Given a query and a piece of text, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related."
rev_score_system_message = "Given a piece of text and a query, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related."

def filter_for_correct_scores_only(x):
    return bool(
        bool(
            x["label"] and bool(x["mean_exp_val_max7"] >= 4)
        ) or bool(
            bool(not x["label"]) and bool(x["mean_exp_val_max7"] <= 4)
        )
    )

format_text_query = lambda t, q: f"<<<Query>>>\n{q}\n\n<<<Context>>>\n{t}"
format_query_text = lambda t, q: f"<<<Context>>>\n{t}\n\n<<<Query>>>\n{q}"

def make_continuous_data(x):
    return {
        "conversations": [
            { "from": "system", "value": score_system_message },
            { "from": "human", "value":format_text_query(x["context"], x["question"])},
            { "from": "gpt", "value": str(int(x["mean_exp_val_max7_round"])) } ]
    }

def make_rev_continuous_data(x):
    return {
        "rev_conversations": [
            { "from": "system", "value": rev_score_system_message },
            { "from": "human", "value":format_query_text(x["context"], x["question"])},
            { "from": "gpt", "value": str(int(x["mean_exp_val_max7_round"])) } ]
    }

calc_exp_val = lambda probs: sum([(i+1) * (p / sum(probs)) for i, p in enumerate(probs)])

def main():
    ds = load_dataset("lightblue/rag_datasets_selected_32B4scored_probs", split="train")

    ds = ds.filter(lambda x: bool(sum(x["32B_score_probs"]) > 0) and bool(sum(x["32B_score_probs_rev"]) > 0), num_proc=32)

    ds = ds.map(
        lambda x: {
            "prob_exp_val": calc_exp_val(x["32B_score_probs"]),
            "rev_prob_exp_val": calc_exp_val(x["32B_score_probs_rev"]),
        }, num_proc=32
    )

    ds = ds.map(
        lambda x: {
            "mean_exp_val": np.array([x["prob_exp_val"], x["rev_prob_exp_val"]]).mean(),
        }, num_proc=32
    )

    ds = ds.map(
        lambda x: {
            "mean_exp_val_max7": ((x["mean_exp_val"] - 1) * (6/4)) + 1
        }, num_proc=32
    )

    ds = ds.map(
        lambda x: {
            "mean_exp_val_max7_round": round(x["mean_exp_val_max7"])
        }, num_proc=32
    )

    ds = ds.map(make_continuous_data, num_proc=32)
    ds = ds.map(make_rev_continuous_data, num_proc=32)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    MAX_LEN = 8192

    ds = ds.filter(
        lambda x: len(
            tokenizer.encode("\n".join([y["value"] for y in x["conversations"]]))
        ) < MAX_LEN, num_proc=32
    )

    ds = ds.shuffle()

    ds.filter(filter_for_correct_scores_only, num_proc=32).push_to_hub(
        "lightblue/reranker_continuous_filt_max7_train_extra", private=True
    )

if __name__ == '__main__':
    main()