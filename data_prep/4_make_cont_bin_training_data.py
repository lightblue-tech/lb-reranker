from datasets import load_dataset
from transformers import AutoTokenizer

score_system_message = "Given a piece of text and a query, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related."
binary_system_message = "If the query is answerable based on the given context output True, else output False."

def filter_for_correct_scores_only(x):
    return bool(
        bool(
            x["label"] and bool(x["32B_score_mean_7base"] >= 3.5)
        ) or bool(
            bool(not x["label"]) and bool(x["32B_score_mean_7base"] < 3.5)
        )
    )

def make_continuous_data(x):
    return {
        "conversations": [
            { "from": "system", "value": score_system_message },
            { "from": "human", "value":format_text_query(x["context"], x["question"])},
            { "from": "gpt", "value": str(int(x["32B_score_mean_7base_round"])) } ]
    }

def make_binary_data(x):
    return {
        "conversations": [
            { "from": "system", "value": binary_system_message },
            { "from": "human", "value":format_text_query(x["context"], x["question"])},
            { "from": "gpt", "value": str(x["label"]) } ]
    }

def parse_int(int_str):
    try:
        score_int = int(int_str)
        if score_int < 1 or score_int > 5:
            return None
        return score_int
    except:
        return None
    
format_text_query = lambda t, q: f"{t}\n\n<<<Query>>>\n{q}"

def main():
    ds = load_dataset("lightblue/rag_datasets_selected_32B4scored_probs", split="train")

    ds = ds.map(lambda x: {"32B_score_probs_mean": np.array([
        x["32B_score_probs"], x["32B_score_probs_rev"]
        ]).mean(axis=0)}, num_proc=16)

    ds = ds.map(lambda x: {
        "32B_score_mean_exp_val": np.array(
            [(i+1) * x["32B_score_probs_mean"][i] for i in range(5)]
        ).sum()},
        num_proc=16)

    ds = ds.map(lambda x: {
        "32B_score_mean_7base": x["32B_score_mean_exp_val"] * (7/5)},
        num_proc=16)

    ds = ds.map(lambda x: {
        "32B_score_mean_7base_round": round(x["32B_score_mean_7base"])},
        num_proc=16)

    ds = ds.map(make_continuous_data, num_proc=32)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    MAX_LEN = 8192

    ds = ds.filter(
        lambda x: len(
            tokenizer.encode("\n".join([y["value"] for y in x["conversations"]]))
        ) < MAX_LEN, num_proc=32
    )

    ds = ds.shuffle()

    num_train_data = int(len(ds) * 0.95)

    tr_ds = ds.select(range(0, num_train_data))
    te_ds = ds.select(range(num_train_data, len(ds)))

    # tr_ds.push_to_hub(
    #     "lightblue/reranker_continuous_train", private=True
    # )

    # te_ds.push_to_hub(
    #     "lightblue/reranker_continuous_val", private=True
    # )

    tr_ds.filter(filter_for_correct_scores_only, num_proc=32).push_to_hub(
        "lightblue/reranker_continuous_filt_7base_train", private=True
    )

    te_ds.filter(filter_for_correct_scores_only, num_proc=32).push_to_hub(
        "lightblue/reranker_continuous_filt_7base_val", private=True
    )

    # tr_ds.filter(filter_for_correct_scores_only, num_proc=32).map(
    #     make_binary_data, num_proc=32
    # ).push_to_hub(
    #     "lightblue/reranker_binary_filt_train", private=True
    # )

    # te_ds.filter(filter_for_correct_scores_only, num_proc=32).map(
    #     make_binary_data, num_proc=32
    # ).push_to_hub(
    #     "lightblue/reranker_binary_filt_val", private=True
    # )

if __name__ == '__main__':
    main()