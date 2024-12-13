from tqdm.auto import tqdm
import torch
from FlagEmbedding import BGEM3FlagModel
from datasets import load_dataset, concatenate_datasets
from datasets.features.features import Value, Sequence
import numpy as np

flatten_list  = lambda ll: [x for l in ll for x in l]

def select_negatives(q_embedding, c_embeddings, context_df, doc_id_set, num_negs=10):
    sim = c_embeddings @ q_embedding

    context_df["sim"] = sim.cpu().numpy()

    context_df["is_pos"] = context_df["doc_id"].apply(lambda x: bool(set(x) & doc_id_set))

    sorted_context_df = context_df.sort_values("sim", ascending=False)

    negatives = sorted_context_df[~sorted_context_df["is_pos"]].iloc[:num_negs].positives.tolist()

    return negatives

def embed_text(text_list, model, max_len):
    return model.encode(
        text_list,
        max_length=max_len
    )['dense_vecs']

def send_array_to_gpu(array):
    return torch.Tensor(array).to(torch.device("cuda"))

def mine_negatives(ds, model):

    if ds[0]["negatives"] is None:
        ds = ds.add_column("added_neg", [True] * len(ds))
    else:
        ds = ds.add_column("added_neg", [False] * len(ds))
        print("No need to mine negatives")
        return ds

    if ds[0]["doc_id"] is None:
        doc_ids = [set([i]) for i in range(len(ds))]
        ds = ds.remove_columns(["doc_id"]).add_column("doc_id", doc_ids)
        ds = ds.add_column("added_doc_id", [True] * len(ds))
    else:
        ds = ds.add_column("added_doc_id", [False] * len(ds))

    context_df = ds.select_columns(
          ["positives", "doc_id"]
        ).to_pandas().explode("positives").groupby("positives").doc_id.apply(
            lambda x: set(flatten_list(x))
        ).reset_index(drop=False)

    context_df = context_df[~context_df.positives.isna()]
    context_df = context_df[context_df.positives.str.strip().str.len() > 0]

    if context_df.shape[0] < 1:
        print("Skipping because of no context")
        return None

    context_df["pos_len"] = context_df.positives.str.strip().str.len()
    context_df = context_df.sort_values("pos_len", ascending=False)

    c_embeddings = embed_text(context_df["positives"].tolist(), model, 8192)
    q_embeddings = embed_text(ds["question"], model, 8192)

    c_embeddings = send_array_to_gpu(c_embeddings)
    q_embeddings = send_array_to_gpu(q_embeddings)

    negatives_list = []
    num_negs = 10

    for q_embedding, doc_id in tqdm(zip(q_embeddings, ds["doc_id"]), total=len(ds)):
        negatives = select_negatives(q_embedding, c_embeddings, context_df, set(doc_id), num_negs=num_negs)
        negatives_list.append(negatives)

    ds = ds.remove_columns(["negatives"]).add_column(
        "negatives",
        negatives_list, 
        feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
    )

    return ds

def sample_dataset(ds):

    MAX_DATASET_LANG_ROWS = 25_000
    MAX_MQA_LANG_ROWS = 5_000

    ds = ds.filter(lambda x: isinstance(x["question"], str) and bool(len(x["question"].strip()) > 0) and bool(len(x["positives"]) > 0), num_proc=32)

    max_rows = MAX_MQA_LANG_ROWS if "mqa" in ds[0]["dataset_name"] else MAX_DATASET_LANG_ROWS
    ds = ds.shuffle().select(range(min(max_rows, len(ds))))

    return ds

def run_get_negatives(ds, model):

    lang_list = ds["language"]
    langs = sorted(set(lang_list))

    ds_list = []
    if len(langs) <= 1:
        lang_ds = sample_dataset(ds)
        ds_list = [mine_negatives(lang_ds, model)]
    else:
        lang_arr = np.array(lang_list)
        for lang in langs:
            print(lang)
            lang_idxs = np.where(lang == lang_arr)[0].tolist()
            lang_ds = ds.select(lang_idxs)
            lang_ds = sample_dataset(lang_ds)

            print(f"Length of {lang} is {len(lang_ds)}")

            ds_list.append(mine_negatives(lang_ds, model))
    
    ds_list = [x for x in ds_list if x is not None]

    if len(ds_list) < 1:
        return None

    return concatenate_datasets(ds_list)

if __name__ == "__main__":

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    original_ds = load_dataset("lightblue/rag_datasets_collection", split="train")

    dataset_names = sorted(set(original_ds["dataset_name"]))

    ds_list = []
    for dataset_name in dataset_names:

        print(dataset_name)
        single_ds = original_ds.filter(lambda x: x["dataset_name"] == dataset_name, num_proc=32)

        ds_w_negs = run_get_negatives(single_ds, model)
        ds_w_negs = ds_w_negs.map(lambda x: {
            "negatives": [y for y in x["negatives"] if y not in x["positives"]]
        }, num_proc=8)
        ds_w_negs = ds_w_negs.filter(lambda x: len(x["negatives"]) > 0, num_proc=32)

        if ds_w_negs is None:
            print(f"None dataset at {dataset_name}")
            continue

        ds_w_negs.to_parquet(f"./negatives/{dataset_name}.parquet")

        ds_list.append(ds_w_negs)

    concatenate_datasets(ds_list).push_to_hub("lightblue/rag_datasets_selected", private=True)
