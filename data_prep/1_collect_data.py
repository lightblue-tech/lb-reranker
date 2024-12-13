from datasets import load_dataset, concatenate_datasets, Dataset
from datasets.features.features import Features, Value, Sequence
import re
from tqdm.auto import tqdm
import pandas as pd
import ast
from cryptography.fernet import Fernet
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from io import StringIO
from functools import partial
from huggingface_hub import hf_hub_download
tqdm.pandas()

def prepare_hotpotqa():
    # Concat all relevant contexts together as our one answerable context
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train", trust_remote_code=True)

    ds = ds.map(lambda x: {
        "positives": ["\n".join([t] + s) for t, s in zip(x["context"]["title"], x["context"]["sentences"]) if t in x["supporting_facts"]["title"]],
        "negatives": ["\n".join([t] + s) for t, s in zip(x["context"]["title"], x["context"]["sentences"]) if t not in x["supporting_facts"]["title"]],
        "dataset_name": "hotpot_qa",
        "language": "en",
        "doc_id": None,
    })
    
    # add all Hotpot positive contexts together as questions require all contexts to be answered fully
    ds = ds.map(lambda x: {
        "positives": ["\n".join(x["positives"])],
    }, num_proc=32)
    
    return ds

def get_trivia_qa_contexts(row):
    contexts = []
    if len(row["entity_pages"]["wiki_context"]) > 0:
        for filename, title, context in zip(row["entity_pages"]["filename"], row["entity_pages"]["title"], row["entity_pages"]["wiki_context"]):
            contexts.append(f"{filename}\n{title}\n{context}")

    if len(row["search_results"]["search_context"]) > 0:
        for title, description, context in zip(row["search_results"]["title"], row["search_results"]["description"], row["search_results"]["search_context"]):
            contexts.append(f"{title}\n{description}\n{context}")
    return contexts

def prepare_triviaqa():
    
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
    
    ds = ds.map(lambda x: {
        "answer": x["answer"]["value"],
        "positives": get_trivia_qa_contexts(x),
        "negatives": None,
        "dataset_name": "trivia_qa",
        "language": "en",
        "doc_id": None
    }, num_proc=16)
    
    return ds

# This dataset is included in our MLQA implementation
# def prepare_squad():
    
#     ds = load_dataset("rajpurkar/squad", split="train")

#     ds = ds.map(lambda x: {
#         "answer": x["answers"]["text"][0],
#         "positives": [x["title"] + "\n" + x["context"]],
#         "negatives": None,
#         "dataset_name": "squad",
#         "language": "en",
#         "doc_id": set([x["title"]]),
#     }, num_proc=16)
    
#     return ds

def prepare_pubmedqa():
    
    ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train")

    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["long_answer"],
        "positives": ["\n".join(x["context"]["contexts"])],
        "negatives": None,
        "dataset_name": "pubmedqa",
        "language": "en",
        "doc_id": None,
    }, num_proc=16)
    
    return ds

def get_mldr_single_lang(lang):
    return load_dataset("Shitao/MLDR", lang, split="train", trust_remote_code=True).map(lambda x: {
        "question": x["query"],
        "answer": None,
        "positives": [y["text"] for y in x["positive_passages"]],
        "negatives": [y["text"] for y in x["negative_passages"]],
        "dataset_name": "mldr",
        "language": lang,
        "doc_id": None,
    }, num_proc=16)

def prepare_mldr():
    
    langs = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']

    ds = concatenate_datasets([get_mldr_single_lang(l) for l in tqdm(langs)])
    
    return ds

def get_scandi_qa_single_lang(lang):
    ds = load_dataset("alexandrainst/scandi-qa", lang, split="train")
    df = ds.to_pandas()
    grouped_df = df.groupby("question").apply(lambda x: {
        "answer": x["answers"].apply(lambda y: y["text"][0]).tolist()[0],
        "positives": x[
            x["answers"].apply(lambda y: y["answer_start"][0] != -1)
        ]["context"].tolist(),
        "doc_id": set(x[
            x["answers"].apply(lambda y: y["answer_start"][0] != -1)
        ]["title_en"].tolist()),
        "negatives": None}
    )

    joined_df = pd.DataFrame(grouped_df.tolist())
    joined_df["question"] = grouped_df.index
    joined_df = joined_df[["question", "answer", "positives", "negatives", "doc_id"]]
    joined_df["answer"] = joined_df["answer"].apply(lambda x: x if len(x) > 0 else None)
    joined_df = joined_df[~joined_df["answer"].isna()]
    joined_df["dataset_name"] = "scandi_qa"
    joined_df["language"] = lang

    return Dataset.from_pandas(joined_df)

def prepare_scandiqa():
    langs = ['da', 'no', 'sv']

    ds = concatenate_datasets([get_scandi_qa_single_lang(l) for l in tqdm(langs)])
    
    return ds

def prepare_logqa():
    ds = load_dataset(
        "json",
        data_files={
            "train": "https://raw.githubusercontent.com/LogQA-dataset/LogQA/refs/heads/main/data/HDFS/qa.json.train"
        },
        split="train"
    )

    ds = ds.map(lambda x: {
        "question": x["Question"],
        "answer": x["Answer"],
        "positives": [x["RawLog"]],
        "negatives": None,
        "dataset_name": "logqa",
        "language": "en",
        "doc_id": None,
    }, num_proc=16)
    
    return ds

def prepare_cpgqa():
    ds = load_dataset(
        "json",
        data_files={
            "train": "https://raw.githubusercontent.com/mmahbub/cpgQA/refs/heads/main/dataset/cpgQA-v1.0.json"
        },
        split="train"
    )

    ds = ds.map(lambda x: {
        "question": x["data"]["paragraphs"]["qas"][0]["question"],
        "answer": x["data"]["paragraphs"]["qas"][0]["answers"][0]["text"],
        "positives": [x["data"]["paragraphs"]["context"]],
        "negatives": None,
        "dataset_name": "cpgqa",
        "language": "en",
        "doc_id": None,
    }, num_proc=16)
    
    return ds

def prepare_sleepqa():
    ds = load_dataset(
        "json",
        data_files={
            "train": "https://raw.githubusercontent.com/IvaBojic/SleepQA/refs/heads/main/data/training/sleep-train.json"
            },
        split="train"
    )

    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["answers"][0],
        "positives": [y["title"] + "\n" + y["text"] for y in x["positive_ctxs"]],
        "negatives": [y["title"] + "\n" + y["text"] for y in x["negative_ctxs"]],
        "dataset_name": "sleepqa",
        "language": "en",
        "doc_id": set([y["title"] for y in x["positive_ctxs"]]),
    }, num_proc=16)

    return ds

def prepare_jqara():
    ds = load_dataset("hotchpotch/JQaRA", split="dev")
    
    df = ds.to_pandas()
    df["text"] = df["title"] + "\n" + df["text"]

    grouped_series = df.groupby("question").apply(lambda x: {
        "answer": x["answers"].tolist()[0][0],
        "positives": x[x["label"] == 1]["text"].tolist(),
        "negatives": x[x["label"] == 0]["text"].tolist(),
        "doc_id": set(x[x["label"] == 1]["title"].tolist()),
    })

    joined_df = pd.DataFrame(grouped_series.tolist())
    joined_df["question"] = grouped_series.index
    joined_df["dataset_name"] = "jqara"
    joined_df["language"] = "ja"

    ds = Dataset.from_pandas(joined_df)
    
    return ds

def get_indicqa_single_lang(lang):
    ds = load_dataset("ai4bharat/IndicQA", lang, split="test", trust_remote_code=True).filter(lambda x: len(x["answers"]["text"][0]))

    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["answers"]["text"][0],
        "positives": [x["context"]],
        "negatives": None,
        "dataset_name": "indicqa",
        "language": lang.split(".")[-1],
        "doc_id": None,
    }, num_proc=16)

    return ds

def prepare_indicqa():
    langs = ['indicqa.as', 'indicqa.bn', 'indicqa.gu', 'indicqa.hi', 'indicqa.kn', 'indicqa.ml', 'indicqa.mr', 'indicqa.or', 'indicqa.pa', 'indicqa.ta', 'indicqa.te']

    ds = concatenate_datasets([get_indicqa_single_lang(l) for l in tqdm(langs)])
    
    return ds

def prepare_qasports():
    ds = load_dataset("PedroCJardim/QASports", "all", split="train").filter(
        lambda x: isinstance(x["answer"], str) and len(x["answer"]) > 0, 
        num_proc=16
    )

    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": ast.literal_eval(x["answer"])["text"],
        "positives": [x["context"]],
        "negatives": None,
        "dataset_name": "qasports",
        "language": "en",
        "doc_id": set([x["context_id"]]),
    }, num_proc=16)

    return ds

def prepare_lsat():
    # Add multiple choice answers to question
    ds = concatenate_datasets([load_dataset(
        "json",
        data_files={
            "train": f"https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/refs/heads/main/complete_lsat_data/train_{x}.json"
            },
        split="train"
    ) for x in ["ar", "lr", "rc"]])

    ds = ds.map(lambda x: {
        "question": "\n".join([x["question"]] + x["answers"]),
        "answer": x["answers"][x["label"]],
        "positives": [x["context"]],
        "negatives": None,
        "dataset_name": "lsat",
        "language": "en",
        "doc_id": set([x["context"]]),
    }, num_proc=16)

    return ds

def parse_squad(row):
    return {
        "positives": [row["context"].strip()],
        "question": row["question"].strip(),
        "answer": row["answers"]["text"][0].strip()
    }

def prepare_m2qa():

    lang_dict = {
        "chinese": "zh",
        "german": "de",
        "turkish": "tr",
    }

    domains = [
        "creative_writing",
        "news",
        "product_reviews"
    ]
    
    ds_list = []

    for lang in tqdm(lang_dict):
        ds = concatenate_datasets([
            load_dataset(
                "UKPLab/m2qa",
                f"m2qa.{lang}.{x}",
                split="validation",
                trust_remote_code=True
                ) for x in domains])

        ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0, num_proc=16)

        # Decrypt it
        fernet = Fernet(b"aRY0LZZb_rPnXWDSiSJn9krCYezQMOBbGII2eGkN5jo=")

        def decrypt(example):
            example["question"] = fernet.decrypt(example["question"].encode()).decode()
            example["context"] = fernet.decrypt(example["context"].encode()).decode()
            example["answers"]["text"] = [fernet.decrypt(answer.encode()).decode() for answer in example["answers"]["text"]]
            return example

        ds = ds.map(decrypt)
        ds = ds.map(parse_squad)
        ds = ds.map(lambda x: {
            "negatives": None,
            "dataset_name": "m2qa",
            "language": lang_dict[lang],
            "doc_id": set(["_".join(x["id"].split("_")[:-1])]),
        })
        
        ds_list.append(ds)

    return concatenate_datasets(ds_list)

def get_mlqa_dataset_list():
    dataset_list = [
        load_dataset("rajpurkar/squad", split="train").map(lambda x: {"language": "en"}, num_proc=16)
    ]

    langs = ["ar", "de", "es", "hi", "vi", "zh"]

    dataset_list = dataset_list + [
        load_dataset(
            "facebook/mlqa",
            f"mlqa-translate-train.{l}",
            split="train",
            trust_remote_code=True
        ).map(lambda x: {"language": l}, num_proc=16) for l in langs 
    ]
    
    return dataset_list

def match_crossling(dataset_list, dataset_name, title_column="title"):
    dataset_dicts = [
        {
            x["id"]: x for x in d
        } for d in tqdm(dataset_list)
    ]
    
    id_set = set()

    for d in dataset_list:
        id_set.update(set(d["id"]))
    
    cross_rows = []

    for row_id in tqdm(id_set):
        rows = [x[row_id] for x in dataset_dicts if row_id in x]

        title = [x[title_column] for x in rows if x["language"] == "en"][0]
        contexts = [x["context"] for x in rows]

        for row in rows:
            cross_rows.append({
                "question": row["question"],
                "answer": row["answers"]["text"][0],
                "positives": contexts,
                "negatives": None,
                "dataset_name": dataset_name,
                "language": row["language"],
                "doc_id": set([title]),
            })
            
    return cross_rows

def prepare_mlqa():
    
    dataset_list = get_mlqa_dataset_list()
        
    cross_rows = match_crossling(dataset_list, "mlqa", title_column="title")
    
    return Dataset.from_pandas(pd.DataFrame(cross_rows))
    
def prepare_xquad():

    dataset_dict = {}

    langs = ['ar', 'de', 'el', 'en', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']
    
    dataset_list = [
        load_dataset(
            "google/xquad",
            f"xquad.{l}",
            split="validation",
            trust_remote_code=True
        ).map(lambda x: {"language": l}, num_proc=16) for l in langs 
    ]
        
    cross_rows = match_crossling(dataset_list, "xquad", title_column="context")

    return Dataset.from_pandas(pd.DataFrame(cross_rows))

def parse_tydi_from_bytes(text, start, end):
    try:
        return text.encode("utf-8")[start:end].decode("utf-8")
    except:
        return None

def prepare_tydiqa_goldp():

    ds = load_dataset("google-research-datasets/tydiqa", "primary_task", split="train").filter(
        lambda x: bool(x["annotations"]["minimal_answers_start_byte"][0] != -1),
        num_proc=16
    )

    ds = ds.map(lambda x: {
        "contexts": [
            parse_tydi_from_bytes(x["document_plaintext"], s, e) for s, e in zip(
            x["passage_answer_candidates"]["plaintext_start_byte"], 
            x["passage_answer_candidates"]["plaintext_end_byte"]
        )],
        "answer": parse_tydi_from_bytes(
            x["document_plaintext"], 
            x["annotations"]["minimal_answers_start_byte"][0], 
            x["annotations"]["minimal_answers_end_byte"][0]),
        "question": x["question_text"],
        }, num_proc=16)

    ds = ds.map(lambda x: {
        "positives": [x["contexts"][x["annotations"]["passage_answer_candidate_index"][0]]],
        "negatives": [x["contexts"][i] for i in range(len(x["contexts"])) if i != x["annotations"]["passage_answer_candidate_index"][0]],
    }, num_proc=16)

    language_code_dict = {
        'arabic': 'ar',
        'bengali': 'bn',
        'english': 'en',
        'finnish': 'fi',
        'indonesian': 'id',
        'japanese': 'ja',
        'korean': 'ko',
        'russian': 'ru',
        'swahili': 'sw',
        'telugu': 'te',
        'thai': 'th'
    }

    ds = ds.map(lambda x: {
        "dataset_name": "tydi",
        "language": language_code_dict[x["language"]],
        "doc_id": set([x["document_title"]]),
    }, num_proc=16)
    
    return ds

def prepare_skquad():
    ds = load_dataset("TUKE-DeutscheTelekom/skquad", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "skquad",
        "language": "sk",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_arcd():
    ds = load_dataset("hsseinmz/arcd", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "arcd",
        "language": "ar",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_persianqa():
    ds = load_dataset("SajjadAyoubi/persian_qa", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "persianqa",
        "language": "fa",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_amharicqa():
    df = pd.read_json("https://raw.githubusercontent.com/semantic-systems/amharic-qa/main/train_data.json")
    df = pd.DataFrame(pd.DataFrame(df.data.tolist()).explode("paragraphs").paragraphs.tolist()).explode("qas")
    df["question"] = df.qas.apply(lambda x: x["question"])
    df["answer"] = df.qas.apply(lambda x: x["answers"][0]["text"])
    df["positives"] = df.context.apply(lambda x: [x])
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "amharicqa",
        "language": "am",
        "doc_id": set([x["document_id"]]),
    }, num_proc=16)
    return ds

def prepare_chaii():
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files('chaii-hindi-and-tamil-question-answering', path='.')

    zip_path = './chaii-hindi-and-tamil-question-answering.zip'

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open("train.csv") as file:
            content = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(content))

    ds = Dataset.from_pandas(df)

    language_map = {
        "tamil": "ta",
        "hindi": "hi",
    }

    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["answer_text"],
        "positives": [x["context"]],
        "negatives": None,
        "dataset_name": "chaii",
        "language": language_map[x["language"]],
        "doc_id": set([x["id"]]),
    })
    
    return ds

def prepare_sberquad():
    ds = load_dataset("kuznetsoffandrey/sberquad", split="train", trust_remote_code=True)
    ds = ds.filter(lambda x: bool(len(x["answers"]["text"]) > 0) and bool(len(x["answers"]["text"][0]) > 0), num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "sberquad",
        "language": "ru",
        "doc_id": set([x["id"]]),
    }, num_proc=16)
    return ds

def prepare_pira():
    ds = load_dataset("paulopirozelli/pira", "default", split="train")

    en_ds = ds.map(lambda x: {
            "positives": [x["abstract"].strip()],
            "negatives": None,
            "question": x["question_en_origin"].strip(),
            "answer": x["answer_en_origin"].strip(),
            "dataset_name": "pira",
            "language": "en",
            "doc_id": set([x["id_qa"]]),
        }, num_proc=16)
    pt_ds = ds.map(lambda x: {
            "positives": [x["abstract_translated_pt"].strip()],
            "negatives": None,
            "question": x["question_pt_origin"].strip(),
            "answer": x["answer_pt_origin"].strip(),
            "dataset_name": "pira",
            "language": "pt",
            "doc_id": set([x["id_qa"]]),
        }, num_proc=16)
    return concatenate_datasets([en_ds, pt_ds])

def parse_jsquad(row):
    row_data = []
    title = row["title"]
    paragraphs = row["paragraphs"]
    
    for p in paragraphs:
        context = p["context"].replace("[SEP]", "\n")
        questions = p["qas"]
        
        for q in questions:
            is_impossible = q["is_impossible"]
            if is_impossible:
                continue
            question = q["question"]
            answer = q["answers"][0]["text"]
            
            row_data.append({
                "question": question,
                "answer": answer,
                "positives": [context],
                "negatives": None,
                "dataset_name": "jsquad",
                "language": "ja",
                "doc_id": set([title]),
            })
            
    return row_data

def prepare_jsquad():
    df = pd.read_json(
        "https://github.com/yahoojapan/JGLUE/raw/refs/heads/main/datasets/jsquad-v1.1/train-v1.1.json"
    )
    
    df = pd.DataFrame(df.data.apply(parse_jsquad).explode().tolist())
    ds = Dataset.from_pandas(df)
    ds = ds.filter(lambda x: bool(
            len(x["question"].strip()) > 0
        ) and bool(
            len(x["answer"].strip()) > 0
        ) and bool(
            len(x["positives"][0].strip()) > 0
        ), num_proc=16)
    return ds

def prepare_korquad():
    ds = load_dataset("KorQuAD/squad_kor_v1", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "korquad",
        "language": "ko",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def parse_nested(df):
    df = pd.DataFrame(df.data.apply(lambda x: [dict(**y, title=x["title"]) for y in x["paragraphs"]]).explode())
    df = df[~df.data.isna()]
    df["positives"] = df.data.apply(lambda x: [x["title"] + "\n" + x["context"]])
    df["data"] = df.data.apply(lambda x: [dict(**y, title=x["title"]) for y in x["qas"]])
    df = df.explode("data")
    df["title"] = df["data"].apply(lambda x: x["title"] if isinstance(x, dict) else None)
    df["question"] = df["data"].apply(lambda x: x["question"] if isinstance(x, dict) else None)
    df["answer"] = df["data"].apply(lambda x: x["answers"][0]["text"] if isinstance(x, dict) else None)
    df = df.dropna()
    ds = Dataset.from_pandas(df)
    return ds

def prepare_tquad():
    df = pd.read_json("https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master/train-v0.1.json")
    ds = parse_nested(df)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "tquad",
        "language": "tr",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_sqac():
    df = pd.read_json("https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/train.json")
    ds = parse_nested(df)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "sqac",
        "language": "es",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_germanquad():
    ds = load_dataset("deepset/germanquad", split="train", trust_remote_code=True)
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "germanquad",
        "language": "de",
        "doc_id": set([x["id"]]),
    }, num_proc=16)
    return ds

def prepare_kenswquad():
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    # max_tok_size = 1_500

    ds = load_dataset("lightblue/KenSwQuAD", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16)
    # ds = ds.filter(lambda x: len(tokenizer.encode(x["context"])) < max_tok_size, num_proc=16)

    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "kenswquad",
        "language": "sw",
        "doc_id": set([x["Story_ID"]]),
    }, num_proc=16)
    return ds

def prepare_drcd():
    ds = load_dataset("voidful/DRCD", split="train")
    ds = parse_nested(pd.DataFrame({"data": ds.to_list()}))
    ds = ds.map(lambda x: {
        "negatives": None,
        "dataset_name": "drcd",
        "language": "zh",
        "doc_id": set([x["title"]]),
    }, num_proc=16)
    return ds

def prepare_narrativeqa():

    ds = load_dataset("deepmind/narrativeqa", split="train")
    
    ds = ds.map(
        lambda x: {
            "positives": [x["document"]["summary"]["text"].strip()],
            "negatives": None,
            "question": x["question"]["text"].strip(),
            "answer": x["answers"][0]["text"],
            "dataset_name": "narrativeqa",
            "language": "en",
            "doc_id": set([x["document"]["summary"]["title"].strip()]),
            }, num_proc=16
        )
    
    return ds

def get_lb_rewording(text):
    pattern = r"### Reworded Text\n(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        selected_text = match.group(1)
        return selected_text
    else:
        return None
    
def get_lb_positives(row):
    positives = []
    
    positives.append(row["selected_chunk"])
    
    if row["rewording_finish_reason"] == "stop":
        reworded_context = get_lb_rewording(row["rewording_response"])
        if reworded_context is not None:
            positives.append(reworded_context)
            
    if row["otherlang_rewording_finish_reason"] == "stop":
        otherlang_reworded_context = get_lb_rewording(row["otherlang_rewording_response"])
        if otherlang_reworded_context is not None:
            positives.append(otherlang_reworded_context)
            
    return positives

def prepare_lb_rag():

    language_map = {'Amharic': 'am', 'Arabic': 'ar', 'Bulgarian': 'bg', 'Bengali': 'bn', 'Czech': 'cs', 'Danish': 'da', 'German': 'de', 'Greek': 'el', 'English': 'en', 'Spanish': 'es', 'Persian': 'fa', 'Finnish': 'fi', 'French': 'fr', 'Gujarati': 'gu', 'Hausa': 'ha', 'Hindi': 'hi', 'Hungarian': 'hu', 'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jv', 'Kannada': 'kn', 'Korean': 'ko', 'Lithuanian': 'lt', 'Marathi': 'mr', 'Dutch': 'nl', 'Norwegian': 'no', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 'Swedish': 'sv', 'Swahili': 'sw', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Tagalog': 'tl', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Vietnamese': 'vi', 'Yoruba': 'yo', 'Chinese': 'zh'}
    
    ds_list = []
    
    for lang in sorted(language_map.values()):
        print(lang)
        ds = load_dataset(
            "lightblue/rag_multilingual_training_negatives", lang, split="train"
        )
        
        # Multilingual
        muling_ds = ds.filter(lambda x: bool(len(x["otherlang_question"]) > 0) and bool(len(x["otherlang_answer"]) > 0) and bool(x["otherlang_qa_finish_reason"] == "stop"), num_proc=16)

        muling_ds = muling_ds.map(
            lambda x: {
                "question": x["otherlang_question"],
                "answer": x["otherlang_answer"],
                "positives": get_lb_positives(x),
                "negatives": x["multilingual_negatives"],
                "dataset_name": "lb_rag_multilingual",
                "language": language_map[x["other_qa_lang"]],
                "doc_id": None,
            }, num_proc=16
        )
        
        # Monolingual
        moling_ds = ds.filter(lambda x: bool(len(x["question"]) > 0) and bool(len(x["answer"]) > 0) and bool(x["raw_qa_finish_reason"] == "stop"), num_proc=16)

        moling_ds = moling_ds.map(
            lambda x: {
                "question": x["question"],
                "answer": x["answer"],
                "positives": get_lb_positives(x),
                "negatives": x["monolingual_negatives"],
                "dataset_name": "lb_rag_monolingual",
                "language": x["language"],
                "doc_id": None,
            }, num_proc=16
        )
        
        ds_list.append(muling_ds)
        ds_list.append(moling_ds)
        
    return concatenate_datasets(ds_list)

def parse_mqa_text(name, text):
    name = "" if name is None else name
    text = "" if text is None else text
    namelower = name.lower().strip()
    textlower = text.lower().strip()
    
    question_text = ""
    question_text += name
    if namelower != textlower:
        question_text += "\n" + text
        
    question_text = re.sub(r"[\=\-\#]{3,}", "", question_text)
    return question_text.strip()

def process_mqa(lang, data_type):
    answer_features = [{'downvote_count': Value(dtype='int64', id=None),
         'is_accepted': Value(dtype='bool', id=None),
         'name': Value(dtype='string', id=None),
         'text': Value(dtype='string', id=None),
         'upvote_count': Value(dtype='int64', id=None)}]
    
    question_features = {'answers': answer_features,
       'comment_count': Value(dtype='int64', id=None),
       'data_type': Value(dtype='string', id=None),
       'downvote_count': Value(dtype='int64', id=None),
       'hash': Value(dtype='string', id=None),
       'name': Value(dtype='string', id=None),
       'text': Value(dtype='string', id=None),
       'upvote_count': Value(dtype='int64', id=None)}
    
    load_features = {'bucket': Value(dtype='float64', id=None),
     'sub_bucket': Value(dtype='string', id=None),
     'language': Value(dtype='string', id=None),
     'hreflang_alternates': [{'href': Value(dtype='string', id=None),
       'hreflang': Value(dtype='string', id=None)}],
     'questions': [question_features],
     'page_hash': Value(dtype='string', id=None),
     'fasttext_language': Value(dtype='string', id=None),
     'domain': Value(dtype='string', id=None)}

    filename = hf_hub_download(repo_id="clips/mqa", filename=f"data/data.{lang}.{data_type}.json.gz", repo_type="dataset")
    ds = load_dataset("json", data_files={"train": filename}, split="train", features=Features(load_features))
    
    # Randomly sample at maximum 100K rows to make this processing tractable
    max_rows = 100_000
    ds = ds.shuffle().select(range(min(max_rows, len(ds))))
    
    load_features["questions"] = question_features
    explode_features = Features(load_features)

    explode_mqa = lambda x: pd.DataFrame(dict(x)).explode("questions").to_dict(orient="list")

    ds = ds.map(explode_mqa, 
                batched=True, 
                batch_size=1000, 
                num_proc=32, 
                remove_columns=ds.column_names, 
                features=explode_features)
    
    ds = ds.filter(lambda x: bool(
            isinstance(x["language"], str)
        ) and bool(
            isinstance(x["fasttext_language"], str)
        ) and bool(
            lang in x["language"].lower()
        ) and bool(
            lang in x["fasttext_language"].lower()
        ), 
        num_proc=32
    )
    
    load_features["accepted_answer"] = answer_features
    accepted_features = Features(load_features)
    
    ds = ds.map(lambda x: {
        "accepted_answer": [y for y in x["questions"]["answers"] if y["is_accepted"]],
    }, num_proc=32, features=accepted_features)

    ds = ds.filter(lambda x: len(x["accepted_answer"]) > 0, num_proc=32)

    ds = ds.map(lambda x: {
        "question": parse_mqa_text(x["questions"]["name"], x["questions"]["text"]),
        "answer": None,
        "positives": [parse_mqa_text(x["accepted_answer"][0]["name"], x["accepted_answer"][0]["text"])],
        "negatives": None,
        "dataset_name": f"mqa_{data_type}",
        "language": lang,
        "doc_id": set([x["domain"]]),
    }, num_proc=32)
    
    return ds

def prepare_mqa(data_type):
    langs = [
        'af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca', 'cbk', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'dsb', 'dty', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'frr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gom', 'gu', 'gv', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lez', 'li', 'lmo', 'lo', 'lrc', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pam', 'pfl', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'sh', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vo', 'wa', 'war', 'wuu', 'xal', 'yi', 'yo', 'yue', 'zh'
    ]
    
    ds_list = []
    
    for lang in langs:
        print(f">>> Starting {lang} - {data_type}")
        try:
            ds = process_mqa(lang, data_type)
        except Exception as e:
            print(e)
            print(f"### Skipping {lang} - {data_type}")
            continue
        ds_list.append(ds)

    cat_ds = concatenate_datasets(ds_list)
        
    return cat_ds

def prepare_mqa_cqa():
    return prepare_mqa("cqa")

def prepare_mqa_faq():
    return prepare_mqa("faq")

if __name__ == "__main__":

    dataset_func_list = [
        prepare_amharicqa,
        prepare_arcd,
        prepare_chaii,
        prepare_cpgqa,
        prepare_drcd,
        prepare_germanquad,
        prepare_hotpotqa,
        prepare_indicqa,
        prepare_jsquad,
        prepare_jqara,
        prepare_kenswquad,
        prepare_korquad,
        prepare_lb_rag,
        prepare_logqa,
        prepare_lsat,
        prepare_m2qa,
        prepare_mldr,
        prepare_mlqa,
        prepare_mqa_cqa,
        prepare_mqa_faq,
        prepare_narrativeqa,
        prepare_persianqa,
        prepare_pira,
        prepare_pubmedqa,
        prepare_qasports,
        prepare_sberquad,
        prepare_scandiqa,
        prepare_skquad,
        prepare_sleepqa,
        prepare_sqac,
        prepare_tquad,
        prepare_triviaqa,
        prepare_tydiqa_goldp,
        prepare_xquad,
    ]

    def write_name_to_file(name):
        with open("temp.txt", "a+") as f:
            f.write(str(name))
            f.write("\n")
        return True
    
    final_features = {
            'question': Value(dtype='string', id=None),
            'answer': Value(dtype='string', id=None),
            'positives': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'negatives': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'dataset_name': Value(dtype='string', id=None),
            'language': Value(dtype='string', id=None),
            'doc_id': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
    }

    required_cols = list(final_features.keys())
    dataset_list = []
    
    for x in tqdm(dataset_func_list):
        print(x)
        write_name_to_file(x)
        ds = x().select_columns(required_cols).map(
            lambda x: {k:v for k, v in x.items()}, 
            features=Features(final_features),
            num_proc=16
        )
        ds.to_parquet("./data/" + str(x).split()[1] + ".parquet")
        dataset_list.append(ds)
    
    ds = concatenate_datasets(dataset_list)

    ds.push_to_hub("lightblue/rag_datasets_collection", private=True)