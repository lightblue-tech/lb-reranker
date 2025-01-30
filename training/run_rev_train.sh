echo '{
  "reranker_continuous_filt_max7_rev_train": {
    "hf_hub_url": "lightblue/reranker_continuous_filt_max7_train_extra",
    "formatting": "sharegpt",
        "columns": {
          "messages": "rev_conversations"
        }
  }
}' > /root/LLaMA-Factory/data/dataset_info.json

cd /root/LLaMA-Factory && llamafactory-cli train /root/lb-reranker/training/rev_train_config.yaml

rm -r /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_rev_train/checkpoint*
huggingface-cli upload lightblue/reranker_0.5_cont_filt_7max_rev /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_rev_train
