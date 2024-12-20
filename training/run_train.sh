echo '{
  "reranker_continuous_filt_max7_train": {
    "hf_hub_url": "lightblue/reranker_continuous_filt_max7_train",
    "formatting": "sharegpt"
  }
}' > /root/LLaMA-Factory/data/dataset_info.json

cd /root/LLaMA-Factory && llamafactory-cli train /root/lb-reranker/training/train_config.yaml

rm -r /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train/checkpoint*
huggingface-cli upload lightblue/reranker_0.5_cont_filt_7max /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train
