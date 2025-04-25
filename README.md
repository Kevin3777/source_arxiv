# run code
The environment has already been configured and adapted to the code in autodl, which can be quite complicated. Before modifying the code, you can backup the Northwest B zone/710 machine and Northwest B zone/274 machine.

The following configuration file occupies about 28GB of video memory. To speed up training, it is necessary to switch to a better machine.

```python
source /etc/network_turbo
python run_experiment.py conf/ours_doc_id_begin.yaml
```
just replace conf/ours_doc_id_begin.yaml（such as conf/ours_doc_id_end.yaml）

The configuration file is divided into two parts:

conf/ours_doc_id_begin.yaml and conf\templates\train_config.yaml

The default training parameter configuration is conf\templates\train_config.yaml，Parameter configurations not mentioned in conf/ours_doc_id_begin.yaml, use the parameters from train_config.yaml

```yaml
# Pretrain a gpt2 style model
text_data_path: /root/autodl-tmp/intrinsic-source-citation/dataset/ours/pretrain
streaming: outputs/experiments/arxiv-citation-doc-id-end/data/streaming/
tokenizer_name: ${streaming}/tokenizer
max_seq_len: 1024  
global_seed: 17
url_trie: ${streaming}/url_trie.pkl
ood_url_trie: ${streaming}/unseen_url_trie.pkl

# Run Name
run_name: arxiv-citation-doc-id-end
cross_doc_attention: false

# Model
model:
  name: hf_causal_lm
  pretrained_model_name_or_path: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
  pretrained: true  # Decide whether to conduct pre training. Pre training takes a short time of two to three minutes and can be left unchanged
  loss:
    type: mask  # Corresponding to the loss_type in the configuration file
    url_loss_factor: 1.0
  

  
  ckpt_dir: outputs/experiments/arxiv-citation-doc-id-end/checkpoints
  # checkpoint: "outputs/experiments/arxiv-citation-doc-id-end/checkpoints/latest"  # Specific checkpoint paths (where breakpoint training can be performed on breakpoints)
    
# Tokenizer 
# Tokenizer
tokenizer:
  name: ${tokenizer_name}
  kwargs:
    model_max_length: ${max_seq_len}


# Dataloaders
dataloaders:
  - name: train_loader_docs
    dataset:
      local: ${streaming}
      split: train
      shuffle: true
      max_seq_len: ${max_seq_len}
      batch_type: lm
      masking:
        cross_doc_attention: ${cross_doc_attention}
    drop_last: false
    num_workers: 0
  
  # Evaluate the data loader section
  - name: in_domain_standard_q_answer_eval_loader
    dataset:
      path: /root/autodl-tmp/intrinsic-source-citation/dataset/ours/qa
      split: qa_train
      shuffle: false
      max_seq_len: ${max_seq_len}
      batch_type: qa
    drop_last: false
    num_workers: 0

  - name: out_of_domain_standard_q_answer_eval_loader
    dataset:
      path: /root/autodl-tmp/intrinsic-source-citation/dataset/ours/qa
      split: qa_train
      shuffle: false
      max_seq_len: ${max_seq_len}
      batch_type: qa-ood 
    drop_last: false
    num_workers: 0

# Optimization
scheduler:
  name: linear_decay_with_warmup
  t_warmup: 1ep
  alpha_f: 0.1

optimizer:
  name: deepspeed_adam
  lr: 1.0e-4
  betas:
  - 0.9
  - 0.98
  eps: 1.0e-06
  weight_decay: 0.0

algorithms:
  gradient_clipping:
    clipping_type: norm
    clipping_threshold: 1.0

max_duration: 10ep # 
eval_interval: 3ep
eval_first: false
eval_subset_num_batches: -1
global_train_batch_size: 64 # 128 initially

# System 
seed: ${global_seed}
device_eval_batch_size: 64      # Evaluate Batch
device_train_microbatch_size: 4  # Send in 4 samples for each training session

# device_train_microbatch_size: auto
precision: amp_bf16

deepspeed_config:
  bf16:
    enabled: true
  train_batch_size: ${global_train_batch_size}
  zero_optimization:
    stage: 3  
    contiguous_gradients: true
    reduce_bucket_size: true
    overlap_comm: true
    allgather_bucket_size: 2e8
    reduce_scatter: true
    offload_optimizer:
      device: cpu
      pin_memory: true
    
    # stage: 2  
    # contiguous_gradients: true
    # reduce_bucket_size: 2e8
    # overlap_comm: true
    # allgather_partitions: true
    # allgather_bucket_size: 2e8
    # reduce_scatter: true
    # cpu_offload: true


# Logging
progress_bar: false
log_to_console: true
console_log_interval: 50ba

callbacks:
  speed_monitor:
    window_size: 10
  lr_monitor: {}
  memory_monitor: {}
  runtime_estimator: {}

loggers:
   wandb: 
    project: intrinsic-source-citation

# Checkpoint to local filesystem or remote object store
save_interval: 1ep
save_num_checkpoints_to_keep: 1 
save_folder: "outputs/experiments/arxiv-citation-doc-id-end/checkpoints"  # You can keep the checkpoints for model training here
```


If the parameter is defined in conf/ours_doc_id_begin. yaml, use the configuration of conf/ours_doc_id_begin. yaml

```yaml
# Experiment Configuration

experiment:
  name: arxiv-citation-doc-id-begin
  output_dir: outputs/experiments/

data:
  text_data_path: dataset/ours/pretrain  # Point to the directory of pre training datasets
  train_data_path: /root/autodl-tmp/intrinsic-source-citation/dataset/ours/pretrain/train
  qa_data_path: /root/autodl-tmp/intrinsic-source-citation/dataset/ours
  augment:
    doc:
      do: false
      method: permute
      n_sample_per_doc: 2
  finetune:
    number_non_attributable_negatives: 0
    neg_create_probability: 0.0

model:
  name: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

train:
  url_location: first  # The URL location is at the beginning of the document
  pretrain: true  # Decide whether to conduct pre training. Pre training takes a short time of two to three minutes and can be left unchanged
  sequential: false
  finetune_q_url_a: false
  repeat_url_across_doc: false
  finetune_q_a_url: true
  finetune_q_a: false
  finetune_q_a_doc_url: false
  q_a_url_predict_url_only: false
  
  # Loss and attention allocation
  cross_doc_attention: false
  url_loss_factor: 1.0
  loss_type: mask
  config_template_path: conf/templates/train_config.yaml
  
  # Training parameters
  device_eval_batch_size: 40
  device_train_microbatch_size: 2
  eval_first: false
  weight_decay: 0.02
  lr: 8.0e-5
  max_duration: 10ep
  save_folder: "outputs/experiments/arxiv-citation-doc-id-begin/checkpoints" # You can keep the checkpoints for model training here

eval:  
  disable_qa_eval: false
  disable_all_eval: false
  disable_attribution_eval: false
  disable_non_attrib_eval: true
  icl_eval: false
  ppl_eval: false
  use_ais: false
```

# quick train and evaluation
Training evaluation results for **conf/ours_doc_id_begin.yaml**：

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T |
| Number of Parameters | 1,100,056,576 (about 1.1B) |
| Optimizer | DeepSpeedCPUAdam (ZeRO Stage 3) |
| Learning Rate | 8.0e-05 (initial) |
| Learning Rate Scheduler | Linear decay with warmup |
| Warmup Steps | 1 epoch |
| Batch Processing Size | Full batch: 80, devices: 80, micro-batch: 2 |
| Gradient Accumulation Steps | 40 |
| Precision | amp_bf16 (mixed precision, BF16) |
| Maximum Sequence Length | 2048 |
| Total Training Cycles | 10 |
| Weight Decay | 0.02 |
| Gradient Clipping | 1.0 |
| Seeds | 17 |


| Cycle | Batch/Total Batch | Training Loss | QA-EM (In-domain) | QA-F1 (In-domain) | QA-EM (Out-of-domain) | QA-F1 (Out-of-domain) |
|-------|------------------|--------------|------------------|-------------------|----------------------|----------------------|
| 1 | 50/134 | 1.7855 | - | - | - | - |
| 1 | 100/134 | 1.6113 | - | - | - | - |
| 1 | 134/134 | - | 0.0000 | 0.1529 | 0.0000 | 0.1529 |
| 2 | 16/134 | 1.1817 | - | - | - | - |
| 2 | 66/134 | 1.0323 | - | - | - | - |
| 2 | 116/134 | 0.8217 | - | - | - | - |
| 2 | 134/134 | - | 0.0000 | 0.1638 | 0.0000 | 0.1638 |
| 3 | 32/134 | 0.3576 | - | - | - | - |
| 3 | 82/134 | 0.3670 | - | - | - | - |
| 3 | 132/134 | 0.2652 | - | - | - | - |
| 3 | 134/134 | - | 0.0000 | 0.1845 | 0.0000 | 0.1845 |
| 4 | 48/134 | 0.1173 | - | - | - | - |
| 4 | 98/134 | 0.1224 | - | - | - | - |
| 4 | 134/134 | - | 0.0000 | 0.1982 | 0.0000 | 0.1982 |
| 5 | 14/134 | 0.0555 | - | - | - | - |
| 5 | 64/134 | 0.0648 | - | - | - | - |
| 5 | 114/134 | 0.0504 | - | - | - | - |
| 5 | 134/134 | - | 0.0000 | 0.2031 | 0.0000 | 0.2031 |

Key Observations:
- Training loss decreases from 1.7855 in the first cycle to 0.0504 in the fifth cycle, indicating the model is effectively learning.
- QA-F1 score gradually improves from 0.1529 to 0.2031, showing performance enhancement.
- QA-EM (exact match) remains consistently at 0.0000, suggesting the model struggles with exact matching.
- The performance metrics are similar for both in-domain and out-of-domain evaluations.

This model was optimized using DeepSpeed ZeRO Stage 3, demonstrating efficient memory management and learning rate scheduling during training.


Training evaluation results for **conf/ours_doc_id_end.yaml**：
| Parameter | Value |
|-----------|-------|
| Model | TinyLlama-1.1B-intermediate-step-1431k-3T |
| Task | Reference/Retrospective Training (Based on Document ID) |
| Hardware | CPU Training (DeepSpeed ZeRO Stage 3 Optimization) |
| Full Batch Training Size | 80 |
| Device Training Micro-batch Size | 2 |
| Gradient Accumulation Steps | 40 |
| Precision | Mixed Precision (bfloat16) |
| Initial Learning Rate | 8e-05 |
| Learning Rate Scheduler | Linear Decay (1 epoch warmup) |
| Optimizer | DeepSpeed Adam (Weight Decay: 0.02) |
| Planned Training Length | 10 epochs |
| Sequence Length | 2048 tokens |
| Loss Type | Loss Based on Sequence Coding |


| Training Results |  |  |
|--------------|--------------|-----------------|
| **Cycle** | **Initial Loss** | **Final Loss** | **Learning Rate Change** |
| 1st Cycle | ~2.22 | ~1.65 | Increased to ~8e-05 |
| 2nd Cycle | ~1.27 | ~0.98 | Started decreasing from ~8e-05 |
| 3rd Cycle | ~0.42 | ~0.37 | Continued Decreasing |
| 4th Cycle | ~0.11 | ~0.14 | Continued Decreasing |
| 5th Cycle | ~0.05 | ~0.06 | Decreased to ~5.2e-05 |

| Evaluation Results |  |  |
|-------------------|----------------------|----------------------|
| **Evaluation Timing** | **QA Exact Match (EM)** | **QA F1 Score (In-domain)** | **QA F1 Score (Out-of-domain)** |
| After 3rd Cycle | 0.0 | 0.1810 | 0.1810 |
| After 6th Cycle | 0.0 | 0.2033 | 0.2033 |

**Key Observations:**

1. Loss decreased significantly over the first few cycles, stabilizing at a very low level (around 0.05) by the 5th cycle, indicating good model convergence.

2. Exact Match (EM) metric remained at 0.0, suggesting difficulty in generating exact matches.

3. F1 scores showed modest improvement, increasing from 0.181 to 0.203.

4. DeepSpeed CPU training was successful, though with high memory consumption (around 160GB CPU memory).

5. Model performance was consistent across in-domain and out-of-domain data, demonstrating generalization capability.


# model checkpoints
model checkpoints(old)：https://huggingface.co/Kevin3777/arxiv-citation-doc-id-begin/tree/main

model checkpoints(new)：https://huggingface.co/Kevin3777/arxiv-citation-doc-id-begin_new/tree/main/checkpoints

Preprocessed dataset: link as above

The structure of the dataset (just focus on ours)：
![alt text](pictures/dataset.png)





# Run Evaluation
```python
python rouge_eval.py conf/eval2.yaml
```
Results:
```yaml
ROUGE evaluation results:
rouge1_precision: 0.9920
rouge1_recall: 0.5321
rouge1_fmeasure: 0.6800
rouge2_precision: 0.9915
rouge2_recall: 0.5294
rouge2_fmeasure: 0.6774
rougeL_precision: 0.9920
rougeL_recall: 0.5321
rougeL_fmeasure: 0.6800
Detailed results saved to: /root/autodl-tmp/intrinsic-source-citation/outputs/experiments/arxiv-citation-doc-id-begin/evaluation_results/detailed_rouge_results.csv
Summary results saved to: /root/autodl-tmp/intrinsic-source-citation/outputs/experiments/arxiv-citation-doc-id-begin/evaluation_results/rouge_summary.csv
```



#### One-script-for-all
To eliminate the need to run many consecutive scripts, I designed the code such that a single script will do everything. Specifically, `run_experiment.py` will take as input a configuration file (more on that later) and will: 
1. Perform data augmentation if necessary (by shuffling facts within the document as described in the paper)
2. Preprocess the pretraining data by injecting Doc IDs (referred to as **URL** throughought the code) into the pretraining data as per the passed config
3. Preprocess and tokenize the instruction tuning comprised of <Question, Answer, Doc ID> triplets
4. Builds the Doc ID Trie needed for constrained decoding when predicting doc IDs. 
5. Save all tokenized data to specified experiment folder in numpy `.npz` format.
6. Run pretraining using next-word objective on the documents with injected doc IDs
7. After pretraining finishes, loads the last checkpoint and does instruction tuning.
8. Logs all evals to W&B



