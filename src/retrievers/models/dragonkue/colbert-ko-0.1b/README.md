---
language:
- ko
license: apache-2.0
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- loss:MatryoshkaColBERTLoss
- korean
- matryoshka
- retrieval
base_model: skt/A.X-Encoder-base
pipeline_tag: sentence-similarity
library_name: PyLate
---

# dragonkue/colbert-ko-0.1b

This is a [PyLate](https://github.com/lightonai/pylate) model finetuned from [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) on the parquet dataset. It maps sentences & paragraphs to sequences of 32, 64, 96, 128-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
- **Base model:** [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) <!-- at revision b5c71f3601aedf38372fe21383ac7d04991af187 -->
- **Document Length:** 2048 tokens
- **Query Length:** 32 tokens
- **Output Dimensionality:** 128 tokens
- **Similarity Function:** MaxSim
- **Training Dataset:**
    - parquet
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [PyLate Documentation](https://lightonai.github.io/pylate/)
- **Repository:** [PyLate on GitHub](https://github.com/lightonai/pylate)
- **Hugging Face:** [PyLate models on Hugging Face](https://huggingface.co/models?library=PyLate)

### Full Model Architecture

```
ColBERTWrapper(
  (0): Transformer({'max_seq_length': 2047, 'do_lower_case': False, 'architecture': 'ModernBertModel'})
  (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)
```

## Usage

This model supports **Matryoshka embeddings** with multiple dimensions (32, 64, 96, 128) using separate projection heads (Jina-ColBERT-v2 style).

### Installation

```bash
pip install colbert-matryoshka
```

### Quick Start (Matryoshka)

```python
from colbert_matryoshka import MatryoshkaColBERT

# Load model
model = MatryoshkaColBERT.from_pretrained("dragonkue/colbert-ko-0.1b")

# Set embedding dimension (32, 64, 96, or 128)
model.set_active_dim(128)

# Encode queries and documents
query_embeddings = model.encode(["검색 쿼리"], is_query=True)
doc_embeddings = model.encode(["문서 내용"], is_query=False)

print(f"Query shape: {query_embeddings[0].shape}")  # (num_tokens, 128)
print(f"Doc shape: {doc_embeddings[0].shape}")      # (num_tokens, 128)
```

### Retrieval (PyLate Index)

Use this model with PyLate to index and retrieve documents. The index uses [FastPLAID](https://github.com/lightonai/fast-plaid) for efficient similarity search.

```python
from colbert_matryoshka import MatryoshkaColBERT
from pylate import indexes, retrieve

# Load model
model = MatryoshkaColBERT.from_pretrained("dragonkue/colbert-ko-0.1b")
model.set_active_dim(128)

# Initialize PLAID index
index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
    override=True,
)

# Encode and index documents
documents_ids = ["1", "2", "3"]
documents = ["첫번째 문서입니다", "두번째 문서입니다", "세번째 문서입니다"]

documents_embeddings = model.encode(documents, is_query=False)
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)

# Retrieve
retriever = retrieve.ColBERT(index=index)
queries_embeddings = model.encode(["첫번째 문서 검색"], is_query=True)

scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=3,
)
print(scores)
# [[{'id': '1', 'score': 24.51}, {'id': '2', 'score': 23.54}, {'id': '3', 'score': 23.33}]]
```

### Reranking

```python
from colbert_matryoshka import MatryoshkaColBERT
from pylate import rank

# Load model
model = MatryoshkaColBERT.from_pretrained("dragonkue/colbert-ko-0.1b")
model.set_active_dim(128)

queries = ["인공지능 기술", "한국어 자연어처리"]

documents = [
    ["AI와 머신러닝에 대한 문서", "요리 레시피 문서"],
    ["한국어 NLP 연구", "영어 문법 설명", "프로그래밍 튜토리얼"],
]

documents_ids = [
    [1, 2],
    [1, 3, 2],
]

# Encode queries
queries_embeddings = model.encode(queries, is_query=True)

# Encode documents (per query)
documents_embeddings = []
for docs in documents:
    documents_embeddings.append(model.encode(docs, is_query=False))

# Rerank
reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
print(reranked_documents)
# Query "인공지능 기술": [{'id': 1, 'score': 3.63}, {'id': 2, 'score': 0.90}]
# Query "한국어 자연어처리": [{'id': 1, 'score': 4.60}, {'id': 3, 'score': 3.88}, {'id': 2, 'score': 1.93}]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Evaluation Results (NDCG@10)

### Comparison with Other Models (dim128)

| Model | AutoRAG | Ko-StrategyQA | NanoBEIR-Ko | Avg |
|-------|---------|---------------|-------------|-----|
| **dragonkue-colbert-ko-0.1b (149M)** | **0.989** | 0.741 | 0.519 | **0.750** |
| BGE-M3-MultiVec (568M) | 0.844 | **0.797** | **0.569** | 0.737 |
| LFM2-ColBERT (353M) | 0.833 | 0.757 | 0.528 | 0.706 |
| colbert-ko-v1 (149M) | 0.966 | 0.713 | 0.476 | 0.718 |

### Performance by Embedding Dimension

| Dimension | AutoRAG | Ko-StrategyQA | NanoBEIR-Ko |
|-----------|---------|---------------|-------------|
| 32 | 0.983 | 0.721 | 0.504 |
| 64 | 0.985 | 0.728 | 0.510 |
| 96 | 0.979 | 0.736 | 0.517 |
| **128** | **0.989** | **0.741** | **0.519** |


* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--

### Full Evaluation Results

<details>
<summary>Detailed metrics (click to expand)</summary>

#### dim128

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.974 | 0.989 | 0.989 | 0.989 | 0.989 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.681 | 0.682 | 0.718 | 0.741 | 0.752 | 0.825 | 0.860 |
| NanoBEIR-Ko | 0.472 | 0.495 | 0.507 | 0.519 | 0.528 | 0.557 | 0.630 |

#### dim96

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.947 | 0.975 | 0.979 | 0.979 | 0.979 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.677 | 0.675 | 0.712 | 0.736 | 0.748 | 0.821 | 0.859 |
| NanoBEIR-Ko | 0.475 | 0.498 | 0.507 | 0.517 | 0.528 | 0.552 | 0.625 |

#### dim64

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.965 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.666 | 0.668 | 0.701 | 0.728 | 0.740 | 0.815 | 0.854 |
| NanoBEIR-Ko | 0.470 | 0.486 | 0.496 | 0.510 | 0.523 | 0.550 | 0.631 |

#### dim32

| Benchmark | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|-----------|--------|--------|--------|---------|---------|-----------|-----------|
| AutoRAG | 0.956 | 0.983 | 0.983 | 0.983 | 0.983 | 1.000 | 1.000 |
| Ko-StrategyQA | 0.657 | 0.667 | 0.699 | 0.721 | 0.732 | 0.807 | 0.841 |
| NanoBEIR-Ko | 0.449 | 0.476 | 0.484 | 0.504 | 0.510 | 0.551 | 0.612 |

</details>

## Training Details

### Training Dataset

#### parquet

* Dataset: parquet
* Size: 379,805 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>neg_0</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                           | positive                                                                              | neg_0                                                                                 |
  |:--------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                                | string                                                                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 23.7 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 419.07 tokens</li><li>max: 1530 tokens</li></ul> | <ul><li>min: 92 tokens</li><li>mean: 334.91 tokens</li><li>max: 1423 tokens</li></ul> |
* Samples:
  | anchor                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | neg_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>트랜스카르파티아의 우크라이나인들은 무엇을 지지하나요?</code> | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>Ⅴ. 결 론<br>□ 우리나라에서는 지방자치단체간의 재정불균형을 완화할 목적으로 중앙정부가 지방교부세를 교부하고 있고 광역자치단체가 기초자치단체에 조정교부금을 교부하는 등 수직적 지방재정조정제도가 운영 중임<br>○ 이에 더하여 지방자치단체간의 수평적 지방재정조정제도인 지역상생발전기금을 운영하고 있음<br>- 지역상생발전기금은 수도권 3개 광역자치단체가 지방소비세의 일부를 출연하여 주로 비수도권 14개 광역자치단체에 배분하는 제도로써, 전국 17개 광역자치단체간의 재정불균형을 시정할 목적으로 도입되었음<br>◆ 하지만 지역상생발전기금과 관련된 제도 및 운영방식에 있어서 불충분하고 불합리한 측면이 있으므로 이에 대한 개선방안이 필요함</code>                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>극단의 일부로 간주되는 그룹은 무엇입니까?</code>       | <code># 1 또한 루신의 일부로 간주됨 # 3 우크라이나인과 벨라루스인 사이의 과도기로 간주됨 # 3 렘코스의 민족적 소속은 이념적 갈등이 되었다.르모스 중에서 "카르파토-루테니아" 민족이라는 개념은 트랜스카르파티아와 해외에 거주하는 르모스에 의해서만 지지된다고 주장되어 왔다.4 역사적인 모라비아의 대부분의 주민들은 자신들을 체코인으로 간주했지만 상당수는 체코와는 다르게(보헤미아와 모라비아 사람들은 같은 공용어를 사용하지만) 모라비아 국적을 선언했다.*5 또한 폴란드인을 고려했습니다.*6 폴란드인들의 일부로 실레지아인들을 보여주는 자료들이 있다.어퍼 실레지아의 최남단 인구의 일부는 때때로 체코인(논란의 여지가 있음)으로 간주된다.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>‘완판돌’ 슈퍼주니어가 코스메틱 브랜드 ‘에이바자르’의 전속 모델로 발탁됐다. 슈퍼주니어는 올해 1월부터 데일리 셀프케어 전문 브랜드인 ‘에이바자르’의 모델로 발탁돼 다양한 컬래버레이션 활동을 펼쳐나갈 예정이라 이목이 집중되고 있다. 더불어 ‘에이바자르’ 측은 “슈퍼주니어는 가수 활동뿐 아니라 각종 예능, 드라마 등에서 활발한 활동을 펼치고 있다. 이처럼 자유롭고 스일리시한 슈퍼주니어의 이미지가 브랜드와 부합한다고 생각해 모델로 발탁했다”며 “각국에 두터운 팬덤을 가지고 있는 슈퍼주니어와 함께 글로벌 시장으로 진출 예정”이라고 전해 전 세계에서 변함 없는 사랑을 받고 있는 슈퍼주니어의 인기를 다시 한 번 확인케 했다. 특히 슈퍼주니어는 지난해 11월 정규 8집 앨범 ‘PLAY’(플레이) 앨범 판매 20만장 돌파 기념 TV홈쇼핑에 출연해 방송 중 4,800여콜이라는 동시 접속 최다 콜 수로 상품을 매진 시키고, 단독 콘서트인 ‘슈퍼주니어 월드투어 - 슈퍼쇼7 (SUPER JUNIOR WORLD TOUR SUPER SHOW7)’도 티켓 예매 9분 만에 전일 매진을 기록해 추가 공연을 결정하는 등 떴다 하면 ‘매진’을 기록하며 ‘완판주니어’라는 수식어를 얻은 바 있기에 더욱 기대가 모아지고 있다. 한편, 최근 슈퍼주니어는 이번 달 개국을 앞둔 오락 전문채널 XtvN에서 새로운 예능 프로그램인 ‘슈퍼TV’를 론칭, 오는 1월 26일 밤 11시에 첫 방송할 예정이다.</code> |
  | <code>중기중앙회는 무엇을 강화하기로 했는가?</code>         | <code># 3대 백화점에 60개 점포를 운영하고 있는 A사는 매출액의 36%를 판매수수료로 내고 있다. 여기에 점포 파견 직원(200여명) 인건비(매출액 14%), 부가세 10%를 포함하면 매출액의 60%가 백화점 점포 운영비용으로 투입된다. 여기에 생산비 33%를 제외하면 매출액의 7%로 제품개발, 사무실운영 등에 충당하면 남는 게 별로 없다. # 3대 백화점에 점포를 가지고 있는 B사는 매년 6억원 이상을 인테리어 비용으로 날리고 있다. 수십개 점포를 운영하고 있는 B사의 경우 연평균 점포 20개 정도 내부시설을 고치고 있다. 1점포당 인테리어 비용이 3000만원을 훌쩍 넘는다. B사는 백화점의 강요로 멀쩡한 시설을 뜯어내고 20개 점포 인테리어에만 년 6억원을 쏟아 붓고 있는 것이다. 이렇듯 롯데 현대 신세계 등 국내 3대 백화점의 불공정행위가 도를 넘어섰다. [IMG1]백화점들이 입점기업에게 받는 판매수수료는 매출액의 1/3 이상이다. 수시로 인테리어 교체와 할인판매를 강요하고 있다. 반면 해외브랜드에는 엄청난 혜택을 주고 있다. 이러한 사실은 중소기업중앙회(회장 김기문)가 한국패션협회와 공동으로 5월 20일부터 27일까지 3대 백화점(롯데, 신세계, 현대) 입점기업 300개를 대상으로 실시한 불공정행위 실태조사에서 확인됐다. 실태조사 결과 백화점 판매수수료율은 2010년 평균 29.33%로 나타났고, 3대 백화점중 롯데백화점 판매수수료율이 30.87%로 가장 높았다. 판매수수료율은 해마다 0.2%p 높아지는 추세를 보였다. 실태조사에서 입점 중소기업 81%는 '판매수수료율이 너무 높다'고 응답했다. 최고 높은 수수료를 내는 업종은 패션·잡화로 38%에 달했다. 또한 최근 3년간 입점기업의 46.9%가 백화점의 불공정행위를 경험한 것으로 나타났다. 불공정행위 는 '인테리어 비용부담' 강요가 54.9%로 가장 높았고, '판촉 및 세일행사 참여' 강요는 48.4%에 달했다. 또한 백화점 입점기업 54.7%는 "백화점이 매년 수수료율을 인상한다"고 응답했다. 이중 ...</code> | <code>[씨티그룹과 협업해 '글로벌자금관리시스템(OCMS)' 도입…에너지공기업 최초]<br>한국남부발전은 2일 에너지공기업 최초로 해외지사에 대한 실시간 자금운영 모니터링을 위한 '글로벌자금관리시스템(OCMS)'을 구축했다고 밝혔다.<br>글로벌 종합금융회사 씨티그룹과 협업해 구축된 OCMS는 남부발전과 해외지사간 자금흐름 등을 한 눈에 확인할 수 있어 자금관리의 효율성과 투명성 제고에 기여할 전망이다.<br>남부발전은 실시간 모니터링을 통해 늘어나는 해외사업장들의 납입 자본금과 자금 입·출입 현황 등을 통합 관리해 자금사고 사각지대를 해소할 계획이다. 또 국가간 시차 등의 이유로 수기보고 하던 업무절차를 개선하는 등 관리단계의 편의성도 높이기로 했다.<br>이를 위해 남부발전은 100% 지분을 소유한 미국, 칠레, 요르단 지사에 우선 OCMS을 도입하고, 이후 출자지분 50% 이상 법인까지 시스템을 확대 적용한다. 동시에 글로벌 전사적자원관리(ERP)를 도입해 투명한 회계처리 프로세스를 정착할 예정이다.<br>신정식 남부발전 사장은 "남부발전은 미국 나일스 프로젝트 같은 양질의 해외사업 발굴 뿐 아니라 단 한 건의 인적 안전사고나 금융사고가 발생하지 않도록 철저한 관리시스템을 구축해 나가겠다"고 밝혔다.</code>                                                                                            |
* Loss: <code>src.losses.MatryoshkaColBERTLoss</code> with these parameters:
  ```json
  {
      "dims": [
          32,
          64,
          96,
          128
      ],
      "weights": [
          0.25,
          0.25,
          0.25,
          0.25
      ],
      "temperature": 1.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `learning_rate`: 1e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `fp16`: True
- `dataloader_drop_last`: True
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: {'use_reentrant': False}
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {'anchor': 'query', 'positive': 'document', 'neg_0': 'document'}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0017 | 10   | 4.1388        |
| 0.0034 | 20   | 4.1142        |
| 0.0051 | 30   | 3.9797        |
| 0.0067 | 40   | 3.8761        |
| 0.0084 | 50   | 3.6167        |
| 0.0101 | 60   | 3.424         |
| 0.0118 | 70   | 3.0256        |
| 0.0135 | 80   | 2.827         |
| 0.0152 | 90   | 2.5787        |
| 0.0169 | 100  | 2.2696        |
| 0.0185 | 110  | 2.0266        |
| 0.0202 | 120  | 1.6815        |
| 0.0219 | 130  | 1.4739        |
| 0.0236 | 140  | 1.2877        |
| 0.0253 | 150  | 1.1474        |
| 0.0270 | 160  | 1.0143        |
| 0.0286 | 170  | 0.9363        |
| 0.0303 | 180  | 0.9189        |
| 0.0320 | 190  | 0.7442        |
| 0.0337 | 200  | 0.6919        |
| 0.0354 | 210  | 0.6251        |
| 0.0371 | 220  | 0.6527        |
| 0.0388 | 230  | 0.5923        |
| 0.0404 | 240  | 0.572         |
| 0.0421 | 250  | 0.5255        |
| 0.0438 | 260  | 0.4407        |
| 0.0455 | 270  | 0.5038        |
| 0.0472 | 280  | 0.3939        |
| 0.0489 | 290  | 0.3938        |
| 0.0506 | 300  | 0.3253        |
| 0.0522 | 310  | 0.335         |
| 0.0539 | 320  | 0.2855        |
| 0.0556 | 330  | 0.2396        |
| 0.0573 | 340  | 0.252         |
| 0.0590 | 350  | 0.2299        |
| 0.0607 | 360  | 0.2133        |
| 0.0624 | 370  | 0.2186        |
| 0.0640 | 380  | 0.1935        |
| 0.0657 | 390  | 0.1743        |
| 0.0674 | 400  | 0.1462        |
| 0.0691 | 410  | 0.1552        |
| 0.0708 | 420  | 0.1491        |
| 0.0725 | 430  | 0.1581        |
| 0.0741 | 440  | 0.1635        |
| 0.0758 | 450  | 0.1383        |
| 0.0775 | 460  | 0.1377        |
| 0.0792 | 470  | 0.1155        |
| 0.0809 | 480  | 0.1184        |
| 0.0826 | 490  | 0.1333        |
| 0.0843 | 500  | 0.1341        |
| 0.0859 | 510  | 0.1259        |
| 0.0876 | 520  | 0.0748        |
| 0.0893 | 530  | 0.1342        |
| 0.0910 | 540  | 0.1058        |
| 0.0927 | 550  | 0.1024        |
| 0.0944 | 560  | 0.0921        |
| 0.0961 | 570  | 0.104         |
| 0.0977 | 580  | 0.1069        |
| 0.0994 | 590  | 0.0925        |
| 0.1011 | 600  | 0.1146        |
| 0.1028 | 610  | 0.0682        |
| 0.1045 | 620  | 0.0711        |
| 0.1062 | 630  | 0.1491        |
| 0.1079 | 640  | 0.0602        |
| 0.1095 | 650  | 0.0753        |
| 0.1112 | 660  | 0.0713        |
| 0.1129 | 670  | 0.0739        |
| 0.1146 | 680  | 0.0783        |
| 0.1163 | 690  | 0.0678        |
| 0.1180 | 700  | 0.0963        |
| 0.1196 | 710  | 0.0677        |
| 0.1213 | 720  | 0.0829        |
| 0.1230 | 730  | 0.0719        |
| 0.1247 | 740  | 0.0646        |
| 0.1264 | 750  | 0.0927        |
| 0.1281 | 760  | 0.0755        |
| 0.1298 | 770  | 0.0799        |
| 0.1314 | 780  | 0.0535        |
| 0.1331 | 790  | 0.0555        |
| 0.1348 | 800  | 0.0804        |
| 0.1365 | 810  | 0.0627        |
| 0.1382 | 820  | 0.0726        |
| 0.1399 | 830  | 0.0685        |
| 0.1416 | 840  | 0.0421        |
| 0.1432 | 850  | 0.0895        |
| 0.1449 | 860  | 0.0964        |
| 0.1466 | 870  | 0.0515        |
| 0.1483 | 880  | 0.0825        |
| 0.1500 | 890  | 0.0801        |
| 0.1517 | 900  | 0.0579        |
| 0.1534 | 910  | 0.0559        |
| 0.1550 | 920  | 0.0432        |
| 0.1567 | 930  | 0.0553        |
| 0.1584 | 940  | 0.0577        |
| 0.1601 | 950  | 0.0451        |
| 0.1618 | 960  | 0.049         |
| 0.1635 | 970  | 0.0459        |
| 0.1651 | 980  | 0.0684        |
| 0.1668 | 990  | 0.0449        |
| 0.1685 | 1000 | 0.0392        |
| 0.1702 | 1010 | 0.071         |
| 0.1719 | 1020 | 0.0511        |
| 0.1736 | 1030 | 0.0501        |
| 0.1753 | 1040 | 0.0464        |
| 0.1769 | 1050 | 0.0678        |
| 0.1786 | 1060 | 0.0597        |
| 0.1803 | 1070 | 0.0569        |
| 0.1820 | 1080 | 0.044         |
| 0.1837 | 1090 | 0.0452        |
| 0.1854 | 1100 | 0.0394        |
| 0.1871 | 1110 | 0.0496        |
| 0.1887 | 1120 | 0.0296        |
| 0.1904 | 1130 | 0.0321        |
| 0.1921 | 1140 | 0.0525        |
| 0.1938 | 1150 | 0.058         |
| 0.1955 | 1160 | 0.0552        |
| 0.1972 | 1170 | 0.035         |
| 0.1989 | 1180 | 0.0468        |
| 0.1999 | 1186 | -             |
| 0.2005 | 1190 | 0.0383        |
| 0.2022 | 1200 | 0.0599        |
| 0.2039 | 1210 | 0.0572        |
| 0.2056 | 1220 | 0.0383        |
| 0.2073 | 1230 | 0.0486        |
| 0.2090 | 1240 | 0.0407        |
| 0.2107 | 1250 | 0.044         |
| 0.2123 | 1260 | 0.04          |
| 0.2140 | 1270 | 0.0338        |
| 0.2157 | 1280 | 0.036         |
| 0.2174 | 1290 | 0.0511        |
| 0.2191 | 1300 | 0.0472        |
| 0.2208 | 1310 | 0.031         |
| 0.2224 | 1320 | 0.0614        |
| 0.2241 | 1330 | 0.0388        |
| 0.2258 | 1340 | 0.0403        |
| 0.2275 | 1350 | 0.047         |
| 0.2292 | 1360 | 0.033         |
| 0.2309 | 1370 | 0.0524        |
| 0.2326 | 1380 | 0.0357        |
| 0.2342 | 1390 | 0.0463        |
| 0.2359 | 1400 | 0.0355        |
| 0.2376 | 1410 | 0.0411        |
| 0.2393 | 1420 | 0.028         |
| 0.2410 | 1430 | 0.0386        |
| 0.2427 | 1440 | 0.0553        |
| 0.2444 | 1450 | 0.0353        |
| 0.2460 | 1460 | 0.0462        |
| 0.2477 | 1470 | 0.0399        |
| 0.2494 | 1480 | 0.0319        |
| 0.2511 | 1490 | 0.0456        |
| 0.2528 | 1500 | 0.0302        |
| 0.2545 | 1510 | 0.0366        |
| 0.2562 | 1520 | 0.0409        |
| 0.2578 | 1530 | 0.0337        |
| 0.2595 | 1540 | 0.0362        |
| 0.2612 | 1550 | 0.0318        |
| 0.2629 | 1560 | 0.0433        |
| 0.2646 | 1570 | 0.0379        |
| 0.2663 | 1580 | 0.0419        |
| 0.2679 | 1590 | 0.0225        |
| 0.2696 | 1600 | 0.0269        |
| 0.2713 | 1610 | 0.0295        |
| 0.2730 | 1620 | 0.048         |
| 0.2747 | 1630 | 0.0382        |
| 0.2764 | 1640 | 0.0341        |
| 0.2781 | 1650 | 0.0334        |
| 0.2797 | 1660 | 0.0534        |
| 0.2814 | 1670 | 0.0445        |
| 0.2831 | 1680 | 0.0284        |
| 0.2848 | 1690 | 0.0327        |
| 0.2865 | 1700 | 0.0309        |
| 0.2882 | 1710 | 0.0372        |
| 0.2899 | 1720 | 0.0384        |
| 0.2915 | 1730 | 0.022         |
| 0.2932 | 1740 | 0.0266        |
| 0.2949 | 1750 | 0.0399        |
| 0.2966 | 1760 | 0.0342        |
| 0.2983 | 1770 | 0.0391        |
| 0.3000 | 1780 | 0.0349        |
| 0.3017 | 1790 | 0.0365        |
| 0.3033 | 1800 | 0.0322        |
| 0.3050 | 1810 | 0.0414        |
| 0.3067 | 1820 | 0.0297        |
| 0.3084 | 1830 | 0.0446        |
| 0.3101 | 1840 | 0.0312        |
| 0.3118 | 1850 | 0.0379        |
| 0.3134 | 1860 | 0.0252        |
| 0.3151 | 1870 | 0.0424        |
| 0.3168 | 1880 | 0.0367        |
| 0.3185 | 1890 | 0.0226        |
| 0.3202 | 1900 | 0.0319        |
| 0.3219 | 1910 | 0.0189        |
| 0.3236 | 1920 | 0.0219        |
| 0.3252 | 1930 | 0.0341        |
| 0.3269 | 1940 | 0.0505        |
| 0.3286 | 1950 | 0.0176        |
| 0.3303 | 1960 | 0.0328        |
| 0.3320 | 1970 | 0.0276        |
| 0.3337 | 1980 | 0.0251        |
| 0.3354 | 1990 | 0.0603        |
| 0.3370 | 2000 | 0.0243        |
| 0.3387 | 2010 | 0.0316        |
| 0.3404 | 2020 | 0.0294        |
| 0.3421 | 2030 | 0.025         |
| 0.3438 | 2040 | 0.0255        |
| 0.3455 | 2050 | 0.0318        |
| 0.3472 | 2060 | 0.025         |
| 0.3488 | 2070 | 0.0273        |
| 0.3505 | 2080 | 0.0338        |
| 0.3522 | 2090 | 0.0299        |
| 0.3539 | 2100 | 0.0275        |
| 0.3556 | 2110 | 0.0184        |
| 0.3573 | 2120 | 0.0244        |
| 0.3589 | 2130 | 0.0432        |
| 0.3606 | 2140 | 0.0325        |
| 0.3623 | 2150 | 0.0525        |
| 0.3640 | 2160 | 0.0329        |
| 0.3657 | 2170 | 0.0236        |
| 0.3674 | 2180 | 0.0309        |
| 0.3691 | 2190 | 0.0195        |
| 0.3707 | 2200 | 0.0318        |
| 0.3724 | 2210 | 0.0229        |
| 0.3741 | 2220 | 0.0312        |
| 0.3758 | 2230 | 0.0186        |
| 0.3775 | 2240 | 0.0231        |
| 0.3792 | 2250 | 0.0262        |
| 0.3809 | 2260 | 0.0287        |
| 0.3825 | 2270 | 0.0299        |
| 0.3842 | 2280 | 0.0302        |
| 0.3859 | 2290 | 0.0281        |
| 0.3876 | 2300 | 0.0252        |
| 0.3893 | 2310 | 0.0362        |
| 0.3910 | 2320 | 0.0266        |
| 0.3927 | 2330 | 0.0304        |
| 0.3943 | 2340 | 0.0259        |
| 0.3960 | 2350 | 0.0276        |
| 0.3977 | 2360 | 0.0219        |
| 0.3994 | 2370 | 0.0361        |
| 0.3997 | 2372 | -             |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.1.1
- PyLate: 1.3.4
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.2-rc0


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@misc{PyLate,
title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
author={Chaffin, Antoine and Sourty, Raphaël},
url={https://github.com/lightonai/pylate},
year={2024}
}
```

#### Jina ColBERT v2
```bibtex
@article{jina-colbert-v2,
    title={Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever},
    author={Rohan Jha and Bo Wang and Michael Günther and Saba Sturua and Mohammad Kalim Akram and Han Xiao},
    year={2024},
    journal={arXiv preprint arXiv:2408.16672},
    url={https://arxiv.org/abs/2408.16672}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->