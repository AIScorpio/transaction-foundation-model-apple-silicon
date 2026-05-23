# Transaction Foundation Model for Apple Silicon

Apple Silicon adaptation of NVIDIA's Transaction Foundation Model developer example.

This fork keeps the original transaction foundation model workflow but replaces the NVIDIA-only runtime path with a local macOS path that can run on Apple Silicon machines using pandas, NumPy, scikit-learn, PyTorch MPS, Hugging Face Transformers, and CPU XGBoost.

Upstream project: <https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model>

This repository is not an NVIDIA release. It is an independent adaptation for Mac / Apple Silicon development and experimentation.

## What Changed

The original blueprint targets NVIDIA GPUs with RAPIDS, cuDF, cuPy, cuML, NeMo AutoModel, CUDA, and containerized training. This fork adapts the same end-to-end notebook sequence for a local Apple Silicon workstation.

| Area | NVIDIA upstream path | Apple Silicon fork path |
|---|---|---|
| Dataframe processing | RAPIDS cuDF | pandas / pyarrow |
| Array operations | CuPy | NumPy |
| Numeric binning | cuML | scikit-learn |
| Tokenizer backend | CUDA-first tokenizer components | `TOKENIZER_BACKEND=cpu` by default |
| Model training demo | NeMo AutoModel | Hugging Face `Trainer` + `LlamaForCausalLM` |
| Device selection | CUDA | MPS, CUDA, then CPU fallback |
| Downstream model | GPU-oriented workflow | XGBoost CPU workflow |

The fork does not claim to reproduce NVIDIA's full-scale multi-GPU pretraining. The local training path is a Mac-safe demonstration of the architecture and notebook workflow.

## Repository Flow

Run the notebooks in order:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `01_dataset_baseline.ipynb` | Prepare TabFormer data, temporal splits, and baseline fraud modeling. |
| 2 | `02_seq_preproc_tokenization.ipynb` | Build the financial tokenizer and generate transaction sequence corpora. |
| 3 | `03_foundation_model_training.ipynb` | Run a local Hugging Face decoder training demo on MPS or CPU. |
| 4 | `04_inference_embedding_extraction.ipynb` | Load a decoder checkpoint and extract transaction embeddings. |
| 5 | `05_xgboost_fraud_detection.ipynb` | Compare raw features, embedding features, and combined downstream fraud models. |

The key implementation files are:

- `src/tokenizer/backend.py` selects CPU or GPU dataframe/math backends.
- `src/tokenizer/` contains the tokenizer components adapted away from direct cuDF/cuPy imports.
- `src/decoder_inference.py` chooses MPS, CUDA, or CPU at runtime.
- `scripts/train_decoder_hf.py` ports the decoder training demo to Hugging Face Trainer.
- `requirements-mac.txt` contains the macOS dependency set.
- `refactor/2026-05-13-apple-silicon-migration-design.md` records the migration design.

## Setup

Tested target environment:

- macOS on Apple Silicon
- Python 3.12
- PyTorch with MPS support
- CPU XGBoost

Create an environment:

```bash
conda create -n tfm python=3.12 -y
conda activate tfm
pip install -r requirements-mac.txt
```

Install the macOS OpenMP runtime used by XGBoost:

```bash
brew install libomp
```

If you need the pretrained checkpoint tracked through Git LFS, install LFS and pull the model artifacts:

```bash
git lfs install
git lfs pull
```

## Tokenizer Backend

The fork defaults to the CPU backend:

```bash
export TOKENIZER_BACKEND=cpu
```

That path uses pandas, NumPy, scikit-learn, and mmh3. It is the intended path for Apple Silicon.

The code still includes a `TOKENIZER_BACKEND=gpu` switch for environments that provide cuDF, CuPy, and cuML, but this fork's maintained target is the Apple Silicon path.

## Local Training Demo

Notebook 03 uses the same compact decoder shape as the original example:

| Parameter | Value |
|---|---|
| Architecture | Llama decoder-only transformer |
| Hidden size | 512 |
| Layers | 8 |
| Attention | GQA, 8 query heads, 2 KV heads |
| Context window | 8,192 RoPE positions |
| Activation | SwiGLU |
| Normalization | RMSNorm |

You can also run the training script directly:

```bash
python scripts/train_decoder_hf.py \
  --train_data data/decoder_corpus/train_corpus.txt \
  --val_data data/decoder_corpus/val_corpus.txt \
  --output_dir models/decoder-demo \
  --max_steps 30 \
  --per_device_train_batch_size 1
```

Use smaller batch sizes on memory-constrained Macs.

## Outputs

Generated data, embeddings, notebook outputs, demo checkpoints, and sharing-deck artifacts are intentionally ignored by Git:

- `data/TabFormer/`
- `data/decoder_corpus/`
- `data/embeddings/`
- `data/outputs/`
- `models/decoder-demo/`

This keeps the repository source-focused. Recreate local artifacts by running the notebooks.

## Known Constraints

- This fork is optimized for local experimentation, not production-scale pretraining.
- The original NVIDIA multi-GPU NeMo path remains the upstream reference.
- Checkpoint files under `models/decoder-foundation-model/` are Git LFS artifacts inherited from the upstream project.
- Running notebooks 04 and 05 requires the expected checkpoint artifacts to be available locally.

## License

This project is distributed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

The original work is from NVIDIA's `NVIDIA-AI-Blueprints/transaction-foundation-model` repository. This fork preserves the upstream license and adds Apple Silicon adaptation work. See [NOTICE](NOTICE) for attribution and modification notes.

Third-party dependencies and datasets are governed by their own licenses and terms.
