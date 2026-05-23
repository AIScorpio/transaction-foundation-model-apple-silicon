# Design: Apple Silicon Migration

**Date:** 2026-05-13
**Status:** Phase 0
**Requirement:** All 5 notebooks runnable on macOS with Apple Silicon (MPS/CPU)

## 1. Problem Statement

This repo depends on NVIDIA CUDA stack (cuDF, cuPy, cuML, NeMo AutoModel, FSDP2, NCCL).
Goal: migrate to run on Apple Silicon Mac using pandas/numpy/sklearn + PyTorch MPS backend.

## 2. Constraints

- All changes limited to repo directory and subfolders only
- Must not touch files outside this directory
- Must maintain functional equivalence (same model architecture, same data pipeline logic)
- No external file downloads beyond what the original repo requires

## 3. Migration Strategy

### Backend Abstraction Pattern
Create `src/tokenizer/backend.py` with environment-variable toggled imports:
- `TOKENIZER_BACKEND=gpu` → cudf/cupy/cuml (original)
- `TOKENIZER_BACKEND=cpu` (default) → pandas/numpy/sklearn

### Component Migration Map

| Component | Original | Replacement |
|-----------|----------|-------------|
| `cudf` | DataFrame ops | `pandas` |
| `cupy` | Array math | `numpy` |
| `cuml.KBinsDiscretizer` | Binning | `sklearn.preprocessing.KBinsDiscretizer` |
| `cudf.Series.hash_values()` | Merchant hashing | `mmh3.hash64` via `.apply()` |
| `cp.cuda.Stream` | Parallel GPU streams | Sequential execution |
| `torch.cuda` | Device ops | `torch.backends.mps` / `torch.mps` |
| `nemo_automodel` | Training framework | HuggingFace `Trainer` + `LlamaForCausalLM` |
| `FSDP2Manager` | Distributed training | Single-device (29M params fits easily) |
| `nccl` | Collective comms | Remove (single device) |
| `MaskedCrossEntropy` | Loss function | `CrossEntropyLoss(ignore_index=-100)` |
| `cuml.manifold.UMAP` | Visualization | `umap.UMAP` (umap-learn) |

## 4. Implementation Plan

### Phase 1.1: backend.py — Backend Abstraction Layer
- New file: `src/tokenizer/backend.py`
- Exports: `DataFrame`, `Series`, `array_lib` (np/cp), `KBinsDiscretizer`, `hash_series()`, `to_datetime()`
- Environment variable `TOKENIZER_BACKEND` selects GPU vs CPU

### Phase 1.2: Tokenizer Base Components
- `base.py`: `cudf.Series` → generic type hint
- `fixed_vocab.py`: `cudf`/`cp.cuda.Stream` → backend imports, remove stream param
- `mapping.py`: `cudf`/`cupy` → backend imports, remove `.to_pandas()`/`.get()`

### Phase 1.3: Tokenizer Advanced Components
- `categorical_hash.py`: `cudf`/`cp.cuda.Stream` → backend imports
- `timedelta.py`: `cupy` → `numpy` via backend, remove `.get()`
- `numerical.py`: `cuml` → `sklearn` via backend

### Phase 1.4: Pipeline & Financial Pipeline
- `pipeline.py`: Remove CUDA streams, use sequential path always on CPU
- `financial_pipeline.py`: `cudf` → `pandas`, `hash_values()` → `hash_series()`

### Phase 1.5: Training Script Rewrite
- `scripts/train_decoder_model.py`: NeMo → HuggingFace Trainer
- `src/clm_data.py`: Already portable (no changes needed)
- `src/decoder_inference.py`: `torch.cuda` → `torch.mps` guards

### Phase 1.6: Notebooks
- NB01: cudf→pandas, torch.cuda→mps, remove .to_pandas()
- NB02: cudf→pandas, remove .to_pandas()
- NB03: Full rewrite with HF Trainer
- NB04: cudf→pandas, cuml→umap-learn, cupy→numpy
- NB05: cudf→pandas, XGB device→cpu

## 5. Acceptance Criteria

- [ ] All tokenizer files import from `backend.py` instead of direct cudf/cupy/cuml
- [ ] `python -c "from src.tokenizer import FinancialTabularTokenizer"` succeeds without cudf
- [ ] `python -c "from src.tokenizer import FinancialTokenizerPipeline"` succeeds without cudf
- [ ] `python -c "from src.decoder_inference import HuggingFaceDecoderInference"` succeeds without CUDA
- [ ] `scripts/train_decoder_model.py` runs without nemo_automodel (dry-run / help)
- [ ] No `import cudf`, `import cupy`, `from cuml` in any src/ file (except backend.py with conditional)
- [ ] No `torch.cuda` in any src/ file (except guarded in decoder_inference.py)
- [ ] Notebooks contain no `import cudf` or `import cupy` cells

## 6. Files Modified

### New Files
- `src/tokenizer/backend.py`
- `scripts/train_decoder_hf.py` (new HF Trainer script)
- `configs/pretrain_financial_decoder_hf.yaml` (new HF-compatible config)

### Modified Files
- `src/tokenizer/base.py`
- `src/tokenizer/fixed_vocab.py`
- `src/tokenizer/mapping.py`
- `src/tokenizer/categorical_hash.py`
- `src/tokenizer/timedelta.py`
- `src/tokenizer/numerical.py`
- `src/tokenizer/pipeline.py`
- `src/tokenizer/financial_pipeline.py`
- `src/decoder_inference.py`
- `01_dataset_baseline.ipynb`
- `02_seq_preproc_tokenization.ipynb`
- `03_foundation_model_training.ipynb`
- `04_inference_embedding_extraction.ipynb`
- `05_xgboost_fraud_detection.ipynb`

### Unchanged Files
- `src/tokenizer/__init__.py` (re-exports only)
- `src/tokenizer/financial_tokenizer.py` (pure Python, no GPU deps)
- `src/clm_data.py` (pure Python/PyTorch)
- `configs/pretrain_financial_decoder.yaml` (original preserved)
- `scripts/train_decoder_model.py` (original preserved)
