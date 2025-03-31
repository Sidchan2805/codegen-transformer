# codegen-transformer
# 🚀 Transformer-based Python Code Generation (CodeT5+)

This project revolves around a carefully designed transformer-based Python code-generation pipeline built using CodeT5+ (220M), fine-tuned on extensive, carefully curated data from CodexGLUE (50k examples) and APPS datasets. The goal is to accurately generate Python code snippets from natural language prompts.



---
## Key Highlights
**Transformer-based Model:** Fine-tuned CodeT5+ (220M parameters).
**Datasets:** Robust training on CodexGLUE (filtered 50k examples) and APPS (entire filtered set).
**Chunk-wise Incremental Training:** For optimal use of computing resources data is divided into manageable 10k-example batches, training incrementally with checkpoints after each batch.
**Evaluation:** Evaluated on the MBPP benchmark for validating the real world performance of the model. Used metrics such as rouge, bleu, chrf

## ⚙️ Training Details (Clearly Explained):

- **Model**: Salesforce CodeT5+ (220M params)
- **Datasets**:
  - **CodexGLUE**: 50,000 filtered examples
  - **APPS**: All filtered examples
- **Training strategy**: Chunk-wise incremental training (10k examples per chunk), checkpointing after each chunk.

## Tech stack used 
- **Data filtering and cleaning**- From both the datasets examples are filetred out by ensuring that the natural language prompts solutions have less than 512 tokens. This reduces computation overheads during attention mechanism in transfomer processing stage in the hidden layers
- **Chunk wise JSONL Storage**- To manage the RAM and momeory resources while batch creation and data feeding to the model the data is structured into JSONL files post tokenization. This method is optimized for streaming.
- **Incremental Training Pipeline**- Incremental training loop used to process one data chunk at a time. To simulate the batch feeding of data gathered_batch_gradients command is used. Model checkpoints stored after each 10k chunk training to ensure recoverability

## Performance Results
Metric     Score(Approx)
ROUGE-1    33%
ROUGE-2    13%
ROUGE-L    30%
BLEU       11%
ChrF       48%

---

## 📌 Quick Start Guide:

### 🔹 Installation:
```bash
pip install -r requirements.txt
