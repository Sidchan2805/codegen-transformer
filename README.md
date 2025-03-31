# codegen-transformer
# 🚀 Transformer-based Python Code Generation (CodeT5+)

An advanced Python code-generation model fine-tuned on the **CodexGLUE** (50k filtered examples) and **APPS** datasets using incremental JSONL streaming training with checkpoints.

---

## ⚙️ Training Details (Clearly Explained):

- **Model**: Salesforce CodeT5+ (220M params)
- **Datasets**:
  - **CodexGLUE**: 50,000 filtered examples (≤512 tokens)
  - **APPS**: All filtered examples (≤512 tokens)
- **Training strategy**: Chunk-wise incremental training (10k examples per chunk), checkpointing after each chunk.

---

## 📌 Quick Start Guide:

### 🔹 Installation:
```bash
pip install -r requirements.txt
