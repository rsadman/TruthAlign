# TruthAlign: Detecting Hallucinations in Generated Text

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

TruthAlign is an automated hallucination detection system that classifies LLM-generated text into three categories: **No Hallucination**, **Partial Hallucination**, and **Hallucinating**. This project implements and compares two distinct fine-tuning approaches using Microsoft's DeBERTa-v3-small model.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Experiments](#experiments)
- [Challenges & Limitations](#challenges--limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Large language models (LLMs) like GPT-3 and GPT-4 can generate fluent text that contains factual inaccuracies—a phenomenon known as "hallucination." TruthAlign addresses this critical challenge by fine-tuning transformer-based models to automatically detect varying degrees of hallucination in generated biographical text.

### Research Questions

- How effectively can DeBERTa-v3-small detect hallucinations in LLM-generated text?
- What are the performance differences between full fine-tuning and LoRA?
- How do these approaches handle class imbalance?
- What are the computational trade-offs?

## ✨ Key Features

- **Dual Fine-Tuning Approaches**: Implements both full fine-tuning and parameter-efficient LoRA adaptation
- **Multi-Class Classification**: Detects three levels of hallucination severity
- **Comprehensive Evaluation**: Includes accuracy, F1 scores, confusion matrices, and per-class metrics
- **Production-Ready Pipeline**: Complete preprocessing, training, and inference system
- **Reproducible Research**: Fixed random seeds and detailed configuration

## 📊 Results

### Performance Comparison

| Approach | Test Accuracy | Macro F1 Score | Trainable Parameters |
|----------|--------------|----------------|---------------------|
| **Full Fine-Tuning** | **62.50%** | **0.4833** | 142M (100%) |
| **LoRA Fine-Tuning** | 60.42% | 0.2511 | 150K (0.11%) |

### Key Findings

- ✅ Full fine-tuning achieved superior performance with better class distribution
- ✅ Successfully distinguished between "No Hallucination" and "Partial Hallucination" classes
- ⚠️ Both approaches struggled with the minority "Hallucinating" class due to severe class imbalance
- 💡 LoRA offers 99.9% parameter reduction but requires careful tuning for imbalanced datasets

### Per-Class Performance (Full Fine-Tuning)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No Hallucination | 0.50 | 0.67 | 0.57 |
| Partial Hallucination | 0.69 | 0.76 | 0.72 |
| Hallucinating | 0.00 | 0.00 | 0.00 |

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/HumairaKabir/TruthAlign.git
cd TruthAlign

# Install dependencies
pip install torch transformers datasets peft scikit-learn numpy pandas matplotlib seaborn
```

### Required Libraries

```python
transformers==4.36.0
torch>=2.0.0
datasets>=2.15.0
peft>=0.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 💻 Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model
model_path = "./deberta-hallucination"  # Path to your trained model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Inference function
def check_hallucination(generated_text, reference_text):
    inputs = tokenizer(
        generated_text,
        reference_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    
    labels = {0: "No Hallucination", 1: "Partial Hallucination", 2: "Hallucinating"}
    return labels[pred]

# Example usage
generated = "Jean Hugo was the son of Victor Hugo."
reference = "Jean Hugo was the great-grandson of Victor Hugo."
result = check_hallucination(generated, reference)
print(f"Prediction: {result}")
```

### Training from Scratch

#### Full Fine-Tuning

```bash
# Open and run the notebook
jupyter notebook 24rd_Nov_Update.ipynb
```

Key configuration:
```python
training_args = TrainingArguments(
    output_dir="./deberta-hallucination",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)
```

#### LoRA Fine-Tuning

```bash
# Open and run the notebook
jupyter notebook 8thDecUpdateTest.ipynb
```

Key configuration:
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
```

## 📚 Dataset

**Wiki-Bio GPT-3 Hallucination Dataset**
- Source: `potsawee/wiki_bio_gpt3_hallucination` (Hugging Face)
- Size: 48 test samples (after 80/20 split)
- Features: GPT-3 generated biographical texts with Wikipedia references

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| No Hallucination | 12 | 25.0% |
| Partial Hallucination | 29 | 60.4% |
| Hallucinating | 7 | 14.6% |

## 🏗️ Model Architecture

### Base Model
- **DeBERTa-v3-small** (Microsoft)
- 44M parameters
- Disentangled attention mechanism
- Max sequence length: 512 tokens

### Fine-Tuning Approaches

**1. Full Fine-Tuning**
- All 142M parameters trainable
- Learning rate: 2×10⁻⁵
- 10 epochs
- Best performance: Epoch 7

**2. LoRA (Low-Rank Adaptation)**
- Only 150K parameters trainable (0.11%)
- Rank (r): 8
- Alpha: 16
- Learning rate: 2×10⁻⁴
- 8 epochs
- 99.9% parameter reduction

## 🧪 Experiments

### Notebooks

1. **`24rd_Nov_Update.ipynb`**: Full fine-tuning implementation
   - Complete training pipeline
   - Comprehensive evaluation metrics
   - Visualization of results

2. **`8thDecUpdateTest.ipynb`**: LoRA fine-tuning implementation
   - Parameter-efficient adaptation
   - Comparative analysis
   - Training dynamics

### Running Experiments

```bash
# For Google Colab users
!git clone https://github.com/HumairaKabir/TruthAlign.git
%cd TruthAlign
!pip install -r requirements.txt

# Run full fine-tuning
%run 24rd_Nov_Update.ipynb

# Run LoRA fine-tuning
%run 8thDecUpdateTest.ipynb
```

## ⚠️ Challenges & Limitations

### Data Challenges
- **Severe class imbalance**: Only 14.6% "Hallucinating" samples
- **Small dataset size**: Limited test samples (n=48)
- **Label ambiguity**: Subjective boundaries between "Partial" and "Hallucinating"

### Model Challenges
- Failed to detect minority "Hallucinating" class
- LoRA showed high sensitivity to class imbalance
- DeBERTa-v3-small capacity limitations for nuanced detection

### Technical Challenges
- GPU memory constraints
- Long training time (97 minutes for full fine-tuning)
- LoRA hyperparameter optimization complexity

## 🔮 Future Work

### Short-term Improvements
- [ ] Implement class weighting and focal loss
- [ ] Test larger DeBERTa variants (base, large)
- [ ] Apply data augmentation techniques
- [ ] Experiment with higher LoRA ranks (16-64)

### Long-term Goals
- [ ] Multi-task learning with NLI and fact verification
- [ ] Cross-domain evaluation (medical, scientific, technical)
- [ ] Multi-lingual hallucination detection
- [ ] Real-world deployment with human-in-the-loop
- [ ] Explainability features (attention visualization)

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{kabir2024truthalign,
  title={TruthAlign: Detecting Hallucinations in Generated Text},
  author={Kabir, Kazi Humaira},
  journal={CCSE 596 Project Report},
  year={2024},
  institution={North South University}
}
```

## 🙏 Acknowledgments

- **Dr. Md Adnan Arefeen** for guidance and support
- **North South University** for computational resources
- **Hugging Face** for the Transformers library and datasets
- **Microsoft Research** for the DeBERTa model
- Original authors of the Wiki-Bio GPT-3 Hallucination dataset

## 📧 Contact

**Kazi Humaira Kabir**
- Email: kazi.kabir.241@northsouth.edu
- GitHub: [@HumairaKabir](https://github.com/HumairaKabir)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ If you find this project helpful, please consider giving it a star!
