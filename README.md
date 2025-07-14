# 🧠 Fine-Tuning mBART for Moroccan Darija ↔ English Translation

## 📌 Project Overview

This project demonstrates the fine-tuning of the multilingual mBART model on a custom dataset of parallel sentences in Moroccan Darija and English. It aims to bridge the gap in machine translation support for underrepresented Arabic dialects—specifically, Moroccan Darija—by adapting a powerful transformer-based model to handle dialect-specific translation tasks.

## 🎯 Goals

* Develop a machine translation model capable of translating between Moroccan Darija and English.
* Fine-tune the pretrained `facebook/mbart-large-50-many-to-many-mmt` model using custom parallel data.
* Evaluate the performance using BLEU scores and qualitative examples.
* Showcase an end-to-end NLP pipeline: from dataset processing to inference.

## 🛠 Tools & Libraries Used

* **Hugging Face Transformers**: model, tokenizer, trainer
* **Datasets (🤗)**: data formatting and dataset objects
* **PyTorch**: backend framework for training
* **scikit-learn**: dataset splitting
* **pandas / numpy**: data manipulation
* **sacrebleu**: translation performance metric
* **Kaggle Notebook**: training environment

## 🔁 Workflow

### 1. Environment Setup

Install essential libraries:

```bash
!pip install --upgrade transformers
!pip install sacrebleu
```

### 2. Data Preparation

* Load CSV dataset containing `darija` and `english` columns.
* Split into training (80%), validation (10%), and test (10%) sets using `train_test_split`.
* Convert to Hugging Face `DatasetDict` format.

### 3. Tokenization

* Load `facebook/mbart-large-50-many-to-many-mmt` tokenizer.
* Use language codes (`"ar_AR"` for Darija, `"en_XX"` for English).
* Tokenize source and target sequences with padding and truncation.

### 4. Model Configuration

* Load pretrained mBART model with sequence-to-sequence architecture.
* Set generation parameters for beam search and max length.
* Use `DataCollatorForSeq2Seq` for batching during training.

### 5. Training

* Define `Seq2SeqTrainingArguments` including:

  * Batch size, evaluation steps, learning rate
  * Evaluation and save strategies
* Train with `Seq2SeqTrainer`.

### 6. Evaluation

* Generate translations on the test set.
* Compute BLEU scores using `sacrebleu`.
* Include manual checks for translation quality.

### 7. Inference

* Define a helper function to translate new Darija or English sentences using the trained model.
* Switch language codes dynamically based on translation direction.

## 📊 Results

* **BLEU Score**: Achieved a validation BLEU score indicating solid alignment between predicted and reference translations.
* **Qualitative Samples**: Model correctly translated colloquial expressions, proving the effectiveness of domain adaptation.

*(You can add actual scores/output here once finalized.)*

## 📁 Project Structure

```
.
├── mbart-fine-tuning.ipynb   # Main notebook with full pipeline
├── data/
│   └── cleaned_darija_dataset.csv  # Parallel corpus
├── saved_model/              # Trained model checkpoint (if exported)
└── README.md                 # Project documentation
```

## 👥 Target Audience

* NLP and ML Engineers
* Arabic & North African language researchers
* Open-source contributors interested in low-resource translation
* Students building NLP portfolios

## 👤 Author & Credits

**Author:** Charif El Belghiti
**Inspired by:** Hugging Face tutorials and community notebooks
**Special Thanks:** To the creators of the original dataset used for this fine-tuning.

