**Multilingual PII Detection in Noisy Social Media via LLaMA 3.1 & QLoRA**
📌 Research Abstract

This project addresses the critical challenge of detecting Personally Identifiable Information (PII) in unstructured, multilingual social media text (English, French, Spanish). Unlike formal documents, social media comments contain slang, emojis, and encoding artifacts (e.g., Ã© mojibake) that cause traditional Zero-Shot LLM approaches to fail.

We conducted a comprehensive benchmark of Falcon-3, Mistral-7B, and LLaMA-3.1 using Zero/One/Few-Shot strategies. Our findings revealed a "Semantic Gap" where models conflate generic nouns (e.g., "my mother") with specific PII. To resolve this, we implemented Parameter-Efficient Fine-Tuning (QLoRA) on a mixed-language dataset, achieving significant performance gains over the best prompting baselines.

🚀 Key Results (The Delta)

Fine-tuning proved mandatory for non-English languages. While Prompt Engineering hit a hard ceiling, QLoRA successfully adapted the model to the noisy domain.

Language

Best Baseline (Prompting)

Fine-Tuned (QLoRA)

Relative Improvement

Status

English

0.5039 F1 (Falcon Few-Shot)

0.6617 F1

+31.31%

Robust Success

Spanish

0.2266 F1 (Falcon Few-Shot)

0.4269 F1

+88.39%

Doubled Performance

French

0.1018 F1 (Falcon One-Shot)

0.2983 F1

+193.02%

Tripled Performance

Scientific Insight: The French baseline was near 0.00 F1 in Zero-Shot due to tokenization collapse on UTF-8 noise. QLoRA fine-tuning successfully taught the model to parse these artifacts, bridging the technical gap that prompting could not solve.

🛠️ Methodology

1. Data Preparation: "The Union Strategy"

Social media datasets often suffer from fragmented labeling. We implemented a robust Union Strategy to construct ground truth:

Input: Raw text (including noise and encoding errors).

Labels: We dynamically merged PII_DETECTION columns (Phones, Emails) with NER_DETECTION columns (Names, Locations) to create a comprehensive "Master Truth."

Split: 80% Training / 10% Validation / 10% Testing (Held-out).

2. Benchmarking (Phase 1)

We evaluated off-the-shelf performance using three strategies:

Zero-Shot: Baseline capabilities. Result: High hallucination rates (up to 30%).

One-Shot: Grounding via single example. Result: Fixed JSON formatting but failed semantic distinction.

5-Shot (Few-Shot): Contextual learning. Result: Worked for Spanish/English but caused Negative Transfer in French.

3. Fine-Tuning (Phase 2)

We fine-tuned LLaMA 3.1 8B Instruct using QLoRA:

Quantization: 4-bit NF4 (to fit on consumer GPUs).

Adapters: LoRA Rank=16, Alpha=16.

Data: ~1,200 rows of mixed English, Spanish, and French instructions (master_train.jsonl).

Optimization: Training loss converged from 1.81 to 0.49.

📂 Repository Structure

├── English-python-file/                # Scripts for English benchmarks
├── English-result/                     # Raw output logs for English
├── Spanish-python-file/                # Scripts for Spanish benchmarks
├── Spanish-result/                     # Raw output logs for Spanish
├── French-python-file/                 # Scripts for French benchmarks
├── French-result/                      # Raw output logs for French
├── FineTuned_Validation-result/        # Final Validation logs after fine-tuning
├── Llama-3.1-8B-PII-Multilingual-Best/ # **The Final Model** (QLoRA adapter weights)
├── finetuning/                         # Helper scripts for training setup
├── train_qlora_smart.py                # Main training script (SFTTrainer)
├── benchmark_finetuned_multilingual.py # Unified inference & evaluation script
├── master_train.jsonl                  # The compiled training dataset (~1200 rows)
├── FINAL_CONSOLIDATED_REPORT-*.csv     # Summary performance tables
└── README.md


💻 Installation & Usage

1. Environment Setup

# Clone repository
git clone [https://github.com/Jemskc/llama3-multilingual-pii-redteam.git](https://github.com/Jemskc/llama3-multilingual-pii-redteam.git)
cd llama3-multilingual-pii-redteam

# Install dependencies
pip install torch transformers peft datasets bitsandbytes trl accelerate pandas


2. Run Benchmarks (Reproduce Phase 3)

To reproduce the validation on the fine-tuned model across all three languages:

python benchmark_finetuned_multilingual.py


3. Train the Model (Reproduce Phase 2)

Requires master_train.jsonl (included in repo).

python train_qlora_smart.py


4. Run Inference (Using the Fine-Tuned Adapter)

You can load the trained adapter from the Llama-3.1-8B-PII-Multilingual-Best folder:

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load Base Model
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")

# 2. Load Adapter from local folder
model = PeftModel.from_pretrained(base, "Llama-3.1-8B-PII-Multilingual-Best")

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Inference logic...


🛡️ Ethical Considerations

This project is designed for Red Teaming and Data Loss Prevention (DLP). The datasets used are anonymized social media comments. The goal is to improve the safety of automated systems by detecting sensitive data leakage in noisy, informal communication channels.
