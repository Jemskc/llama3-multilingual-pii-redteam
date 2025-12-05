import os
import sys
import re
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Path to your saved QLoRA adapter
ADAPTER_PATH = "Llama-3.1-8B-PII-Multilingual-Best" 
OUTPUT_DIR = "Benchmark_Results/FineTuned_Validation"

# Dataset Configuration
DATASETS = {
    "English": {
        "file": "english_val.csv",
        "type": "simple", 
        "pii_col": "detected_entities"
    },
    "French": {
        "file": "french_val.csv", 
        "type": "merged", 
        "pii_col": "french_pii_detection", 
        "ner_col_keyword": "NER" # Will look for 'NER FOUND IN SPANISH'
    },
    "Spanish": {
        "file": "spanish_val.csv", 
        "type": "merged", 
        "pii_col": "spanish_pii_detection",
        "ner_col_keyword": "NER" # Will look for 'NER Found...'
    }
}

# ==========================================
# 2. DATA PROCESSING HELPERS
# ==========================================
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_and_clean_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    # Clean headers (fix spaces/quotes)
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

def find_column_by_keyword(df, keyword):
    """Finds a column containing specific keywords (case-insensitive)."""
    for col in df.columns:
        if keyword.lower() in col.lower() and "spanish" in col.lower():
            return col
    return None

def parse_gold_labels(label_str):
    if pd.isna(label_str) or str(label_str).lower() in ['nan', 'none', '']:
        return {}
    
    # Comprehensive Tag Map (Merging EN, FR, ES tags)
    tag_map = {
        "PHONE_NUMBER": "PHONE", "EMAIL_ADDRESS": "EMAIL", "PERSON": "PERSON",
        "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION", "DATE_TIME": "DATE_TIME",
        "US_DRIVER_LICENSE": "ID_NUM", "US_SSN": "ID_NUM", "IP_ADDRESS": "IP_ADDRESS",
        "APP": "APP", "AGE": "AGE", "DATE": "DATE_TIME", 
        "PROD": "PRODUCT", "MISC": "MISC", "ORG": "ORGANIZATION",
        "LOC": "LOCATION", "PER": "PERSON", 
        "CARDINAL": "NUMBER", "ORDINAL": "NUMBER", "WORK_OF_ART": "WORK_OF_ART",
        "NORP": "GROUP", "PERCENT": "PERCENT", "EVENT": "EVENT", "MONEY": "MONEY",
        "HEIGHT": "MISC", "MEASUREMENT": "MISC", "TIME": "DATE_TIME"
    }
    
    result = {}
    matches = re.findall(r'([A-Z_]+)\((.*?)\)', str(label_str))
    for tag, val in matches:
        clean_key = tag_map.get(tag, tag)
        if clean_key not in result:
            result[clean_key] = []
        val_clean = val.strip()
        if val_clean not in result[clean_key]:
            result[clean_key].append(val_clean)
    return result

def get_gold_label(row, config, df_columns):
    """
    Extracts the Ground Truth based on language strategy.
    """
    # 1. Simple Strategy (English)
    if config['type'] == "simple":
        return parse_gold_labels(row.get(config['pii_col'], ''))
    
    # 2. Merged Strategy (French/Spanish)
    elif config['type'] == "merged":
        # Parse PII
        pii_dict = parse_gold_labels(row.get(config['pii_col'], ''))
        
        # Find NER Column dynamically if not already known
        ner_col_name = None
        for col in df_columns:
            # Logic: Contains "NER" and "spanish" (common for both files)
            if "NER" in col and "spanish" in col.lower():
                ner_col_name = col
                break
        
        # Parse NER
        ner_dict = parse_gold_labels(row.get(ner_col_name, ''))
        
        # Merge
        merged = pii_dict.copy()
        for key, val_list in ner_dict.items():
            if key not in merged:
                merged[key] = []
            for v in val_list:
                if v not in merged[key]:
                    merged[key].append(v)
        return merged
    
    return {}

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# ==========================================
# 3. INFERENCE ENGINE
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("-" * 60)
    print(f"[START] Fine-Tuned Validation Benchmark")
    print(f"[MODEL] Base: {BASE_MODEL_ID}")
    print(f"[ADAPTER] {ADAPTER_PATH}")
    print("-" * 60)

    # 1. Load Base Model & Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # 2. Load Fine-Tuned Adapter
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("[SUCCESS] Fine-Tuned Model Loaded.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not load model: {e}")
        return

    summary_table = []

    # 3. Loop Through Each Language
    for lang, config in DATASETS.items():
        print(f"\n{'='*40}")
        print(f"TESTING LANGUAGE: {lang}")
        print(f"{'='*40}")
        
        df = load_and_clean_csv(config['file'])
        if df is None: continue
        
        raw_results = []
        
        # TQDM Loop
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row['text'])
            
            # Get Truth using correct logic
            gold_label = get_gold_label(row, config, df.columns)
            
            # Construct Zero-Shot Prompt (Model should know the task now)
            system_msg = (
                "You are an expert PII detection system. Analyze the text. "
                "Extract ALL sensitive entities (PERSON, EMAIL, PHONE, LOCATION, ID_NUM, PRODUCT, etc). "
                "Return output strictly in JSON format."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Text: {text}\nOutput JSON:"}
            ]
            
            # Apply Chat Template
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs, 
                    max_new_tokens=350, 
                    temperature=0.1, 
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Parse JSON output
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                pred_label = eval(json_match.group(0)) if json_match else {}
            except:
                pred_label = {}

            # Scoring Logic
            tp = 0; fp = 0; fn = 0; hallucinations = 0
            all_keys = set(gold_label.keys()).union(set(pred_label.keys()))
            
            for key in all_keys:
                g_items = set([str(x).lower().strip() for x in gold_label.get(key, [])])
                p_items = set([str(x).lower().strip() for x in pred_label.get(key, [])])
                
                tp += len(g_items.intersection(p_items))
                fp_items = p_items - g_items
                fp += len(fp_items)
                fn += len(g_items - p_items)
                
                for item in fp_items:
                    if item not in text.lower():
                        hallucinations += 1

            raw_results.append({
                "Language": lang,
                "Text_Full": text,
                "Gold": str(gold_label),
                "Pred": str(pred_label),
                "TP": tp, "FP": fp, "FN": fn, "Hallucinations": hallucinations
            })

        # Save Raw Data per Language
        df_res = pd.DataFrame(raw_results)
        df_res.to_excel(f"{OUTPUT_DIR}/raw_results_{lang}.xlsx", index=False)
        
        # Calculate Metrics
        tp_sum, fp_sum, fn_sum = df_res['TP'].sum(), df_res['FP'].sum(), df_res['FN'].sum()
        hal_sum = df_res['Hallucinations'].sum()
        p, r, f1 = calculate_f1(tp_sum, fp_sum, fn_sum)
        hal_rate = (hal_sum / (tp_sum+fp_sum) * 100) if (tp_sum+fp_sum) > 0 else 0.0
        
        print(f"   > {lang} Results: F1={f1:.4f} | Hal%={hal_rate:.2f}%")
        
        summary_table.append({
            "Language": lang,
            "Micro-Precision": f"{p:.4f}",
            "Micro-Recall": f"{r:.4f}",
            "Micro-F1": f"{f1:.4f}",
            "Hallucination Rate": f"{hal_rate:.2f}%"
        })

    # Save Final Comparison Table
    pd.DataFrame(summary_table).to_csv(f"{OUTPUT_DIR}/FINAL_VALIDATION_REPORT.csv", index=False)
    print(f"\n[DONE] Final Report saved to: {OUTPUT_DIR}/FINAL_VALIDATION_REPORT.csv")

if __name__ == "__main__":
    main()
