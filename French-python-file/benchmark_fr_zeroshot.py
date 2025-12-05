"""
--------------------------------------------------------------------------------
SCRIPT: Zero-Shot Benchmark for French PII Detection (Union Strategy)
--------------------------------------------------------------------------------

WHAT THIS CODE IS ABOUT:
This script establishes the baseline performance of three Large Language Models 
(Falcon-3, Mistral-7B, LLaMA-3.1) on the French dataset without providing any 
training examples (Zero-Shot inference).

HOW IT WORKS (THE LOGIC):
1. Data Loading: It reads 'french_test.csv' (50 rows).
2. The "Union Strategy" (CRITICAL): 
   The dataset contains valid labels in two separate columns due to legacy formatting:
   - 'french_pii_detection': Contains PII like Phone/Email.
   - 'NER FOUND IN SPANISH': Mislabeled column actually containing French NER (Names/Locations).
   
   The script parses BOTH columns and merges them into a single 'Ground Truth' dictionary. 
   If it didn't do this, the models would be penalized for finding valid entities 
   that happened to be in the "other" column.

3. Inference: It feeds the raw French text to each model with a strict JSON system prompt.
4. Evaluation: It compares the model's prediction against the Merged Ground Truth.
5. Reporting: It generates 3 CSV tables (Executive Summary, Class Breakdown, Hallucinations).

--------------------------------------------------------------------------------
"""

import os
import sys
import re
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODELS_TO_TEST = [
    "tiiuae/Falcon3-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct"
]

TEST_FILE = "french_test.csv"
OUTPUT_DIR = "Benchmark_Results/French_ZeroShot_Final"

# ==========================================
# 2. DATA PROCESSING (Verified Logic)
# ==========================================
def load_and_clean_csv(filepath):
    """Loads CSV and strips whitespace from headers to fix ' NER' bug."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip() # Fixes the leading space in column names
    return df

def parse_gold_labels(label_str):
    if pd.isna(label_str) or str(label_str).lower() == 'nan':
        return {}
    
    # Comprehensive Mapping
    tag_map = {
        "PHONE_NUMBER": "PHONE", "EMAIL_ADDRESS": "EMAIL", "PERSON": "PERSON",
        "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION", "DATE_TIME": "DATE_TIME",
        "US_DRIVER_LICENSE": "ID_NUM", "US_SSN": "ID_NUM", "IP_ADDRESS": "IP_ADDRESS",
        "APP": "APP", "AGE": "AGE", "DATE": "DATE_TIME", 
        "PROD": "PRODUCT", "MISC": "MISC", "ORG": "ORGANIZATION",
        "CARDINAL": "NUMBER", "ORDINAL": "NUMBER", "WORK_OF_ART": "WORK_OF_ART",
        "NORP": "GROUP", "PERCENT": "PERCENT"
    }
    
    result = {}
    matches = re.findall(r'([A-Z_]+)\((.*?)\)', str(label_str))
    for tag, val in matches:
        clean_key = tag_map.get(tag, tag)
        if clean_key not in result:
            result[clean_key] = []
        if val.strip() not in result[clean_key]:
            result[clean_key].append(val.strip())
    return result

def merge_gold_data(row):
    """
    Verified Union Strategy: PII + NER = Master Truth
    """
    # 1. Parse PII Column
    pii_dict = parse_gold_labels(row.get('french_pii_detection', ''))
    
    # 2. Parse NER Column (using the cleaned name)
    ner_dict = parse_gold_labels(row.get('NER FOUND IN SPANISH', ''))
    
    # 3. Merge
    merged = pii_dict.copy()
    for key, val_list in ner_dict.items():
        if key not in merged:
            merged[key] = []
        for v in val_list:
            if v not in merged[key]:
                merged[key].append(v)
    return merged

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# ==========================================
# 3. PROMPT & INFERENCE
# ==========================================
def get_zeroshot_prompt(tokenizer, text):
    """
    Standard Zero-Shot Prompt (No Examples)
    """
    system_msg = (
        "You are an expert PII detection system. Analyze the French text. "
        "Extract ALL sensitive entities (PERSON, EMAIL, PHONE, LOCATION, ID_NUM, PRODUCT, etc). "
        "Return output strictly in JSON format."
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text: {text}\nOutput JSON:"}
    ]
    
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"System: {system_msg}\nUser: Text: {text}\nOutput JSON:\nAssistant:"

def run_inference():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"[INFO] Loading Data...")
    df_test = load_and_clean_csv(TEST_FILE)
    
    raw_results = []

    for model_id in MODELS_TO_TEST:
        short_name = model_id.split("/")[-1]
        print(f"\n[PROCESSING] {short_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        # Inference Loop
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"   > Testing"):
            text = str(row['text'])
            # MERGE LABELS
            gold_label = merge_gold_data(row)
            
            prompt = get_zeroshot_prompt(tokenizer, text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=350, temperature=0.1, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                pred_label = eval(json_match.group(0)) if json_match else {}
            except:
                pred_label = {}

            # Score
            tp = 0; fp = 0; fn = 0; hallucinations = 0
            all_keys = set(gold_label.keys()).union(set(pred_label.keys()))
            for key in all_keys:
                g_items = set([x.lower().strip() for x in gold_label.get(key, [])])
                p_items = set([str(x).lower().strip() for x in pred_label.get(key, [])])
                
                tp += len(g_items.intersection(p_items))
                fp_items = p_items - g_items
                fp += len(fp_items)
                fn += len(g_items - p_items)
                
                for item in fp_items:
                    if item not in text.lower():
                        hallucinations += 1

            raw_results.append({
                "Model": short_name,
                "Text_Full": text,
                "Gold": str(gold_label),
                "Pred": str(pred_label),
                "TP": tp, "FP": fp, "FN": fn, "Hallucinations": hallucinations
            })
        
        del model
        del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    return pd.DataFrame(raw_results)

# ==========================================
# 4. REPORT GENERATOR
# ==========================================
def generate_tables(df):
    print(f"\n[ANALYSIS] Generating Tables in {OUTPUT_DIR}...")
    models = df['Model'].unique()
    
    model_stats = {m: {'micro': {'tp':0,'fp':0,'fn':0,'hal':0}, 'classes': {}} for m in models}
    global_support = {}
    hallucination_log = []
    
    for _, row in df.iterrows():
        model = row['Model']
        gold = eval(row['Gold'])
        pred = eval(row['Pred'])
        
        model_stats[model]['micro']['tp'] += row['TP']
        model_stats[model]['micro']['fp'] += row['FP']
        model_stats[model]['micro']['fn'] += row['FN']
        model_stats[model]['micro']['hal'] += row['Hallucinations']
        
        if row['Hallucinations'] > 0:
            all_keys = set(gold.keys()).union(set(pred.keys()))
            for key in all_keys:
                g_items = set([x.lower().strip() for x in gold.get(key, [])])
                p_items = set([str(x).lower().strip() for x in pred.get(key, [])])
                fp_items = p_items - g_items
                for item in fp_items:
                    hallucination_log.append({
                        "Model": model,
                        "Hallucinated Entity": f"{key}: {item}",
                        "Input Text Snippet": str(row['Text_Full'])[:100]
                    })

        all_keys = set(gold.keys()).union(set(pred.keys()))
        for key in all_keys:
            k = key.upper()
            if model == models[0]:
                g_cnt = len(gold.get(key, []))
                if k not in global_support: global_support[k] = 0
                global_support[k] += g_cnt

            g_items = set([x.lower().strip() for x in gold.get(key, [])])
            p_items = set([str(x).lower().strip() for x in pred.get(key, [])])
            
            tp = len(g_items.intersection(p_items))
            fp = len(p_items - g_items)
            fn = len(g_items - p_items)
            
            if k not in model_stats[model]['classes']:
                model_stats[model]['classes'][k] = {'tp':0, 'fp':0, 'fn':0}
            model_stats[model]['classes'][k]['tp'] += tp
            model_stats[model]['classes'][k]['fp'] += fp
            model_stats[model]['classes'][k]['fn'] += fn

    # Table 1
    table1 = []
    for model in models:
        m = model_stats[model]['micro']
        mic_p, mic_r, mic_f1 = calculate_f1(m['tp'], m['fp'], m['fn'])
        
        class_f1s = []
        for k in model_stats[model]['classes']:
            c = model_stats[model]['classes'][k]
            _, _, c_f1 = calculate_f1(c['tp'], c['fp'], c['fn'])
            class_f1s.append(c_f1)
        mac_f1 = np.mean(class_f1s) if class_f1s else 0.0
        
        total_pred = m['tp'] + m['fp']
        hal_rate = (m['hal'] / total_pred * 100) if total_pred > 0 else 0.0
        
        table1.append({
            "Model Name": model,
            "Micro-Precision": f"{mic_p:.4f}",
            "Micro-Recall": f"{mic_r:.4f}",
            "Micro-F1": f"{mic_f1:.4f}",
            "Macro-F1": f"{mac_f1:.4f}",
            "Hallucination Rate": f"{hal_rate:.2f}%"
        })
    pd.DataFrame(table1).to_csv(f"{OUTPUT_DIR}/table1_executive_summary.csv", index=False)

    # Table 2
    table2 = []
    sorted_classes = sorted(list(global_support.keys()))
    for k in sorted_classes:
        row = {"Entity Category": k}
        for model in models:
            if k in model_stats[model]['classes']:
                c = model_stats[model]['classes'][k]
                _, _, c_f1 = calculate_f1(c['tp'], c['fp'], c['fn'])
                row[f"{model} (F1)"] = f"{c_f1:.4f}"
            else:
                row[f"{model} (F1)"] = "0.0000"
        row["Total Count (Support)"] = global_support[k]
        table2.append(row)
    pd.DataFrame(table2).to_csv(f"{OUTPUT_DIR}/table2_entity_breakdown.csv", index=False)

    # Table 3
    pd.DataFrame(hallucination_log).to_csv(f"{OUTPUT_DIR}/table3_hallucination_log.csv", index=False)

    print(f"[COMPLETE] Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    raw_df = run_inference()
    raw_df.to_excel(f"{OUTPUT_DIR}/raw_results_robust.xlsx", index=False)
    generate_tables(raw_df)
