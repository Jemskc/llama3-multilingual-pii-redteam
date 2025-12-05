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
TRAIN_FILE = "french_train.csv"
OUTPUT_DIR = "Benchmark_Results/French_FewShot_5"
NUM_SHOTS = 5  # <--- Using 5 distinct examples to teach variety

# ==========================================
# 2. DATA PROCESSING (The Verified Logic)
# ==========================================
def load_and_clean_csv(filepath):
    """Loads CSV and strips whitespace from headers to fix ' NER' bug."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip() # Fixes the space in column names
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
    
    # 2. Parse NER Column (Robust check)
    ner_col = 'NER FOUND IN SPANISH' 
    ner_dict = parse_gold_labels(row.get(ner_col, ''))
    
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
def get_fewshot_prompt(tokenizer, test_text, example_rows):
    """
    Builds a 5-Shot history using MERGED labels for the examples.
    """
    system_msg = (
        "You are an expert PII detection system. Analyze the French text. "
        "Extract ALL sensitive entities (PERSON, EMAIL, PHONE, LOCATION, ID_NUM, PRODUCT, etc). "
        "Return output strictly in JSON format."
    )
    
    messages = [{"role": "system", "content": system_msg}]

    # Build History from 5 Examples
    for _, row in example_rows.iterrows():
        ex_text = str(row['text'])
        # CRITICAL: Show the model the MERGED truth
        ex_label = merge_gold_data(row)
        ex_json = str(ex_label).replace("'", '"')
        
        messages.append({"role": "user", "content": f"Text: {ex_text}\nOutput JSON:"})
        messages.append({"role": "assistant", "content": ex_json})

    # The Test Question
    messages.append({"role": "user", "content": f"Text: {test_text}\nOutput JSON:"})
    
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        # Fallback
        prompt = f"System: {system_msg}\n"
        for _, row in example_rows.iterrows():
            ex_t = str(row['text'])
            ex_l = str(merge_gold_data(row)).replace("'", '"')
            prompt += f"User: Text: {ex_t}\nOutput JSON:\nAssistant: {ex_l}\n"
        prompt += f"User: Text: {test_text}\nOutput JSON:\nAssistant:"
        return prompt

def run_inference():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"[INFO] Loading Data...")
    df_test = load_and_clean_csv(TEST_FILE)
    df_train = load_and_clean_csv(TRAIN_FILE)
    
    # Pick 5 Random Examples (Seed 42 guarantees the same 5 every time)
    example_rows = df_train.sample(n=NUM_SHOTS, random_state=42)
    print(f"[INFO] Using 5-Shot Example IDs: {example_rows['id'].tolist()}")

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
            gold_label = merge_gold_data(row)
            
            # Pass the 5 examples here
            prompt = get_fewshot_prompt(tokenizer, text, example_rows)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # Increased tokens because 5-shot prompt is longer
                outputs = model.generate(
                    **inputs, max_new_tokens=400, temperature=0.1, do_sample=True,
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
