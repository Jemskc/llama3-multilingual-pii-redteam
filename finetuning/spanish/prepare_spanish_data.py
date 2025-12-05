import pandas as pd
import json
import re
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = "spanish_train.csv"
OUTPUT_FILE = "spanish_prepared.jsonl"

# ==========================================
# 2. ROBUST PARSING LOGIC
# ==========================================
def load_and_clean_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    # Clean headers (strip spaces/quotes) to handle " NER..." issues
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Drop empty rows to ensure training stability
    initial_count = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]
    
    print(f"[INFO] Initial rows: {initial_count}")
    print(f"[INFO] Cleaned rows: {len(df)}")
    
    return df

def find_ner_column(df):
    """Dynamically finds the Spanish NER column."""
    for col in df.columns:
        if "NER" in col and "spanish" in col.lower():
            return col
    return None

def parse_gold_labels(label_str):
    """Parses tags like 'PHONE(123)' into a dict."""
    if pd.isna(label_str) or str(label_str).lower() in ['nan', 'none', '']:
        return {}
    
    # Comprehensive Tag Map
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

def merge_gold_data(row, ner_col_name):
    """Union Strategy: PII + NER"""
    # 1. Parse PII
    pii_dict = parse_gold_labels(row.get('spanish_pii_detection', ''))
    # 2. Parse NER
    ner_dict = {}
    if ner_col_name:
        ner_dict = parse_gold_labels(row.get(ner_col_name, ''))
    # 3. Merge
    merged = pii_dict.copy()
    for key, val_list in ner_dict.items():
        if key not in merged:
            merged[key] = []
        for v in val_list:
            if v not in merged[key]:
                merged[key].append(v)
    return merged

# ==========================================
# 3. LLAMA FORMATTING LOGIC
# ==========================================
def format_llama_entry(text, labels):
    """
    Constructs the exact string structure LLaMA 3.1 expects.
    """
    system_prompt = (
        "You are an expert PII detection system. Analyze the Spanish text. "
        "Extract ALL sensitive entities (PERSON, EMAIL, PHONE, LOCATION, ID_NUM, PRODUCT, etc). "
        "Return output strictly in JSON format."
    )
    
    assistant_response = json.dumps(labels, ensure_ascii=False)
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Text: {text}\nOutput JSON:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_response}<|eot_id|>"
    )
    return {"text": prompt}

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def process_spanish():
    print("-" * 60)
    print("PROCESSING SPANISH DATA FOR FINE-TUNING")
    print("-" * 60)
    
    df = load_and_clean_csv(INPUT_FILE)
    if df is None: return

    ner_col = find_ner_column(df)
    print(f"[INFO] NER Column detected: '{ner_col}'")
    
    formatted_rows = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        gold_label = merge_gold_data(row, ner_col)
        
        jsonl_entry = format_llama_entry(text, gold_label)
        formatted_rows.append(jsonl_entry)

    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in formatted_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    # Verification Print
    print("\n[VERIFICATION] Row 0 Preview:")
    print(formatted_rows[0]['text'][:300] + "...")
    
    print("-" * 60)
    print(f"TOTAL ROWS PROCESSED AND SAVED: {len(formatted_rows)}")
    print("-" * 60)

if __name__ == "__main__":
    process_spanish()
