import pandas as pd
import json
import re
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = "english_train.csv"
OUTPUT_FILE = "english_prepared.jsonl"

# ==========================================
# 2. PARSING LOGIC
# ==========================================
def load_and_clean_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    # Clean headers just in case
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Drop empty rows (The "Garbage Removal")
    initial_count = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]
    
    print(f"[INFO] Initial rows: {initial_count}")
    print(f"[INFO] Cleaned rows: {len(df)}")
    
    return df

def parse_gold_labels(label_str):
    """
    Parses 'PHONE_NUMBER(123)' into dict.
    """
    if pd.isna(label_str) or str(label_str).lower() in ['nan', 'none', '']:
        return {}
    
    # Comprehensive Tag Map (Consistent with French/Spanish)
    tag_map = {
        "PHONE_NUMBER": "PHONE", "EMAIL_ADDRESS": "EMAIL", "PERSON": "PERSON",
        "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION", "DATE_TIME": "DATE_TIME",
        "US_DRIVER_LICENSE": "ID_NUM", "US_SSN": "ID_NUM", "IP_ADDRESS": "IP_ADDRESS",
        "APP": "APP", "AGE": "AGE", "DATE": "DATE_TIME", 
        "PROD": "PRODUCT", "MISC": "MISC", "ORG": "ORGANIZATION"
    }
    
    result = {}
    matches = re.findall(r'([A-Z_]+)\((.*?)\)', str(label_str))
    for tag, val in matches:
        clean_key = tag_map.get(tag, tag)
        if clean_key not in result:
            result[clean_key] = []
        val_clean = val.strip()
        # Avoid duplicates
        if val_clean not in result[clean_key]:
            result[clean_key].append(val_clean)
    return result

# ==========================================
# 3. LLAMA FORMATTING
# ==========================================
def format_llama_entry(text, labels):
    system_msg = (
        "You are an expert PII detection system. Extract ALL sensitive entities "
        "(PERSON, EMAIL, PHONE, LOCATION, ID_NUM, PRODUCT, etc). "
        "Return output strictly in JSON format."
    )
    
    assistant_response = json.dumps(labels, ensure_ascii=False)
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Text: {text}\nOutput JSON:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_response}<|eot_id|>"
    )
    return {"text": prompt}

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def process_english():
    print("-" * 60)
    print("PROCESSING ENGLISH DATA FOR FINE-TUNING")
    print("-" * 60)
    
    df = load_and_clean_csv(INPUT_FILE)
    if df is None: return
    
    formatted_rows = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        # English uses 'detected_entities' column
        gold_label = parse_gold_labels(row.get('detected_entities', ''))
        
        jsonl_entry = format_llama_entry(text, gold_label)
        formatted_rows.append(jsonl_entry)

    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in formatted_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"[SUCCESS] Saved {len(formatted_rows)} English training rows to {OUTPUT_FILE}")
    
    # Verification
    print("\n[VERIFICATION] Row 0 Preview:")
    print(formatted_rows[0]['text'][:300] + "...")

if __name__ == "__main__":
    process_english()
