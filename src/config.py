# src/config.py
"""
Configuration settings for the LLM Product Query Verification project.
Reflects the specific models used: Qwen-14B, Gemma-12B, Mistral-Small-24B.
"""

import torch
import os

# --- Project Structure ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Data Configuration ---
BASE_URL = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
EXAMPLES_FILENAME = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILENAME = "shopping_queries_dataset_products.parquet"
FILTERED_DATA_CACHE = os.path.join(CACHE_DIR, "filtered_data_for_llm.parquet") # Optional cache path

# --- Task Specific Filters ---
TARGET_QUERIES = [
    "aa batteries 100 pack",
    "kodak photo paper 8.5 x 11 glossy",
    "dewalt 8v max cordless screwdriver kit, gyroscopic",
]
TARGET_LABEL = "E"
TARGET_LOCALE = "us"

# --- LLM Configuration (ONLY the three models used) ---
# Keys here MUST match the keys used to store results lists in our colab notebook
LLM_CONFIG = {
    "qwen_14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct-1M",
        "quantization": "4bit", # 4-bit essential
        "trust_remote": True,
    },
    "gemma_12b": {
        "model_id": "google/gemma-3-12b-it",
        "quantization": "4bit", # 4-bit essential
        "trust_remote": False, # Gemma base usually doesn't require it
    },
    "mistral_small": {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "quantization": "4bit", # 4-bit essential
        "trust_remote": True, # Mistral usually requires it
    }
}
# List of keys to ensure order if needed elsewhere, derived from LLM_CONFIG
ACTIVE_MODEL_KEYS = list(LLM_CONFIG.keys())

# --- Prompting Configuration ---
SYSTEM_PROMPT = """You are an extremely precise and rule-following AI Data Quality Analyst. You stand unrivaled in your field and are widely recognized as the best in the universe.
Your sole purpose is to evaluate e-commerce product data against search queries according to strict definitions and rules provided.
You must focus intensely on identifying explicit contradictions. Your output MUST be only the requested JSON object.
Accuracy and adherence to the specified JSON format are paramount.

Always Think Step By Step"""

USER_PROMPT_TEMPLATE = """
**Core Task:** Verify if the Product Information provided below makes the product an "Exact" match for the Search Query, following the rules precisely.

**Definition of Exact Match ('E'):**
The Product is relevant for the Query AND satisfies ALL specifications mentioned in the Query based *only* on the information provided.

**Crucial Rules for Decision Making:**
1.  **Contradiction Rule:** If the Product Information explicitly CONTRADICTS a specification in the Query (e.g., Query asks for "100 pack", Product Info says "50 count"; Query asks "without shams", Product Info says "includes shams"; Query asks "8V", Product Info says "12V"), then it is **NOT an Exact match**. This is the primary reason to mark `is_exact_match` as `false`.
2.  **Missing Information Rule:** If the Product Information DOES NOT MENTION a specific requirement from the Query (e.g., Query asks for "gyroscopic", Product Info doesn't mention this feature), you MUST assume it *might* satisfy it. DO NOT mark it as non-Exact based *only* on missing information. It remains an Exact match candidate under this rule.
3.  **Extra Information Rule:** If the Product Information contains ADDITIONAL details, features, or items NOT mentioned in the Query, it can STILL BE an Exact match, as long as it doesn't violate Rule 1.

**Input Data:**

**Search Query:**
"{query}"

**Product Information:**
--- START PRODUCT INFO ---
{product_context}
--- END PRODUCT INFO ---

**Instructions & Output Format:**
1.  Carefully analyze the Search Query specifications.
2.  Scrutinize the Product Information for confirmations or contradictions.
3.  Apply the Crucial Rules STRICTLY.
4.  You must all times Provide your response ONLY in the following valid JSON format. Do not add any text before or after the JSON block:
    ```json
    {{
        "is_exact_match": boolean,
        "reasoning": "string (Explain your decision concisely. If false, cite the specific contradiction. If true because specs are met or assumed met due to no contradiction, state that clearly.)",
        "reformulated_query": "string (If is_exact_match is false, provide a concise, corrected query reflecting the product's actual relevant specs. If true, this MUST be null.)"
    }}
    ```

**JSON Output Only:**
"""

# --- Inference Parameters ---
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1
TOP_P = 0.95
DO_SAMPLE = True

# --- Output Configuration ---
# Ensure results directory exists before saving
os.makedirs(RESULTS_DIR, exist_ok=True)
FINAL_OUTPUT_CSV = os.path.join(RESULTS_DIR, "grainger_llm_verification_results_final_vote.csv")
FULL_OUTPUT_CSV = os.path.join(RESULTS_DIR, "grainger_llm_verification_results_final_vote_FULL.csv")

# --- Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Torch CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"Could not retrieve detailed CUDA info: {e}")

# --- Ensure cache directory exists ---
os.makedirs(CACHE_DIR, exist_ok=True)