# Grainger LLM Product-Query Verification Exercise

This project addresses the **Grainger Applied ML LLM Exercise**, designed as a technical component for the Senior/Staff Applied Machine Learning Scientist interview process. The core task involves verifying the accuracy of 'Exact' ('E') labels within a specific subset of the **Amazon ESCI (Experts, Substitutes, Complements, Irrelevant) Shopping Queries Dataset** using Large Language Models (LLMs). The goal is to identify query-product pairs where the 'E' label might be misapplied due to contradictions between the query specifications and the product details, and to suggest reformulated queries for these inaccurate matches based on defined rules.

**Core Task:** Given a product and one of the target search queries initially labeled as an 'Exact' match ('E'), use LLMs to:
1.  Verify if the product information strictly satisfies all specifications mentioned in the query according to predefined rules (Contradiction, Missing Information, Extra Information).
2.  Output a boolean flag (`accurate_label`) indicating the verification result (True if 'E' is correct, False otherwise).
3.  If `accurate_label` is False, provide a `reformulated_query` that accurately reflects the product's key specifications relevant to the original query.

**Target Queries Analyzed:**
*   `aa batteries 100 pack`
*   `kodak photo paper 8.5 x 11 glossy`
*   `dewalt 8v max cordless screwdriver kit, gyroscopic`

## Dataset: Amazon ESCI Shopping Queries

*   **Source:** [amazon-science/esci-data on GitHub](https://github.com/amazon-science/esci-data)
*   **Description:** A large-scale, multilingual dataset designed for research in semantic matching of queries and products. It contains query-product pairs with ESCI relevance judgments (Exact, Substitute, Complement, Irrelevant).
*   **Files Used:** This project utilizes the `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet` files from the `shopping_queries_dataset/` directory within the ESCI repository.
*   **Key Fields Used:** `query_id`, `product_id`, `query`, `esci_label`, `product_locale`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`.

## Project Structure

grainger-llm-product-query-verification/
│
├── .gitignore # Standard Python gitignore
├── LICENSE # e.g., MIT or Apache 2.0
├── README.md # This file
├── requirements.txt # Python package dependencies
│
├── notebooks/
│ └── 01_LLM_Label_Verification.ipynb # Main analysis notebook
│
├── src/
│ ├── init.py
│ ├── config.py # Configuration (file paths, model IDs, prompts)
│ ├── data_processing.py # Data loading, merging, filtering, text prep
│ ├── llm_interaction.py # LLM loading, inference, parsing
│ ├── aggregation.py # Merging results, majority voting
│ └── utils.py # Helper functions (GPU cleanup, download, sys info)
│
├── data/ # Created by script - For downloaded data
│
├── cache/ # Created by script - For optional filtered data cache
│
└── results/ # Created by script - For final output CSVs
├── grainger_llm_verification_results_final_vote.csv
└── grainger_llm_verification_results_final_vote_FULL.csv
## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:jorgeutd/grainger-llm-label-verifier.git
    cd grainger-llm-product-query-verification
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Ensure you have Python 3.11+ and pip. **GPU Set up:** It's highly recommended to install PyTorch matching your specific CUDA version *before* installing other requirements. Visit [https://pytorch.org/](https://pytorch.org/) for instructions.

    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** `bitsandbytes` installation can be OS/CUDA dependent. Consult its documentation if issues arise.

4.  **Hardware:** A **CUDA-enabled GPU with >= 24GB VRAM** is strongly recommended (tested on 40GB A100). 4-bit quantization is used, but models like Mistral-24B still require significant memory. CPU execution is possible but extremely slow.

## Usage

The primary way to run this analysis is via the Jupyter Notebook:

1.  **Launch Jupyter:**
    ```bash
    jupyter lab  # or jupyter notebook
    ```
2.  **Open Notebook:** Navigate to `notebooks/` and open `00_end_to_end_workflow_llm_verfication.ipynb`.
3.  **Run Cells:** Execute the notebook cells sequentially. The notebook handles:
    *   Displaying system information.
    *   Downloading and preparing data (using functions from `src/data_processing.py`).
    *   Sequentially loading each LLM specified in `src/config.py` (Qwen-14B, Gemma-12B, Mistral-Small), running inference (using `src/llm_interaction.py`), and unloading the model to manage GPU memory (using `src/utils.py`).
    *   Aggregating results from all models (using `src/aggregation.py`).
    *   Performing final analysis and generating summary statistics.
    *   Saving the final output CSV files to the `results/` directory.

*(A conceptual script-based execution is possible using the `src/` modules but is not implemented as a single `main.py` in this version.)*

## Approach

1.  **Data Acquisition & Preparation (`src/data_processing.py`):**
    *   Downloads ESCI examples and products parquet files.
    *   Merges datasets on `product_id` and `product_locale`.
    *   Filters for the 3 target queries, 'E' label, and 'us' locale.
    *   Aggregates `product_title`, `description`, `bullet_point`, `brand`, `color` into a cleaned `llm_product_context` string, handling missing data.

2.  **LLM Selection & Setup (`src/config.py`, `src/llm_interaction.py`):**
    *   Uses an ensemble of three models: **Qwen-14B** (`Qwen/Qwen2.5-14B-Instruct-1M`), **Gemma-12B** (`google/gemma-3-12b-it`), and **Mistral-Small-24B** (`mistralai/Mistral-Small-24B-Instruct-2501`).
    *   Applies **4-bit quantization** (`BitsAndBytesConfig`) to all models for memory efficiency.
    *   Loads/unloads models sequentially using Hugging Face `transformers`, `accelerate`, and `bitsandbytes`.

3.  **Prompt Engineering (`src/config.py`):**
    *   Uses a detailed **System Prompt** defining the AI's role and rules.
    *   Uses a **User Prompt Template** providing task details, rules (Contradiction, Missing Info, Extra Info), placeholders for query/context, and the mandatory JSON output structure (`is_exact_match`, `reasoning`, `reformulated_query`).
    *   Formats prompts using model-specific chat templates (`tokenizer.apply_chat_template`).

4.  **Inference & Parsing (`src/llm_interaction.py`):**
    *   `run_llm_verification` orchestrates inference for one query-product pair.
    *   `parse_llm_output_json` robustly extracts the JSON response, handling markdown code blocks and validating required keys/types.

5.  **Aggregation & Majority Voting (`src/aggregation.py`):**
    *   `merge_model_results` combines results from all successful model runs into a single DataFrame.
    *   `get_majority_vote` determines the consensus `accurate_label` (True/False based on >= 2/3 agreement). Ties are impossible with 3 voters providing valid outputs. Aggregates reasoning and identifies consensus `reformulated_query` if the final vote is False. Tracks processing errors.
    *   `apply_majority_voting` applies the vote logic and formats the final output table.

6.  **Output Generation:**
    *   Saves the required 4-column output to `results/grainger_llm_verification_results_final_vote.csv`.
    *   Saves a full table including reasoning and errors to `results/grainger_llm_verification_results_final_vote_FULL.csv`.

## Results Summary (Based on Qwen-14B, Gemma-12B, Mistral-Small-24B)

*   **Models Used:** `Qwen/Qwen2.5-14B-Instruct-1M`, `google/gemma-3-12b-it`, `mistralai/Mistral-Small-24B-Instruct-2501`
*   **Total Items Analyzed:** 24
*   **Final Label Distribution (Consensus Vote):**
    *   ✅ **Accurate:** 16 (66.7%) - *Original 'E' label confirmed correct.*
    *   ❌ **Inaccurate:** 8 (33.3%) - *Original 'E' label deemed incorrect due to contradictions.*
    *   ❓ **Tied / Undecided:** 0 (0.0%) - *Consensus reached for all items.*
*   **Items with Processing Errors during Vote:** 0

*   **Key Findings:**
    *   The 3-model ensemble successfully identified **8** instances where the original 'E' label was incorrect due to explicit contradictions (e.g., pack size, product attributes, technical specs, brand, product type).
    *   A majority consensus (>= 2/3 agreement) was achieved for **all 24 items**, demonstrating the decisiveness of the odd-numbered ensemble.
    *   The generated `reformulated_query` values in the output provide sensible corrections for the 'Inaccurate' items based on product details.

*(See the `notebooks/00_end_to_end_workflow_llm_verfication.ipynbb` notebook and `results/` directory for detailed analysis of consensus cases and individual model performance.)*

## Assumptions & Design Decisions

*   **LLM Ensemble:** Used Qwen-14B, Gemma-12B, Mistral-Small for diversity and robustness.
*   **Quantization:** Applied 4-bit quantization as necessary for hardware constraints (40GB A100).
*   **Prompting:** Single detailed prompt per call, relying on LLM's instruction following and JSON capabilities.
*   **Majority Voting:** Best 2-out-of-3 determines consensus. Guarantees a decision if models provide valid outputs.
*   **Rule Interpretation:** Strictly followed rules provided in the exercise description (esp. regarding missing info).
*   **Text Cleaning:** Basic HTML/whitespace cleaning.
*   **Locale:** Filtered for 'us' locale based on target queries.

## Potential Improvements & Future Work

*   **Structured Extraction:** Extract key specifications (size, count, features) from text first, then compare structurally for more robust contradiction detection.
*   **Fine-tuning:** Fine-tune a smaller, efficient model on verified examples for potentially better performance or cost-efficiency.
*   **Advanced Prompting:** Explore Chain-of-Thought or ReAct patterns for more complex reasoning tasks.
*   **Error Analysis:** Deeper dive into any parsing failures (though none occurred in the final vote) to improve prompt/parsing robustness for edge cases.
*   **Cost Analysis:** Evaluate cost/benefit of using multiple models vs. a single top-tier model in a production/API setting.

## Disclaimer

This project was completed as part of the Grainger interview process. The code and findings are based on the provided dataset and instructions PDF.