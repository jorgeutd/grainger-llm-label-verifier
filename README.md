# Grainger LLM Product-Query Verification Exercise

This project addresses the **Grainger Applied ML LLM Exercise**, designed as a technical component for the Staff Applied Machine Learning Scientist interview process. The core task involves verifying the accuracy of 'Exact' ('E') labels within a specific subset of the **Amazon ESCI (Experts, Substitutes, Complements, Irrelevant) Shopping Queries Dataset** using Large Language Models (LLMs). The goal is to identify query-product pairs where the 'E' label might be misapplied due to contradictions between the query specifications and the product details, and to suggest reformulated queries for these inaccurate matches based on defined rules.

**Core Task:** Given a product and one of the target search queries initially labeled as an 'Exact' match ('E'), use LLMs to:
1.  Verify if the product information strictly satisfies all specifications mentioned in the query according to predefined rules (Contradiction, Missing Information, Extra Information).
2.  Output a boolean flag (`accurate_label`) indicating the verification result (True if 'E' is correct, False otherwise).
3.  If `accurate_label` is False, provide a `reformulated_query` that accurately reflects the product's key specifications relevant to the original query.

**Target Queries Analyzed:**
*   `aa batteries 100 pack`
*   `kodak photo paper 8.5 x 11 glossy`
*   `dewalt 8v max cordless screwdriver kit, gyroscopic`

Two primary approaches are implemented and documented in this repository:
1.  **Multi-LLM Ensemble with Majority Voting:** Uses several open-source LLMs (Qwen-14B, Gemma-12B, Mistral-Small) and aggregates their judgments via voting for robustness. (Implemented in `src/` and demonstrated in `00_end_to_end_workflow_llm_verfication.ipynb`)
2.  **Reflection Agent (LangGraph):** Uses a single powerful generator LLM (Claude 3.5 Sonnet) that iteratively refines its assessment based on critique from a separate, efficient critic LLM (Claude 3.7 Sonnet1). (Implemented in `notebooks/01_agentic_implementation_langgraph.ipynb`)

## Dataset: Amazon ESCI Shopping Queries

*   **Source:** [amazon-science/esci-data on GitHub](https://github.com/amazon-science/esci-data)
*   **Description:** A large-scale, multilingual dataset designed for research in semantic matching of queries and products. It contains query-product pairs with ESCI relevance judgments (Exact, Substitute, Complement, Irrelevant).
*   **Files Used:** This project utilizes the `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet` files from the `shopping_queries_dataset/` directory within the ESCI repository.
*   **Key Fields Used:** `query_id`, `product_id`, `query`, `esci_label`, `product_locale`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`.

## Project Structure
grainger-llm-label-verifier/
│
├── .gitignore
├── README.md                                           # This file
├── requirements.txt                                    # Dependencies
│
├── notebooks/
│   ├── 00_end_to_end_workflow_llm_verfication.ipynb    # Ensemble Approach
│   └── 01_agentic_implementation_langgraph.ipynb       # Reflection Agent Approach
│
├── results/                                            # Contains output CSVs (gitignored by default)
│   ├── grainger_llm_reflection_results_multi_model_04_23.csv      # Reflection Agent Results
│   ├── grainger_llm_reflection_results_multi_model_04_23_FULL.csv # Reflection Agent Full Results
│   ├── grainger_llm_verification_results_final_vote_04_22.csv     # Ensemble Results
│   └── grainger_llm_verification_results_final_vote_FULL.csv      # Ensemble Full Results
│
└── src/                                                # Source code modules (primarily for Ensemble approach)
    ├── aggregation.py
    ├── config.py
    ├── data_processing.py
    ├── llm_interaction.py
    ├── utils.py
    └── __init__.py



## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:jorgeutd/grainger-llm-label-verifier.git
    cd grainger-llm-label-verifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Ensure you have Python 3.11+.
    ```bash
    pip install -r requirements.txt
    ```
    *   **GPU Setup (for Ensemble):** If running the ensemble notebook (`notebooks/00_end_to_end_workflow_llm_verfication.ipynb`), a CUDA-enabled GPU with >= 24GB VRAM. Install PyTorch matching the CUDA version first ([https://pytorch.org/](https://pytorch.org/)). `bitsandbytes` installation might require specific steps depending on your OS/CUDA.
    *   **Visualization ** To render the LangGraph visualization in the reflection notebook, you may need Graphviz system libraries (`apt-get install graphviz libgraphviz-dev` on Debian/Ubuntu) and the Python package (`pip install pygraphviz`).

4.  **API Keys (for Reflection Agent):**
    *   The reflection agent notebook (`notebooks/01_agentic_implementation_langgraph.ipynb`) uses the Anthropic API. Set your API key using **Colab Secrets** (Recommended):
        *   Click the Key icon (🔑) in Colab sidebar.
        *   Add a secret named `ANTHROPIC_API_KEY` with your key value (`sk-ant-...`).
        *   Enable "Notebook access".
    *   Alternatively, create a `.env` file in the project root with `ANTHROPIC_API_KEY='your_key'` (ensure `.env` is in `.gitignore`) or modify the notebook to load the key from the uploaded `.txt` file as demonstrated in the notebook's setup comments.

## Usage

This project provides two distinct implementations, run via their respective notebooks:

### 1. Ensemble Approach (Notebook 00)

*   **File:** `notebooks/00_end_to_end_workflow_llm_verfication.ipynb`
*   **Method:** Loads multiple open-source models (Qwen, Gemma, Mistral) sequentially with 4-bit quantization, runs inference, aggregates results, performs majority voting, and saves outputs.
*   **Requires:** GPU with sufficient VRAM (>=24GB recommended).
*   **Output Files:** `results/grainger_llm_verification_results_final_vote_*.csv`

### 2. Reflection Agent Approach (Notebook 01)

*   **File:** `notebooks/01_agentic_implementation_langgraph.ipynb`
*   **Method:** Uses LangGraph to create an agent loop with a Generator LLM (Claude 3.5 Sonnet) and a Critic LLM (Claude 3.7 Sonnet). The agent verifies, reflects, and revises its assessment until accepted or max iterations are reached.
*   **Requires:** Anthropic API Key (setup via Colab Secrets or other methods). 
*   **Output Files:** `results/grainger_llm_reflection_results_multi_model_04_23.csv`

**To Run:**
1.  Complete the Setup steps.
2.  Launch Jupyter Lab (`jupyter lab`).
3.  Open the desired notebook (`00_...` or `01_...`).
4.  Ensure necessary prerequisites (GPU for Notebook 00, Anthropic API Key for Notebook 01) are met.
5.  Run the notebook cells sequentially.

## Approach Details

### 1. Ensemble (Majority Vote)

*   **Data Prep (`src/data_processing.py`):** Downloads, merges, filters data; aggregates product text into `llm_product_context`.
*   **LLM Inference (`src/llm_interaction.py`):** Loads/unloads models sequentially (Qwen-14B, Gemma-12B, Mistral-Small-24B) with 4-bit quantization. Runs inference using structured prompts requiring JSON output. Parses JSON robustly.
*   **Aggregation (`src/aggregation.py`):** Merges results, performs 2-out-of-3 majority vote for `accurate_label`, determines consensus `reformulated_query`.
*   **Output:** Saves 4-column required CSV and a full CSV with details.

### 2. Reflection Agent (LangGraph)

*   **Data Prep:** Loads pre-processed data from the first notebook's output (`filtered_data_with_context_*.parquet`).
*   **LLM Setup:** Initializes Generator (Claude 3.5 Sonnet) and Critic (Claude 3.7 Sonnet) models via Anthropic API. Binds Generator to Pydantic schema (`VerificationResult`) for structured output.
*   **Graph Definition:**
    *   **State (`VerificationState`):** Tracks query, context, current assessment, critique, revision count, errors.
    *   **Nodes:** `initial_verify` (Generator), `reflect` (Critic), `revise` (Generator).
    *   **Edges:** Defines flow: `verify` -> `reflect`. `reflect` conditionally routes to `revise` (if critique received & iterations < max) or `END` (if "ACCEPT", error, or max iterations). `revise` loops back to `reflect`.
*   **Execution:** Iterates through data, invoking the compiled LangGraph `app` for each item. Processes the final state to determine outcome and extract results.
*   **Output:** Saves 4-column required CSV and a full CSV with details (iterations, outcome, reasoning).

## Results Summary

Both approaches were run on the same 24 filtered query-product pairs.

### Ensemble Approach (Qwen-14B, Gemma-12B, Mistral-Small)

*   **Final Label Distribution (Consensus Vote):**
    *   ✅ **Accurate:** 16 (66.7%)
    *   ❌ **Inaccurate:** 8 (33.3%)
    *   ❓ **Tied / Undecided:** 0 (0.0%)
*   **Key Finding:** Achieved 100% consensus across the 24 items.

### Reflection Agent Approach (Claude 3.5 Sonnet + Claude 3 Haiku)

*   **Final Label Distribution (After Reflection):**
    *   ✅ **Accurate:** 16 (66.7%)
    *   ❌ **Inaccurate:** 8 (33.3%)
    *   ❓ **Undetermined:** 0 (0.0%) *(The 1 item hitting Max Iterations still yielded a final label)*
*   **Workflow Outcomes:**
    *   **Accepted:** 23 items reached "ACCEPT".
    *   **Max Iterations Reached:** 1 item hit the limit.
    *   **Errors:** 0 items encountered processing errors.
*   **Key Finding:** Demonstrated self-correction capability (e.g., Item 6) and converged to the same final classifications as the ensemble method for this dataset.

*(See the respective notebooks and `results/` directory for detailed analysis.)*

## Comparison of Approaches

| Feature               | Ensemble (Majority Vote)                     | Reflection Agent (Self-Correction)           |
| :-------------------- | :------------------------------------------- | :------------------------------------------- |
| **Core Idea**         | Wisdom of the crowd; diverse perspectives    | Iterative refinement; self-critique          |
| **LLM Usage**         | Multiple (e.g., 3) models run sequentially   | 1 Generator + 1 Critic (sequential loop)   |
| **Implementation**    | Modular Python scripts (`src/`), Notebook 00 | LangGraph state machine, Notebook 01         |
| **Robustness**        | High (reduces impact of single model failure)| High (potential for self-correction)       |
| **Latency (per item)**| Sum of sequential model runs                 | Sequential loop; depends on iterations       |
| **Compute/Cost**      | Higher (multiple large models, GPU needed)   | API costs; depends on iterations             |
| **Complexity**        | Simpler voting logic                         | More complex graph/state/prompt management   |
| **Error Handling**    | Handles individual model failures via voting | Can correct initial errors; relies on critic |
| **Outcome (This Task)**| 16 Accurate, 8 Inaccurate                   | 16 Accurate, 8 Inaccurate                   |

**Conclusion (This Task):** For this specific dataset and task rules, both the multi-model ensemble (using quantized open-source models) and the reflection agent (using Claude API models) effectively identified the same 8 inaccurate labels and achieved the same final classification results. The reflection agent showcased its ability to self-correct based on critique, while the ensemble demonstrated robustness through voting. The choice in a production setting would depend on factors like available hardware (GPU vs. API), latency requirements, budget, and the desired level of explicit self-correction logic.

## Assumptions & Design Decisions

*   **LLMs:** Specific models chosen for each approach based on capability and accessibility (open-source vs. API).
*   **Quantization (Ensemble):** Applied 4-bit quantization for feasibility on available hardware.
*   **Prompting:** Detailed prompts designed to elicit specific JSON output and enforce rules.
*   **Voting/Reflection Logic:** Implemented majority voting (Ensemble) and a verify-reflect-revise loop (Reflection Agent).
*   **Rule Interpretation:** Strictly followed rules provided in the PDF exercise description (esp. regarding missing info).
*   **Text Cleaning:** Basic HTML/whitespace cleaning applied to product context.

## Potential Improvements & Future Work

*   **Hybrid Approaches:** Combine ensemble generation with a reflection step on the consensus result.
*   **Structured Extraction:** Pre-process text to extract key attributes (size, count, voltage, etc.) for more direct comparison, potentially reducing LLM ambiguity.
*   **Fine-tuning:** Fine-tune smaller models (open-source or API-based) on verified data for efficiency.
*   **Advanced Agentic Patterns:** I could Explore more complex planning agents if reasoning becomes more intricate.
*   **Error Analysis & Recovery:** Implement more robust error handling within the LangGraph agent (e.g., retry mechanisms, specific error states) and LangSmith observability to our reflection agent. 
*   **Scalability:** For larger datasets, we could batch API calls (Reflection) or parallel execution across GPUs (Ensemble).

## Disclaimer

This project was completed as part of the Grainger interview process. The code and findings are based on the provided dataset and instructions.