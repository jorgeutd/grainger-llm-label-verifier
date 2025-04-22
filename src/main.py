# src/main.py
import pandas as pd
import warnings
import time
from tqdm.auto import tqdm # For progress bar

# Import from local modules
from . import config
from .data_processing import download_file, prepare_data_for_llm
from .llm_interaction import load_llm, run_llm_check_json_chat, unload_llm, log_memory_usage
from .aggregation import combine_and_vote

# Suppress specific warnings if needed (e.g., FutureWarning from pandas merge)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find the quantized class") # warning

def run_pipeline():
    """Runs the full LLM label verification pipeline."""
    print("="*30 + " Starting Grainger LLM Verification Pipeline " + "="*30)
    start_time_pipeline = time.time()

    # --- 1. Data Preparation ---
    print("\n--- Step 1: Data Preparation ---")
    try:
        # download_file(config.EXAMPLES_URL, config.LOCAL_EXAMPLES_PATH)
        # download_file(config.PRODUCTS_URL, config.LOCAL_PRODUCTS_PATH)
        # Assuming files are already downloaded or paths point correctly
        # Use web URLs directly if not downloading
        examples_src = config.EXAMPLES_URL # config.LOCAL_EXAMPLES_PATH
        products_src = config.PRODUCTS_URL # config.LOCAL_PRODUCTS_PATH

        df_processed = prepare_data_for_llm(
            examples_path=examples_src,
            products_path=products_src,
            target_queries=config.TARGET_QUERIES,
            target_label=config.TARGET_LABEL,
            target_locale=config.TARGET_LOCALE
        )
    except Exception as e:
        print(f"FATAL ERROR during data preparation: {e}")
        return # Stop execution

    if df_processed.empty:
        print("ERROR: No data found matching the criteria. Pipeline cannot continue.")
        return

    # --- 2. LLM Inference ---
    print("\n--- Step 2: LLM Inference ---")
    all_model_results = {} # Store results {model_key: [list_of_result_dicts]}
    models_used_in_inference = [] # Track which models actually ran

    for model_key, params in config.LLM_CONFIG.items():
        print(f"\n--- Processing Model: {model_key} ({params['model_id']}) ---")
        model, tokenizer = None, None # Ensure cleanup scope
        results_current_model = []
        try:
            # Load Model
            model, tokenizer = load_llm(
                model_key=model_key,
                model_id=params['model_id'],
                trust_remote=params['trust_remote'],
                quant_setting=params.get('quantization') # Use .get for safety
            )

            if model is None or tokenizer is None:
                print(f"Skipping inference for {model_key} due to loading failure.")
                continue # Skip to next model

            models_used_in_inference.append(model_key) # Mark model as used
            num_rows = len(df_processed)
            start_time_model = time.time()

            # Inference Loop for this model
            for index, row in tqdm(df_processed.iterrows(), total=num_rows, desc=f"Inferring {model_key}"):
                query = row['query']
                product_context = row.get('llm_product_context', "No product information available.") # Safe get
                query_id = row['query_id']
                product_id = row['product_id']

                # Run inference
                llm_output = run_llm_check_json_chat(
                    query=query,
                    product_context=product_context,
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=config.SYSTEM_PROMPT,
                    user_template=config.USER_PROMPT_TEMPLATE,
                    max_new_tokens=config.INFERENCE_MAX_NEW_TOKENS,
                    temperature=config.INFERENCE_TEMPERATURE,
                    top_p=config.INFERENCE_TOP_P,
                    do_sample=config.INFERENCE_DO_SAMPLE,
                    model_name_key=model_key
                )

                # Append results including identifiers
                results_current_model.append({
                    'query_id': query_id,
                    'product_id': product_id,
                    'accurate_label': llm_output['is_exact_match'],
                    'reasoning': llm_output['reasoning'],
                    'reformulated_query': llm_output['reformulated_query'],
                    'raw_output': llm_output['raw_output'],
                    'parsing_error': llm_output['parsing_error']
                })

                # Log parsing errors immediately if they occur
                if llm_output['parsing_error']:
                    tqdm.write(f"  Warning: {model_key} Parsing Error (Row {index}): {llm_output['parsing_error']}. Raw: '{llm_output['raw_output'][:100]}...'")

            end_time_model = time.time()
            print(f"  {model_key} inference completed in {end_time_model - start_time_model:.2f}s (Avg: {(end_time_model - start_time_model)/num_rows:.2f}s/row)")
            all_model_results[model_key] = results_current_model

        except Exception as e:
             print(f"FATAL ERROR during inference loop for {model_key}: {e}")
             # Continue to next model if one fails, but log it
        finally:
             # Unload model to free memory
             if model is not None or tokenizer is not None:
                 unload_llm(model, tokenizer, model_key)
                 log_memory_usage(f"After unloading {model_key}") # Log memory after unload

    # --- 3. Aggregation and Voting ---
    print("\n--- Step 3: Aggregation and Voting ---")
    if not all_model_results or len(models_used_in_inference) < 2:
        print(f"ERROR: Insufficient successful model results ({len(models_used_in_inference)} models) to perform voting. Pipeline halting.")
        # Optionally save the individual results that *were* generated
        # for key, results in all_model_results.items():
        #     if results: pd.DataFrame(results).to_csv(f"{config.RESULTS_DIR}/{key}_individual_results.csv", index=False)
        return

    # Combine results and apply majority voting
    df_final_combined = combine_and_vote(all_model_results, models_used_in_inference)

    if df_final_combined.empty:
        print("ERROR: Failed to combine results or apply voting. Pipeline halting.")
        return

    # --- 4. Final Output Preparation & Saving ---
    print("\n--- Step 4: Preparing Final Output ---")
    try:
        # Merge back original query for context if available
        if 'query' in df_processed.columns:
             df_final_output = pd.merge(
                 df_final_combined,
                 df_processed[['query_id', 'product_id', 'query']].drop_duplicates(),
                 on=['query_id', 'product_id'],
                 how='left'
             )
        else:
             df_final_output = df_final_combined # Proceed without query if not present

        # Rename final columns for submission
        df_final_output = df_final_output.rename(columns={
            'final_accurate': 'accurate_label',
            'final_reformulated': 'reformulated_query'
        })

        # Select and order columns for the final deliverable CSV
        submission_df = df_final_output[config.SUBMISSION_COLUMNS].copy()
        # Clean final submission data (e.g., fillna for reformulation)
        submission_df['reformulated_query'] = submission_df['reformulated_query'].fillna('').astype(str)
        # accurate_label can remain boolean/None, CSV handles it.

        # Save submission file
        submission_df.to_csv(config.FINAL_OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"Successfully saved final submission file to: {config.FINAL_OUTPUT_CSV}")

        # Optionally save the full DataFrame with all intermediate results/reasoning
        if config.FULL_OUTPUT_CSV:
            # Select desired columns for full output
            # Example: keep identifiers, final vote, consensus, errors, and individual model votes
            full_cols = ['query_id', 'product_id', 'query'] if 'query' in df_final_output.columns else ['query_id', 'product_id']
            full_cols += ['accurate_label', 'reformulated_query', 'consensus_reasoning', 'voting_errors']
            # Add individual model accuracy columns
            for key in models_used_in_inference:
                if f'acc_{key}' in df_final_output.columns: full_cols.append(f'acc_{key}')
            # Filter for existing columns before saving
            df_full_to_save = df_final_output[[col for col in full_cols if col in df_final_output.columns]]
            df_full_to_save.to_csv(config.FULL_OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"Successfully saved full results table to: {config.FULL_OUTPUT_CSV}")

    except KeyError as e:
        print(f"ERROR: Missing expected column during final output preparation: {e}")
    except Exception as e:
        print(f"ERROR during final output saving: {e}")

    # --- Pipeline Summary ---
    end_time_pipeline = time.time()
    print("\n" + "="*30 + " Pipeline Completed " + "="*30)
    print(f"Total execution time: {end_time_pipeline - start_time_pipeline:.2f} seconds")
    # Basic results summary
    print(f"\nFinal results summary (from {config.FINAL_OUTPUT_CSV}):")
    if 'submission_df' in locals():
        print(submission_df['accurate_label'].value_counts(dropna=False).rename({True: 'Accurate', False: 'Inaccurate', pd.NA: 'Tied/Undecided'}))
    print("="*60)


if __name__ == "__main__":
    run_pipeline()