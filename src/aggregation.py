# src/aggregation.py
"""
Functions for merging results from multiple LLM runs and performing
majority voting to determine a consensus verification result.
"""
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple

# Use relative imports within the src package
from . import config # May need config for model keys if not passed explicitly

def merge_model_results(
    results_dict: Dict[str, List[Dict[str, Any]]],
    model_keys: List[str]
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Merges results lists from multiple models into a single DataFrame.

    Each model's results ('accurate_label', 'reformulated_query', 'reasoning',
    'parsing_error') are added as distinct columns (e.g., 'acc_qwen_14b').
    Uses an outer join to keep all unique query_id/product_id pairs.

    Parameters:
        results_dict (Dict): Dictionary where keys are model_keys and values are
                             lists of result dictionaries from run_llm_verification.
        model_keys (List): List of model keys identifying which results in the
                           dictionary should be merged.

    Returns:
        tuple:
            - pd.DataFrame or None: The merged DataFrame, or None if merging fails.
            - List[str]: The list of model keys that were successfully merged (had valid data).
    """
    print("\n" + "="*30 + " Step: Merge Model Results " + "="*30)
    df_merged_all = None
    successfully_merged_keys = [] # Track keys that actually get merged

    if not results_dict or not model_keys:
        print("ERROR: results_dict or model_keys is empty. Cannot merge.")
        return None, []

    for i, model_key in enumerate(model_keys):
        if model_key not in results_dict or not results_dict[model_key]:
             print(f"Warning: No results found for model key '{model_key}'. Skipping.")
             continue

        print(f"Processing results for merging: {model_key}...")
        try:
            # Create DataFrame from the list of dictionaries for the current model
            df_model = pd.DataFrame(results_dict[model_key])

            # --- Data Validation ---
            required_cols = ['query_id', 'product_id', 'is_exact_match', 'reformulated_query', 'reasoning', 'parsing_error']
            if not all(col in df_model.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_model.columns]
                print(f"  [ERROR] Missing required columns in results for {model_key}: {missing}. Skipping merge for this model.")
                continue # Skip to next model

            # Select and rename columns to be model-specific
            # Using f-strings for cleaner renaming
            df_model_renamed = df_model[['query_id', 'product_id'] + required_cols[2:]].rename(columns={
                'is_exact_match': f'acc_{model_key}',         # Renamed to match usage
                'reformulated_query': f'reform_{model_key}',
                'reasoning': f'reason_{model_key}',
                'parsing_error': f'parse_err_{model_key}'
            })

            # --- Perform Merge ---
            if df_merged_all is None:
                # Initialize with the first valid model's results
                df_merged_all = df_model_renamed
                print(f"  Initialized merged DataFrame with {model_key} results.")
            else:
                # Merge subsequent models using an outer join on identifiers
                # Outer join ensures rows are kept even if one model failed processing for it
                df_merged_all = pd.merge(
                    df_merged_all,
                    df_model_renamed,
                    on=['query_id', 'product_id'],
                    how='outer' # Keep all unique query/product pairs
                )
                print(f"  Merged results from {model_key}.")

            successfully_merged_keys.append(model_key) # Add key if merge step completed

        except Exception as e:
            print(f"  [ERROR] Failed to process or merge results for model {model_key}: {e}")
            # Do not add to successfully_merged_keys if an error occurs

    # --- Final Checks ---
    if df_merged_all is None:
        print("ERROR: Failed to create a merged DataFrame. No valid model results were processed.")
        return None, []
    if not successfully_merged_keys:
         print("ERROR: No models were successfully merged into the DataFrame.")
         return df_merged_all, [] # Return the potentially initialized (but not added to) DF? Or None? Let's return None.
         # return None, []
    elif len(successfully_merged_keys) < 2:
         print(f"Warning: Only {len(successfully_merged_keys)} model ({successfully_merged_keys}) successfully merged. Majority voting requires at least 2.")
         # Proceeding allows analysis of the single model, but voting logic needs >= 2


    print(f"\nSuccessfully merged results from {len(successfully_merged_keys)} models: {successfully_merged_keys}")
    print(f"Merged DataFrame shape: {df_merged_all.shape}")
    return df_merged_all, successfully_merged_keys


def get_majority_vote(row: pd.Series, model_keys: List[str]) -> pd.Series:
    """
    Performs majority voting on a single row of merged LLM results.

    Determines the final accuracy label (True, False, or None for tie/error)
    and the consensus reformulated query if the final label is False. Aggregates
    reasoning and collects parsing/voting errors.

    Parameters:
        row (pd.Series): A row from the merged DataFrame containing results
                         (acc_*, reform_*, reason_*, parse_err_*) for each model key.
        model_keys (List[str]): A list of model keys that were successfully merged
                                and should participate in the vote for this row.

    Returns:
        pd.Series: Contains the final voted label ('final_accurate'), aggregated
                   reasoning ('consensus_reasoning'), the consensus reformulated
                   query ('final_reformulated'), and any voting/parsing errors
                   encountered ('voting_errors'). Index names match these keys.
    """
    votes = [] # List to store valid boolean votes (True/False)
    reformulations_if_false = [] # Collect reformulations ONLY from models that voted False
    reasonings = [] # Collect reasoning strings from all valid votes
    parsing_errors = [] # Collect errors encountered for this row

    # --- Iterate through each model's results for the current row ---
    for key in model_keys:
        # Define expected column names for this model
        acc_col = f'acc_{key}'
        reason_col = f'reason_{key}'
        reform_col = f'reform_{key}'
        parse_err_col = f'parse_err_{key}'

        # --- Robustness Checks ---
        # 1. Check if columns for this model exist in the row (important for outer joins)
        if acc_col not in row.index or parse_err_col not in row.index:
             # This shouldn't happen if model_keys only contains successfully merged keys, but safety first.
             parsing_errors.append(f"{key}: Result columns missing in merged row.")
             continue # Skip to the next model for this row

        # 2. Check if the model had a parsing error reported for this row
        # Use pd.notna to handle None/NaN correctly
        if pd.notna(row.get(parse_err_col)):
            parsing_errors.append(f"{key}: Parsing Failed ({row.get(parse_err_col)})")
            continue # Skip vote from this model as it failed parsing

        # 3. Check if the accuracy vote is a valid boolean value (not None, NaN, or other types)
        vote_val = row.get(acc_col)
        # `isinstance(vote_val, bool)` correctly handles Python True/False
        # It will be False for None, NaN, strings, numbers, etc.
        if not isinstance(vote_val, bool):
            parsing_errors.append(f"{key}: Invalid 'accurate_label' value ({vote_val}). Expected boolean.")
            continue # Skip vote from this model

        # --- If checks pass, record the valid vote and reasoning ---
        votes.append(vote_val)
        # Format reasoning clearly indicating the source model and its vote
        reasonings.append(f"[{key.upper()} Vote: {str(vote_val).upper()}] {row.get(reason_col, 'N/A')}")

        # 4. If the vote was False, record its proposed reformulation (if provided and valid)
        if not vote_val: # If vote is False
            reformulation = row.get(reform_col)
            # Check if reformulation is a non-empty string
            if pd.notna(reformulation) and isinstance(reformulation, str) and reformulation.strip():
                reformulations_if_false.append(reformulation.strip()) # Add cleaned reformulation
            # else: # Optional: note if a False voter didn't provide reformulation
            #     reasonings.append(f"[{key.upper()} Note: Voted False but no reformulation provided/valid.]")

    # --- Determine the Final Verdict based on collected valid votes ---
    num_valid_votes = len(votes)
    final_accurate = None # Default to None (tie or insufficient votes/errors)
    final_reformulated = None # Default to None
    consensus_reasoning = "; ".join(reasonings) # Combine collected reasonings

    if num_valid_votes == 0:
        consensus_reasoning = (consensus_reasoning + "; " if consensus_reasoning else "") + "No valid votes received from any model."
        # final_accurate remains None
    else:
        num_true = sum(votes) # Count True votes (True == 1, False == 0)
        num_false = num_valid_votes - num_true # Count False votes

        # Apply majority rule
        if num_true > num_false:
            final_accurate = True
        elif num_false > num_true:
            final_accurate = False
        else: # Tie (e.g., 1 True, 1 False if only 2 models provided valid votes)
            final_accurate = None # Explicitly mark ties as None/Undecided
            consensus_reasoning += "; [Result: Tied Vote]" # Add tie indicator to reasoning

        # Determine the consensus reformulation ONLY IF the final verdict is False
        if final_accurate is False and reformulations_if_false:
            # Find the most common reformulation proposed by the models that voted False
            try:
                 reform_counts = Counter(reformulations_if_false)
                 if reform_counts: # Ensure the counter is not empty
                    # Get the most frequent reformulation string
                    final_reformulated = reform_counts.most_common(1)[0][0]
                 # else: # This case should be unlikely if reformulations_if_false is not empty
                 #    final_reformulated = "[Error: Reformulations list was empty despite False votes]"
            except Exception as e:
                 # Handle potential errors with Counter or list processing
                 final_reformulated = f"[Error finding consensus reformulation: {e}]"
                 warnings.warn(f"Error processing reformulations: {e} for row {row.get('query_id', 'N/A')}/{row.get('product_id', 'N/A')}")


    # --- Compile Voting/Parsing Errors ---
    # Combine any parsing errors collected during the loop
    voting_errors_str = "; ".join(parsing_errors) if parsing_errors else None # Use None if no errors

    # --- Return results as a Pandas Series ---
    # Ensure index names match expected keys for downstream use
    return pd.Series([final_accurate, consensus_reasoning, final_reformulated, voting_errors_str],
                     index=['final_accurate', 'consensus_reasoning', 'final_reformulated', 'voting_errors'])


def apply_majority_voting(
    df_merged: pd.DataFrame,
    model_keys: List[str],
    original_df: Optional[pd.DataFrame] = None # Make original_df optional
    ) -> Optional[pd.DataFrame]:
    """
    Applies the majority voting logic to the merged results DataFrame and
    formats the final output table.

    Parameters:
        df_merged (pd.DataFrame): The DataFrame with merged results from models.
                                  Must contain 'query_id', 'product_id', and model-specific
                                  columns (acc_*, reason_*, reform_*, parse_err_*).
        model_keys (List[str]): List of model keys that were successfully merged and
                                should be included in the vote.
        original_df (Optional[pd.DataFrame]): The original filtered dataframe containing
                                             the 'query' column for context. If None,
                                             the 'query' column won't be added.

    Returns:
        pd.DataFrame or None: The final DataFrame with voted labels, reformulations,
                              reasoning, and errors, or None if voting fails.
    """
    print("\n" + "="*30 + " Step: Apply Majority Voting & Format Output " + "="*30)

    if not isinstance(df_merged, pd.DataFrame) or df_merged.empty:
         print("ERROR: Merged DataFrame is empty or invalid. Cannot apply voting.")
         return None
    if not model_keys or len(model_keys) < 1: # Need at least 1 model to proceed, though voting is trivial
         print("ERROR: No valid model keys provided for voting.")
         return None
    if len(model_keys) < 2:
         print(f"Warning: Applying 'voting' with only {len(model_keys)} model ({model_keys}). Result will reflect only this model's valid output.")

    # Ensure required identifier columns exist
    if not all(col in df_merged.columns for col in ['query_id', 'product_id']):
         print("ERROR: Merged DataFrame missing 'query_id' or 'product_id'. Cannot proceed.")
         return None


    print(f"Applying majority vote using models: {model_keys}...")
    try:
        # Apply the voting function row-wise (axis=1)
        # This returns a DataFrame where each row contains the Series from get_majority_vote
        vote_results = df_merged.apply(
            lambda row: get_majority_vote(row, model_keys),
            axis=1
        )
        print("Voting function applied successfully.")

        # --- Combine vote results with identifiers ---
        # Use reset_index to ensure alignment if df_merged index is not standard
        df_voted_with_ids = pd.concat(
            [df_merged[['query_id', 'product_id']].reset_index(drop=True), vote_results],
            axis=1
        )

        # --- Merge back the original query text for context (Optional) ---
        df_final_output = df_voted_with_ids # Start with voted results
        if isinstance(original_df, pd.DataFrame) and 'query' in original_df.columns:
             print("Merging back original query text...")
             try:
                 # Ensure keys in original_df are suitable for merge
                 query_context_df = original_df[['query_id', 'product_id', 'query']].drop_duplicates()
                 df_final_output = pd.merge(
                    df_voted_with_ids,
                    query_context_df,
                    on=['query_id', 'product_id'],
                    how='left' # Keep all voted rows, add query where match found
                 )
                 # Check if merge added NaNs unexpectedly
                 if df_final_output['query'].isnull().sum() > df_voted_with_ids.shape[0] - query_context_df.shape[0] + 5: # Heuristic check
                      warnings.warn("Query merge resulted in more nulls than expected. Check keys.", UserWarning)

             except Exception as merge_e:
                 warnings.warn(f"Could not merge back query text: {merge_e}. Proceeding without it.")
                 df_final_output = df_voted_with_ids # Fallback
        else:
            print("Original DataFrame with 'query' column not provided or invalid. Skipping query merge.")
            df_final_output['query'] = None # Add query column with None


        # --- Select, Rename, and Order Columns for Final Output ---
        df_final_output = df_final_output.rename(columns={
            'final_accurate': 'accurate_label',      # Rename to match exercise requirement
            'final_reformulated': 'reformulated_query' # Rename to match exercise requirement
        })

        # Define desired columns and order for the final table
        final_cols_order = [
            'query_id', 'product_id', 'query', 'accurate_label',
            'reformulated_query', 'consensus_reasoning', 'voting_errors'
        ]
        # Filter to only columns that actually exist in the DataFrame
        final_cols_order_existing = [col for col in final_cols_order if col in df_final_output.columns]
        df_final_output = df_final_output[final_cols_order_existing]


        print("\n--- Final Output Table After Voting (Preview) ---")
        # Display a sample of the final formatted table
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
             display(df_final_output.head())

        return df_final_output

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during majority voting application or final formatting: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None