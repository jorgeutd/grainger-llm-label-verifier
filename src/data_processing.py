# src/data_processing.py
"""
Functions for data loading, merging, filtering, and text preparation
for the LLM Product Query Verification project.
"""
import pandas as pd
import os
import re
import warnings
import html # For HTML entity decoding
from typing import Optional

# Use relative imports within the src package
from . import config
from . import utils # 

def load_and_merge_data(data_dir: str = config.DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Downloads (if needed), loads, and merges the ESCI examples and products datasets.

    Parameters:
        data_dir (str): Directory containing or to download the parquet files.
                        Defaults to config.DATA_DIR.

    Returns:
        pd.DataFrame or None: The merged DataFrame, or None if loading/download fails.
    """
    print("\n" + "="*30 + " Step: Load and Merge Data " + "="*30)
    # Use filenames directly from config
    examples_filename_only = os.path.basename(config.EXAMPLES_FILENAME)
    products_filename_only = os.path.basename(config.PRODUCTS_FILENAME)
    examples_path = os.path.join(data_dir, examples_filename_only)
    products_path = os.path.join(data_dir, products_filename_only)

    # --- Download using utils ---
    try:
        # Construct full URLs
        examples_url = config.BASE_URL + examples_filename_only
        products_url = config.BASE_URL + products_filename_only
        # Use the utility function for download
        utils.download_file(examples_url, examples_filename_only, data_dir)
        utils.download_file(products_url, products_filename_only, data_dir)
    except Exception as e:
        print(f"FATAL: Halting execution due to download failure: {e}")
        return None

    # --- Load ---
    try:
        print(f"Loading examples from: {examples_path}")
        df_examples = pd.read_parquet(examples_path)
        print(f"Loaded df_examples: Shape = {df_examples.shape}")

        print(f"\nLoading products from: {products_path}")
        df_products = pd.read_parquet(products_path)
        print(f"Loaded df_products: Shape = {df_products.shape}")
    except FileNotFoundError:
        print(f"ERROR: Parquet files not found in {data_dir}. Please check paths and ensure download was successful.")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during file loading: {e}")
        return None

    # --- Merge ---
    print(f"\nMerging df_examples ({df_examples.shape[0]} rows) with df_products ({df_products.shape[0]} rows)...")
    print(f"Merge keys: ['product_id', 'product_locale']")
    try:
        # Ensure merge keys exist in both dataframes before attempting merge
        merge_keys = ['product_id', 'product_locale']
        if not all(k in df_examples.columns for k in merge_keys):
             raise KeyError(f"Merge key(s) missing in df_examples: {[k for k in merge_keys if k not in df_examples.columns]}")
        if not all(k in df_products.columns for k in merge_keys):
             raise KeyError(f"Merge key(s) missing in df_products: {[k for k in merge_keys if k not in df_products.columns]}")

        # Optional: Ensure merge keys are appropriate types if needed (usually handled by pandas)
        # df_examples[merge_keys] = df_examples[merge_keys].astype(str)
        # df_products[merge_keys] = df_products[merge_keys].astype(str)

        df_merged = pd.merge(
            df_examples,
            df_products,
            on=merge_keys,
            how='left', # Keep all examples, match products where available
            validate="many_to_one" # Assumes each product_id/locale is unique in df_products
        )
        print(f"Merged DataFrame shape: {df_merged.shape}")

        # --- Verification ---
        if df_merged.shape[0] != df_examples.shape[0]:
            warnings.warn(f"Row count changed after merge ({df_merged.shape[0]}) from original examples ({df_examples.shape[0]}). Check merge keys and uniqueness in product data.", UserWarning)
        else:
            print("Row count consistent after merge.")

        # Check merge quality (how many products were actually found?)
        # Use a core product column like 'product_title'
        if 'product_title' in df_merged.columns:
             null_titles_after_merge = df_merged['product_title'].isnull().sum()
             perc_null_titles = (null_titles_after_merge / len(df_merged) * 100) if len(df_merged) > 0 else 0
             print(f"Rows with null product_title after merge: {null_titles_after_merge} ({perc_null_titles:.1f}%)")
             # Warn if a high percentage of products weren't found
             if perc_null_titles > 25.0: # Example threshold: 25%
                 warnings.warn("High percentage of null titles after merge - review data quality or merge keys.", UserWarning)
        else:
             warnings.warn("Column 'product_title' not found after merge, cannot check merge quality effectively.")


        return df_merged

    except KeyError as ke:
         print(f"ERROR during merge setup: Missing key column - {ke}")
         return None
    except pd.errors.MergeError as me:
         print(f"ERROR during merge operation: {me}. Check key uniqueness and data types.")
         return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during merging: {e}")
        return None


def filter_data_for_task(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the merged DataFrame for the specific queries, 'E' label, and locale
    defined in the config file. Optionally caches the result.

    Parameters:
        df_merged (pd.DataFrame): The merged dataframe from load_and_merge_data.

    Returns:
        pd.DataFrame: The filtered DataFrame ready for LLM processing.
                      Returns an empty DataFrame if no rows match or input is invalid.
    """
    print("\n" + "="*30 + " Step: Filter Data for Task " + "="*30)

    if not isinstance(df_merged, pd.DataFrame) or df_merged.empty:
        print("Warning: Input DataFrame for filtering is empty or invalid. Returning empty DataFrame.")
        return pd.DataFrame() # Return empty DF

    # Ensure required columns for filtering and identification exist
    required_cols = ['query', 'esci_label', 'product_locale', 'query_id', 'product_id']
    if not all(col in df_merged.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_merged.columns]
        print(f"ERROR: Input DataFrame is missing required columns for filtering: {missing}. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Target Queries: {config.TARGET_QUERIES}")
    print(f"Target ESCI Label: '{config.TARGET_LABEL}'")
    print(f"Target Locale: '{config.TARGET_LOCALE}'")

    try:
        # Apply filters using config values
        condition = (
            df_merged['query'].isin(config.TARGET_QUERIES) &
            (df_merged['esci_label'] == config.TARGET_LABEL) &
            (df_merged['product_locale'] == config.TARGET_LOCALE)
        )
        # Use .loc for boolean indexing, is generally preferred
        filtered_df = df_merged.loc[condition].copy() # Use .copy() to avoid SettingWithCopyWarning

        print(f"Shape of DataFrame AFTER filtering: {filtered_df.shape}")

        if filtered_df.empty:
            warnings.warn("Result: The filtered DataFrame is empty. No rows matched the specified query, label, and locale criteria.", UserWarning)
        else:
            print(f"Successfully filtered down to {filtered_df.shape[0]} rows.")
            # Optional: Cache the filtered data if path is set
            if config.FILTERED_DATA_CACHE:
                try:
                    os.makedirs(config.CACHE_DIR, exist_ok=True)
                    filtered_df.to_parquet(config.FILTERED_DATA_CACHE, index=False)
                    print(f"Filtered data cached to {config.FILTERED_DATA_CACHE}")
                except Exception as e:
                    warnings.warn(f"Could not cache filtered data: {e}")

        # Reset index for cleaner iteration later
        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df

    except KeyError as e:
        # This should be caught by the initial check, but as a safeguard
        print(f"ERROR filtering data: Unexpected missing column - {e}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during filtering: {e}")
        return pd.DataFrame()


def clean_text(text: Optional[str]) -> str:
    """
    Performs basic cleaning on a text string (HTML removal, specific chars, whitespace norm).

    Parameters:
        text (Optional[str]): The input string to clean, or None.

    Returns:
        str: The cleaned string, or an empty string if input is None or invalid.
    """
    if text is None or pd.isna(text): # Handle None and Pandas NA types
        return ""
    if not isinstance(text, str):
        text = str(text) # Attempt conversion if not string (e.g., numbers)

    # 1. Remove HTML tags - replace with space to avoid merging words
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Decode HTML entities (like &)
    try:
        import html
        text = html.unescape(text)
    except ImportError:
        text = text.replace('&', '&').replace('<', '<').replace('>', '>') # Basic fallback
    # 3. Remove specific decorative characters (add more if needed)
    text = text.replace('★', '').replace('【', '').replace('】', '')
    # 4. Normalize whitespace: replace multiple whitespace chars with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def aggregate_product_info_for_llm(row: pd.Series) -> str:
    """
    Aggregates and cleans relevant product information from a DataFrame row
    into a single formatted string suitable for LLM input context.

    Parameters:
        row (pd.Series): A row from the DataFrame, expected to contain standard
                         product detail columns (e.g., 'product_title', 'product_brand').

    Returns:
        str: A formatted string containing the cleaned, aggregated product info,
             or a message indicating incompleteness if no useful info found.
    """
    parts = [] # List to hold formatted parts (e.g., "Title: Cleaned Title")

    # Helper to safely get, clean, and check string data from the row
    def get_cleaned_field(field_name: str) -> Optional[str]:
        """Gets data from row, cleans it, and checks if it's meaningful."""
        raw_data = row.get(field_name) # Safely get data, returns None if column missing
        if pd.isna(raw_data): # Explicit check for Pandas NA types
            return None
        cleaned_data = clean_text(str(raw_data)) # Convert to string and clean
        # Consider common non-informative placeholders as empty after cleaning
        if cleaned_data.lower() in ['nan', 'none', 'n/a', 'null', '']:
            return None # Indicate no meaningful data
        return cleaned_data

    # Process each relevant field defined in a list for maintainability
    fields_to_aggregate = [
        ('product_title', 'Title'),
        ('product_brand', 'Brand'),
        ('product_color', 'Color'),
        ('product_description', 'Description'),
        ('product_bullet_point', 'Bullet Points')
    ]

    for field_name, display_label in fields_to_aggregate:
        cleaned_value = get_cleaned_field(field_name)
        if cleaned_value:
            # Add display label only if value is not empty
            parts.append(f"{display_label}: {cleaned_value}")

    # Combine the collected parts with newlines
    full_info = "\n".join(parts)

    # Return the combined info, or a specific message if nothing useful was found
    return full_info if full_info else "Product information is missing or incomplete."


def apply_text_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the text aggregation function to create the 'llm_product_context' column.

    Handles empty input DataFrame gracefully. Includes progress reporting using tqdm if available.

    Parameters:
        df (pd.DataFrame): The DataFrame to process (usually the filtered one).

    Returns:
        pd.DataFrame: The DataFrame with the added 'llm_product_context' column.
                      Returns the input DataFrame (possibly with an empty column added)
                      if input is invalid or errors occur.
    """
    print("\n" + "="*30 + " Step: Apply Text Aggregation " + "="*30)

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Input DataFrame is empty or invalid, skipping aggregation.")
        if isinstance(df, pd.DataFrame) and 'llm_product_context' not in df.columns:
            # Ensure column exists even if empty for schema consistency
             df['llm_product_context'] = pd.Series(dtype='object')
        return df # Return original empty/invalid df

    # Check if any standard text columns actually exist before proceeding
    required_cols = ['product_title', 'product_brand', 'product_color', 'product_description', 'product_bullet_point']
    if not any(col in df.columns for col in required_cols):
         warnings.warn("None of the standard product text columns found. Aggregation will likely result in default message for all rows.")
         # Proceed, but context will be minimal

    try:
        # Check if tqdm is available for progress bar
        try:
             from tqdm.auto import tqdm
             tqdm.pandas(desc="Aggregating context")
             apply_func = df.progress_apply
        except ImportError:
             print("tqdm not found, applying aggregation without progress bar...")
             apply_func = df.apply

        # Apply the aggregation function row-wise
        df['llm_product_context'] = apply_func(aggregate_product_info_for_llm, axis=1)

        print("\nSuccessfully created 'llm_product_context' column.")

        # Preview the first few contexts for verification
        if not df.empty:
             print("\nPreview of aggregated context (first 3 rows):")
             with pd.option_context('display.max_colwidth', 250): # Adjust width as needed
                 preview_cols = ['query_id', 'product_id', 'query', 'llm_product_context']
                 existing_preview_cols = [col for col in preview_cols if col in df.columns]
                 print(df[existing_preview_cols].head(3))
        return df

    except Exception as e:
        print(f"ERROR applying aggregation function: {e}")
        # Add an empty/null column in case of error to maintain schema consistency
        df['llm_product_context'] = None
        return df # Return the DataFrame