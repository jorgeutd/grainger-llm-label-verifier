# src/llm_interaction.py
"""
Functions for loading LLMs, formatting prompts, running inference,
and parsing JSON results for the LLM Product Query Verification project.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import warnings
import time
import gc
from typing import Tuple, Optional, Dict, Any

# Use relative imports within the src package
from . import config
from . import utils

# --- Model Loading ---

def get_quantization_config(quant_mode: Optional[str]) -> Optional[BitsAndBytesConfig]:
    """
    Creates a BitsAndBytesConfig object based on the specified quantization mode.

    Parameters:
        quant_mode (Optional[str]): Quantization mode ('4bit', '8bit', or None).

    Returns:
        Optional[BitsAndBytesConfig]: Configuration object or None.
    """
    if quant_mode == "4bit":
        if not torch.cuda.is_available():
             warnings.warn("4-bit quantization requested, but CUDA is not available. LLM loading might fail or use CPU.", UserWarning)
             return None # Cannot quantize without CUDA

        print("Applying 4-bit quantization (BitsAndBytesConfig).")
        # Check for bfloat16 support for compute dtype - provides better performance
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"  Using bnb_4bit_compute_dtype: {compute_dtype}")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",          # NF4 is often recommended
            bnb_4bit_use_double_quant=True,     # Can improve quality slightly
        )
    elif quant_mode == "8bit":
        if not torch.cuda.is_available():
             warnings.warn("8-bit quantization requested, but CUDA is not available. LLM loading might fail or use CPU.", UserWarning)
             return None

        print("Applying 8-bit quantization (BitsAndBytesConfig).")
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode is None or str(quant_mode).lower() in ['none', '']:
         print("No quantization applied.")
         return None
    else:
        warnings.warn(f"Unsupported quantization mode '{quant_mode}' specified in config. No quantization will be applied.", UserWarning)
        return None

def load_llm(model_key: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Loads a specified LLM and tokenizer using details from config.LLM_CONFIG.
    Handles quantization, device mapping, potential errors, and memory footprint reporting.

    Parameters:
        model_key (str): The short key for the model (e.g., 'qwen_14b') defined
                         as a key in config.LLM_CONFIG.

    Returns:
        tuple: (model, tokenizer) if successful, otherwise (None, None).
    """
    if model_key not in config.LLM_CONFIG:
        print(f"ERROR: Model key '{model_key}' not found in config.LLM_CONFIG.")
        return None, None

    model_conf = config.LLM_CONFIG[model_key]
    model_id = model_conf.get('model_id')
    quant_mode = model_conf.get('quantization') # Get quantization mode from config
    trust_remote = model_conf.get('trust_remote', False) # Get trust_remote flag

    if not model_id:
        print(f"ERROR: 'model_id' not specified for model key '{model_key}' in config.")
        return None, None

    print(f"\nAttempting to load model: {model_id} (Key: {model_key})...")
    print(f"  Quantization Mode: {quant_mode if quant_mode else 'None'}")
    print(f"  Trust Remote Code: {trust_remote}")

    model = None
    tokenizer = None
    quantization_cfg = get_quantization_config(quant_mode)

    # Determine appropriate torch_dtype for loading (especially if not quantizing)
    if quantization_cfg is None and torch.cuda.is_available():
        # Use BF16 if supported and not quantizing, otherwise FP16
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"  Loading model with explicit dtype: {model_dtype}")
    elif not torch.cuda.is_available():
         model_dtype = torch.float32 # Use FP32 for CPU
         print("  Loading model on CPU with dtype: torch.float32")
    else:
        # Let BitsAndBytes handle dtype selection during quantization
        model_dtype = None # Important: Don't specify dtype if using BNB quantization config
        print("  Model dtype will be managed by BitsAndBytes during quantization.")

    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
        print("  Tokenizer loaded successfully.")

        # 2. Load Model
        model_load_kwargs = {
            "device_map": "auto", # Handles multi-GPU or CPU fallback
            "quantization_config": quantization_cfg,
            "trust_remote_code": trust_remote,
            "low_cpu_mem_usage": True if quantization_cfg and torch.cuda.is_available() else False # Helps with large models during quantized load
        }
        # Only specify torch_dtype if not using quantization OR if loading on CPU
        if model_dtype is not None:
             model_load_kwargs["torch_dtype"] = model_dtype

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_load_kwargs
        )
        print(f"Successfully loaded model {model_id} onto device: {model.device}") # Confirm device

        # 3. Report Memory Footprint (Optional)
        try:
             # Note: get_memory_footprint might overestimate slightly with quantization
             footprint_bytes = model.get_memory_footprint()
             print(f"  Estimated Model Memory Footprint: {footprint_bytes / (1024**3):.2f} GB")
        except AttributeError:
             print("  Model object does not have get_memory_footprint method.")
        except Exception as e:
             print(f"  Could not get memory footprint: {e}")

        return model, tokenizer

    except torch.cuda.OutOfMemoryError as oom_err:
        error_msg = f"ERROR: CUDA Out of Memory loading {model_id}. {oom_err}"
        if quantization_cfg is None:
             error_msg += " Consider enabling '4bit' quantization in config.py."
        else:
             error_msg += f" Even with {quant_mode} quantization, memory is insufficient. Ensure sufficient VRAM."
        print(error_msg)
        utils.clear_gpu_memory(model, tokenizer) # Clean up aggressively
        return None, None
    except ImportError as imp_err:
         print(f"ERROR: Missing library required for {model_id}. {imp_err}. Check requirements.txt.")
         return None, None
    except Exception as e:
        print(f"ERROR: Failed to load model or tokenizer {model_id}. Error Type: {type(e).__name__}, Message: {e}")
        utils.clear_gpu_memory(model, tokenizer) # Clean up aggressively
        return None, None


# --- Inference & Parsing ---

def format_prompt_chat(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Formats prompts using the tokenizer's chat template for instruct/chat models.

    Parameters:
        tokenizer (AutoTokenizer): The loaded tokenizer for the specific model.
        system_prompt (str): The system prompt content.
        user_prompt (str): The user prompt content.

    Returns:
        Optional[str]: The fully formatted prompt string ready for tokenization,
                       or None if formatting fails.
    """
    messages = []
    if system_prompt: # Only add system prompt if provided
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    else:
         warnings.warn("User prompt is empty, formatting may fail or produce unexpected results.")
         return None # Cannot proceed without user prompt

    try:
        # Use tokenize=False to get the formatted string
        # add_generation_prompt=True is crucial for most instruct/chat models
        # Some models might handle this differently - adjust if needed based on model docs
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    except Exception as e:
        warnings.warn(f"Error applying chat template for tokenizer {tokenizer.name_or_path}: {e}. Check model documentation or template implementation.", UserWarning)
        # Fallback to simple concatenation (less ideal, might not work well)
        # return f"{system_prompt}\n\nUSER: {user_prompt}\nASSISTANT:"
        return None


def parse_llm_output_json(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Robustly parses JSON from the LLM's raw output string.
    Handles markdown code blocks, leading/trailing text, and basic validation.

    Parameters:
        raw_output (str): The raw string output from the LLM.

    Returns:
        tuple: (parsed_json_dict, error_message_or_None)
               - parsed_json_dict (Dict | None): The parsed dictionary if successful and valid.
               - error_message_or_None (str | None): Description of parsing/validation error if any.
    """
    parsed_data = None
    parsing_error_msg = None
    json_str = None

    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, "Raw output is empty or not a string."

    try:
        # 1. Prioritize finding ```json ... ``` block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            # print("DEBUG: Found JSON in markdown block.") # Debugging
        else:
            # 2. Fallback: find first '{' and last '}' greedily
            start_index = raw_output.find('{')
            end_index = raw_output.rfind('}')
            if start_index != -1 and end_index != -1 and end_index >= start_index:
                json_str = raw_output[start_index : end_index + 1].strip()
                # print("DEBUG: Found JSON using find '{...}'.") # Debugging
            else:
                parsing_error_msg = "Could not find JSON structure ({...} or ```json...```)"

        if json_str:
            # Attempt parsing the extracted string
            parsed_data = json.loads(json_str)

            # --- Basic Validation of Parsed JSON ---
            if not isinstance(parsed_data, dict):
                 parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + "Parsed JSON is not a dictionary."
                 parsed_data = None # Invalidate if not a dict
            else:
                 # Check for mandatory keys defined by the prompt
                 required_keys = ["is_exact_match", "reasoning", "reformulated_query"]
                 missing_keys = [k for k in required_keys if k not in parsed_data]
                 if missing_keys:
                     parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + f"Parsed JSON missing expected keys: {missing_keys}"
                     # Decide if this invalidates: For this task, yes, as we need all fields.
                     parsed_data = None
                 else:
                     # Optional: Type check values (already somewhat handled in run_llm_verification)
                     if not isinstance(parsed_data.get('is_exact_match'), bool):
                          parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + "Key 'is_exact_match' is not boolean."
                          parsed_data = None
                     if not isinstance(parsed_data.get('reasoning'), str):
                          parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + "Key 'reasoning' is not a string."
                          parsed_data = None
                     # Reformulated query can be string or null
                     ref_q = parsed_data.get('reformulated_query')
                     if not (isinstance(ref_q, str) or ref_q is None):
                           parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + "Key 'reformulated_query' is not string or null."
                           parsed_data = None


    except json.JSONDecodeError as json_err:
        parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + f"JSON parsing failed: {json_err}. Attempted on: '{str(json_str)[:200]}...'"
        # print(f"DEBUG: JSONDecodeError. Raw: {raw_output[:200]}...") # Debugging
        parsed_data = None # Ensure data is None on decode error
    except Exception as parse_e:
        # Catch other potential errors during regex or slicing
        parsing_error_msg = (parsing_error_msg + "; " if parsing_error_msg else "") + f"Unexpected error during JSON extraction/parsing: {parse_e}"
        # print(f"DEBUG: Unexpected parsing error. Raw: {raw_output[:200]}...") # Debugging
        parsed_data = None # Ensure data is None on other errors

    return parsed_data, parsing_error_msg.strip('; ') if parsing_error_msg else None


def run_llm_verification(
    query: str,
    product_context: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name_key: str # For logging and potentially model-specific adjustments
) -> Dict[str, Any]:
    """
    Runs a single query-product verification check using the LLM.
    Handles prompt formatting, tokenization, inference, decoding, and JSON parsing.

    Parameters:
        query (str): The search query.
        product_context (str): Aggregated product information.
        model (AutoModelForCausalLM): The loaded Hugging Face model.
        tokenizer (AutoTokenizer): The loaded Hugging Face tokenizer.
        model_name_key (str): Short key for the model being used (e.g., 'qwen_14b').

    Returns:
        dict: A dictionary containing the verification results:
              {'is_exact_match': bool | None, 'reasoning': str | None,
               'reformulated_query': str | None, 'raw_output': str,
               'parsing_error': str | None}
              Values are None if errors occurred during processing or parsing.
    """
    start_time = time.time()
    raw_output = ""
    result_dict = {
        'is_exact_match': None, 'reasoning': None, 'reformulated_query': None,
        'raw_output': None, 'parsing_error': None
    }

    # --- Basic Input Validation ---
    if not query or not product_context:
         result_dict['parsing_error'] = "Missing query or product context."
         result_dict['raw_output'] = "Input validation failed."
         return result_dict

    # --- 1. Format the Prompt ---
    user_prompt = config.USER_PROMPT_TEMPLATE.format(query=query, product_context=product_context)
    formatted_input = format_prompt_chat(tokenizer, config.SYSTEM_PROMPT, user_prompt)

    if not formatted_input:
         result_dict['parsing_error'] = "Prompt formatting failed (check chat template)."
         result_dict['raw_output'] = "Prompt formatting failed."
         return result_dict

    # --- 2. Tokenize Input ---
    try:
        # Ensure truncation is handled - adjust max_length based on model limits if known
        # Using a safe default like 4096, common for many models.
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    except Exception as e:
         result_dict['parsing_error'] = f"Tokenization error: {e}"
         result_dict['raw_output'] = f"Tokenization error: {e}"
         return result_dict

    # --- 3. Run Inference ---
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=config.DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id # Crucial for stopping generation
            )

        # Decode only the newly generated tokens, skipping special tokens
        input_length = inputs['input_ids'].shape[1]
        # Handle potential variations in output shape (batch dim)
        if output_sequences.ndim > 1 and output_sequences.shape[0] > 0:
            generated_ids = output_sequences[0, input_length:]
        elif output_sequences.ndim == 1:
             generated_ids = output_sequences[input_length:]
        else:
             raise ValueError("Unexpected output sequence shape from model.generate")

        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        result_dict['raw_output'] = raw_output # Store raw output regardless of parsing success

    except Exception as infer_e:
         # Handle potential runtime errors during generation (e.g., OOM during generation)
         error_message = f"Inference error ({model_name_key}): {type(infer_e).__name__} - {str(infer_e)[:150]}..."
         result_dict['parsing_error'] = error_message
         result_dict['raw_output'] = f"Inference failed: {infer_e}"
         # Do not proceed to parsing if inference failed catastrophically
         return result_dict

    # --- 4. Parse Output & Extract Values ---
    parsed_data, json_parse_error = parse_llm_output_json(raw_output)

    if json_parse_error:
        # Combine potential earlier errors (though unlikely if inference succeeded) with parsing errors
        result_dict['parsing_error'] = f"{result_dict['parsing_error'] or ''}; {json_parse_error}".strip('; ')
    elif parsed_data:
        # Successfully parsed, extract values
        result_dict['is_exact_match'] = parsed_data.get('is_exact_match') # Already type-checked in parse func
        result_dict['reasoning'] = parsed_data.get('reasoning')
        result_dict['reformulated_query'] = parsed_data.get('reformulated_query')

        # --- Rule Enforcement: Reformulated query should be null if match is True ---
        if result_dict['is_exact_match'] is True and result_dict['reformulated_query'] is not None:
             warnings.warn(f"Model {model_name_key} provided reformulation '{result_dict['reformulated_query']}' when is_exact_match was True for query '{query}'. Setting reformulation to None.", UserWarning)
             result_dict['reformulated_query'] = None
    else:
        # This case means parsing didn't find valid JSON but didn't raise an explicit error (should be rare with improved parser)
        result_dict['parsing_error'] = (result_dict['parsing_error'] or "") + "; JSON structure found but failed validation or parsing."

    # Final check: Ensure reformulation is None if the final decision isn't False
    if result_dict['is_exact_match'] is not False:
         result_dict['reformulated_query'] = None

    end_time = time.time()
    # Optional: Log timing per inference call
    # print(f"  - Inference time for row ({model_name_key}): {end_time - start_time:.2f}s")

    # Clean up temporary variables if needed (usually handled by Python's GC)
    del inputs, output_sequences, generated_ids
    gc.collect() # Optionally force garbage collection

    return result_dict