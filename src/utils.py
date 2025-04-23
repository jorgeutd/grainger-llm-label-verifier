\# src/utils.py
"""
Utility functions for the LLM Product Query Verification project.
Includes GPU memory management, file downloading, and system info printing.
"""
import torch
import gc
import os
import requests
import subprocess
import sys
import warnings
import time # Added for sleep in download retry
from typing import Optional

# --- GPU Memory Management ---
def clear_gpu_memory():
    """
    Attempts to clear GPU memory by running garbage collection and emptying the CUDA cache.
    Note: Ensure large objects (models, tensors) are deleted (del variable_name)
    in the calling scope *before* calling this function for best results.
    """
    print("Attempting to clear GPU memory (GC Collect + Empty Cache)...")
    # Run Python garbage collector multiple times potentially
    collected_1 = gc.collect()
    collected_2 = gc.collect()
    print(f"  - Garbage collector ran, collected {collected_1}, {collected_2} objects.")

    # Clear PyTorch CUDA cache if GPU is available
    if torch.cuda.is_available():
        try:
            # Log memory *before* clearing cache
            allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_before = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"  - GPU Memory BEFORE clear: Allocated={allocated_before:.2f} GB, Reserved={reserved_before:.2f} GB")

            torch.cuda.empty_cache()
            print("  - PyTorch CUDA cache emptied.")

            # Log memory *after* clearing cache
            allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"  - GPU Memory AFTER clear:  Allocated={allocated_after:.2f} GB, Reserved={reserved_after:.2f} GB")
        except Exception as e:
            warnings.warn(f"Could not empty CUDA cache: {e}")
    else:
        print("  - No CUDA device available, only Python garbage collection performed.")

# --- File Downloading ---
def download_file(url: str, filename: str, data_dir: str):
    """
    Downloads a file from a URL to a specified directory if it doesn't exist.
    Includes basic retry logic and progress bar (if tqdm installed).

    Parameters:
        url (str): The URL of the file to download.
        filename (str): The name to save the file as (basename).
        data_dir (str): The directory to save the file in. Creates if not exists.

    Raises:
        requests.exceptions.RequestException: If download fails after retries.
        OSError: If directory creation fails.
        Exception: For other unexpected errors during download/write.
    """
    try:
        os.makedirs(data_dir, exist_ok=True) # Ensure directory exists
    except OSError as e:
        print(f"ERROR creating directory {data_dir}: {e}")
        raise

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url} to {data_dir}...")
        retries = 3
        for attempt in range(retries):
            try:
                # Use a session object for potential connection pooling
                with requests.Session() as session:
                    # Stream=True is important for large files
                    response = session.get(url, stream=True, timeout=90) # Increased timeout
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192 * 16 # Increased chunk size for potentially faster downloads

                    # Use tqdm for progress bar if available
                    progress_bar = None
                    try:
                        from tqdm.auto import tqdm
                        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"  Downloading {filename}", leave=False)
                    except ImportError:
                        # tqdm not found, proceed without progress bar
                        print("  tqdm not found, download progress bar disabled.")


                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk: # filter out keep-alive new chunks
                                if progress_bar: progress_bar.update(len(chunk))
                                f.write(chunk)

                    if progress_bar: progress_bar.close()

                    # Verify file size if possible
                    actual_size = os.path.getsize(filepath)
                    if total_size != 0 and actual_size != total_size:
                         warnings.warn(f"Download size mismatch for {filename}. Expected {total_size}, got {actual_size}. File might be corrupt.")
                    elif total_size == 0 and actual_size == 0:
                         warnings.warn(f"Downloaded file {filename} has zero size. Check URL or source.")


                print(f"Successfully downloaded {filename}")
                return # Exit loop on success

            except requests.exceptions.RequestException as e:
                print(f"ERROR downloading {filename} (Attempt {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1: # If last attempt failed
                     # Optionally remove partial file
                     if os.path.exists(filepath):
                         try:
                             os.remove(filepath)
                             print(f"Removed partially downloaded file: {filepath}")
                         except OSError as rm_e:
                             print(f"Could not remove partial file {filepath}: {rm_e}")
                     raise # Re-raise the exception to halt execution
                print(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt) # Exponential backoff
            except Exception as e: # Catch other potential errors like disk full
                 print(f"ERROR during download or writing of {filename}: {e}")
                 if os.path.exists(filepath):
                     try:
                         os.remove(filepath)
                     except OSError: pass
                 raise # Re-raise unexpected errors

    else: # File already exists
        print(f"{filename} already exists in {data_dir}. Skipping download.")
        # Optional: Add size check for existing files here if desired


# --- System Information ---
def print_system_info():
    """Prints relevant system, GPU, and library version information."""
    print("\n" + "="*30 + " System & Library Information " + "="*30)
    print(f"Python Version: {sys.version}")
    # Expanded list to include potential agent libraries if used elsewhere
    libs = ['pandas', 'numpy', 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'pyarrow', 'sentencepiece', 'requests', 'tqdm', 'langgraph', 'langchain_core', 'langchain_anthropic']
    print("\nLibrary Versions:")
    for lib in libs:
        try:
            # Use importlib to avoid polluting namespace and handle optional libs
            import importlib
            # Handle hyphens in package names for import
            module_name = lib.replace('-', '_')
            imported_lib = importlib.import_module(module_name)
            version = getattr(imported_lib, '__version__', 'N/A')
            print(f"  {lib:<25}: {version}")
        except ImportError:
            # Only print "Not found" if it's expected to be optional
            if lib in ['langgraph', 'langchain_core', 'langchain_anthropic', 'langchain_openai']:
                 print(f"  {lib:<25}: Not found (Optional - for agentic approach)")
            else:
                 # Core libs missing indicates setup issue
                 print(f"  {lib:<25}: Not found (ERROR - Check requirements.txt)")
        except Exception as e:
            print(f"  Could not get version for {lib}: {e}")

    # --- GPU Info ---
    if torch.cuda.is_available():
        print(f"\nPyTorch CUDA available: True")
        try:
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                 gpu_name = torch.cuda.get_device_name(i)
                 gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                 print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            cuda_torch_version = torch.version.cuda
            print(f"  PyTorch linked CUDA Version: {cuda_torch_version}")
            cudnn_version = torch.backends.cudnn.version()
            print(f"  cuDNN Version: {cudnn_version}")
            print(f"  BF16 Supported: {torch.cuda.is_bf16_supported()}")
        except Exception as e:
             print(f"  Could not get Torch CUDA details: {e}")

        # Try running nvidia-smi for driver version
        try:
            # Use shell=True for broader compatibility, but be mindful of security if command were dynamic
            process = subprocess.run('nvidia-smi --query-gpu=driver_version --format=csv,noheader',
                                     capture_output=True, text=True, check=True, timeout=10, shell=True)
            driver_version = process.stdout.strip().split('\n')[0] # Get first line if multiple GPUs
            print(f"  NVIDIA Driver Version: {driver_version.strip()}")
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
            print(f"  Could not run nvidia-smi or parse its output: {e}")
    else:
        print("\nPyTorch CUDA available: False. Operations will run on CPU.")

    print("="*80)