# src/utils.py
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
from typing import Optional

def clear_gpu_memory(*args):
    """
    Attempts to clear GPU memory by deleting variables and emptying the CUDA cache.

    Parameters:
        *args: Variable number of objects (e.g., models, tokenizers) to attempt deletion.
    """
    print("Attempting to clear GPU memory...")
    deleted_vars = []
    for i, var in enumerate(args):
        if var is not None:
            # It's tricky to get the variable name reliably here, just report index
            try:
                del var
                deleted_vars.append(f"arg[{i}]")
            except NameError:
                # Variable might have already been deleted or was None
                pass
            except Exception as e:
                warnings.warn(f"Could not delete variable arg[{i}]: {e}")

    if deleted_vars:
        print(f"  - Attempted deletion of variables: {', '.join(deleted_vars)}")

    # Run Python garbage collector forcefully
    collected = gc.collect()
    print(f"  - Garbage collector ran, collected {collected} objects.")

    # Clear PyTorch CUDA cache if GPU is available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("  - PyTorch CUDA cache emptied.")
            # Optional: Verify memory release
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"  - GPU Memory after clear: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
        except Exception as e:
            warnings.warn(f"Could not empty CUDA cache: {e}")
    else:
        print("  - No CUDA device available, only Python garbage collection performed.")


def download_file(url: str, filename: str, data_dir: str):
    """
    Downloads a file from a URL to a specified directory if it doesn't exist.

    Parameters:
        url (str): The URL of the file to download.
        filename (str): The name to save the file as.
        data_dir (str): The directory to save the file in. Creates if not exists.

    Raises:
        requests.exceptions.RequestException: If download fails.
        OSError: If directory creation fails.
    """
    try:
        os.makedirs(data_dir, exist_ok=True) # Ensure directory exists
    except OSError as e:
        print(f"Error creating directory {data_dir}: {e}")
        raise

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url}...")
        try:
            # Use a session object for potential connection pooling
            with requests.Session() as session:
                response = session.get(url, stream=True, timeout=60) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192 * 16): # Increased chunk size
                        f.write(chunk)
            print(f"Successfully downloaded {filename} to {data_dir}")
        except requests.exceptions.RequestException as e:
            print(f"ERROR downloading {filename}: {e}")
            # Optionally remove partial file if download failed
            if os.path.exists(filepath):
                 try:
                     os.remove(filepath)
                     print(f"Removed partially downloaded file: {filepath}")
                 except OSError as rm_e:
                     print(f"Could not remove partial file {filepath}: {rm_e}")
            raise # Re-raise the exception to halt execution
        except Exception as e: # Catch other potential errors like disk full
             print(f"ERROR during download or writing of {filename}: {e}")
             if os.path.exists(filepath):
                 try:
                     os.remove(filepath)
                 except OSError: pass
             raise
    else:
        print(f"{filename} already exists in {data_dir}. Skipping download.")


def print_system_info():
    """Prints relevant system, GPU, and library version information."""
    print("\n" + "="*30 + " System & Library Information " + "="*30)
    print(f"Python Version: {sys.version}")
    libs = ['pandas', 'numpy', 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'pyarrow', 'sentencepiece', 'requests', 'tqdm']
    for lib in libs:
        try:
            imported_lib = __import__(lib)
            version = getattr(imported_lib, '__version__', 'N/A')
            print(f"{lib.capitalize()} Version: {version}")
        except ImportError:
            print(f"{lib.capitalize()} not found.")
        except Exception as e:
            print(f"Could not get version for {lib}: {e}")


    if torch.cuda.is_available():
        print(f"\nPyTorch CUDA available: True")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_torch_version = torch.version.cuda
            print(f"GPU Detected by PyTorch: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            print(f"PyTorch linked CUDA Version: {cuda_torch_version}")
        except Exception as e:
             print(f"Could not get Torch CUDA details: {e}")

        # Try running nvidia-smi for driver and system CUDA version
        try:
            process = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
                                     capture_output=True, text=True, check=True)
            driver_version, _, _ = process.stdout.strip().split(',') # Get driver version
            print(f"NVIDIA Driver Version: {driver_version.strip()}")
            # Note: System CUDA version (from nvcc) might differ from Torch's colab verion
            # Running nvcc might not always work in all environments.
            # nvcc_process = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            # if nvcc_process.returncode == 0:
            #     print("nvcc output:\n", nvcc_process.stdout)
            # else:
            #     print("nvcc command not found or failed.")

            # Display full nvidia-smi output if needed
            # smi_process = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            # print("\nnvidia-smi output:")
            # print(smi_process.stdout)

        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            print(f"\nCould not run nvidia-smi or parse its output: {e}")
    else:
        print("\nPyTorch CUDA available: False. Inference will run on CPU.")

    print("="*80)