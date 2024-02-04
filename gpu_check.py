import torch

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Get the name of the current GPU
    current_gpu_name = torch.cuda.get_device_name(0)  # Assuming you have at least one GPU

    # Get GPU properties
    gpu_properties = torch.cuda.get_device_properties(0)

    print(f"Number of available GPUs: {num_gpus}")
    print(f"Current GPU name: {current_gpu_name}")
    print(f"GPU properties:\n{gpu_properties}")
else:
    print("CUDA is not available.")
