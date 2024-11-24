import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Get the ID of the current GPU
print(torch.cuda.get_device_name(0))