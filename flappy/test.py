import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
    print(f"Total MPS devices : {torch.mps.device_count()}")
else:
    print("MPS not available on this device")