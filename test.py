import torch

checkpoint = torch.load("exp/lightning_logs/version_0/checkpoints/epoch=9-step=100.ckpt", map_location="cpu")
print(checkpoint['state_dict'].keys())
