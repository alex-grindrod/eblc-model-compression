import sys
sys.path.append('.')
import torch
from scripts.eval_resnet18 import eval
import pickle
import os
import zfpy
import copy

def save_pkl(file: str, state_dict: dict) -> float:
    with open(file, 'wb') as f:
        pickle.dump(state_dict, f)
    return os.path.getsize(file)
        
def load_pkl(file: str) -> dict:
    with open(file, 'rb') as f:
        return pickle.load(f)

def apply_zfp(model, tolerance):
    params = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_cpu = param.cpu().detach().numpy()
            params[name] = zfpy.compress_numpy(param_cpu, tolerance=tolerance)
    return params

def load_and_decompress(file: str, model) -> None:
    device = torch.device('cpu')
    params = load_pkl(file)
    for name, param in model.named_parameters():
        if name in params:
            decompressed = zfpy.decompress_numpy(params[name])
            param.data = torch.tensor(decompressed).to(device)

def zfp_compress(model, name, tolerance=1e-3):
    zfp_model = copy.deepcopy(model)

    org_size = save_pkl(f'models/{name}.pkl', model.state_dict())
    print(f"Original Size: {org_size}")

    compressed_params = apply_zfp(model, tolerance)
    compressed_size = save_pkl(f"models/{name}.pkl", compressed_params)
    print(f"Compressed Size: {compressed_size}")

    load_and_decompress(f"models/{name}.pkl", zfp_model)

    return zfp_model, org_size, compressed_size
