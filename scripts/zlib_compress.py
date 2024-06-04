import copy
from scripts.zfp_compress import save_pkl, load_pkl
import torch
import zlib
import numpy as np


def zlib_compress(model, name="generic_model_type"):
    zlib_model = copy.deepcopy(model)
    org_size = save_pkl(f'models/{name}.pkl', model.state_dict())
    print(f"Original Size: {org_size}")

    compressed_params, shapes = apply_zlib(model)
    compressed_size = save_pkl(f"models/{name}.pkl", compressed_params)
    print(f"Compressed Size: {compressed_size}")

    load_and_decompress(f"models/{name}.pkl", zlib_model, shapes)

    return zlib_model, org_size, compressed_size

def apply_zlib(model):    
    params = {}
    shapes = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_cpu = param.cpu().detach().numpy()
            params[name] = zlib.compress(param_cpu)  
            shapes[name] = param.size()
    return params, shapes


def load_and_decompress(file: str, model, shapes) -> None:
    device = torch.device('cpu')
    params = load_pkl(file)
    for name, param in model.named_parameters():
        if name in params:
            decompressed_bytes = zlib.decompress(params[name])
            decompressed = np.frombuffer(decompressed_bytes, dtype=np.float32)
            shape = shapes[name]
            param.data = torch.tensor(decompressed.reshape(shape)).to(device)