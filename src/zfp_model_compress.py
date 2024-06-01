import torch
import zfpy
import numpy as np

#May or may not work properly :)

def zfp_compress_model(model, tolerance=1e-3, save_path="models/resnet18_weights_compressed.pth"):
    """
    Compress the weights of a model and save the compressed ckpt file in save_path
    """

    compressed_state_dict = {}
    for key, tensor in model.state_dict().items():
        np_vals = tensor.numpy()
        if len(np_vals.shape) == 0: #Compress nothing
            compressed_state_dict[key] = np_vals 
        else:
            compressed = zfpy.compress_numpy(np_vals, tolerance=tolerance)
            compressed_state_dict[key] = compressed

            #Ensure Compression Actually Happens
            decompressed_array = zfpy.decompress_numpy(compressed)
            # np.testing.assert_allclose(np_vals, decompressed_array, atol=1e-2)
    torch.save(compressed_state_dict, save_path)


def zfp_decompress_model(compressed_state_dict):
    """
    compressed_state_dict: Dict[str, bytestream (NOT TENSOR)]

    Decompress loaded model from compressed_state_dict
    """

    # print(type(compressed_state_dict))
    decompressed_state_dict = {}
    for key in compressed_state_dict:
        compress = compressed_state_dict[key]
        if not np.isscalar(compress): #If nothing was compressed
            decompressed_state_dict[key] = torch.tensor(0)
        else:                         #Decompress bytes back to be used in model
            decompress = zfpy.decompress_numpy(compress)
            decompressed_state_dict[key] = torch.from_numpy(decompress)
    return decompressed_state_dict