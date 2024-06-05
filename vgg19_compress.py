import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader

import ctypes
import copy

import zfpy

from scripts.zfp_compress import zfp_compress

lib = ctypes.CDLL('./speed_cluster.so')
lib.speed_cluster.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int)
lib.speed_cluster.restype = None

lib.speed_cluster_multithread.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int)
lib.speed_cluster.multithreadrestype = None

lib.speed_cluster_idx_already_clustered.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64)
lib.speed_cluster_idx_already_clustered.restype = None

lib.speed_cluster_idx.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64)
lib.speed_cluster_idx.restype = None

lib.unpack_indices.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64)
lib.unpack_indices.restype = None

def calculate_accuracy(model, data_loader, num_batches=3):
    correct = 0
    total = 0

    with torch.no_grad():
        batches = 0
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batches += 1
            if batches >= num_batches:
                break

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    preprocess = models.VGG19_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()

    val_dataset = datasets.ImageNet(root='../EMLC_plus_LA_and_compression/', split='val', transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    zfp_compress(model, name="generic_model_type", tolerance=1e-3)

    accuracy = calculate_accuracy(model, val_loader, num_batches=40)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')


    clusters = torch.tensor([0.0,
                         0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005
                         -0.2, -0.15, -0.1, -0.05, -0.02, -0.01, -0.05
                        ], dtype=torch.float32)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and name != 'classifier.6.weight' and name != 'features.0.weight':
                # print(name, param.shape)
    
                # to standard normal
                # mu = param.mean()
                # sigma = param.std()
                
                # param.sub_(mu).div_(sigma)
    
                # plt.figure()
                # plt.title(name)
                # plt.hist(param.view(-1)[:10000], bins=20)
    
                # threshold pruning
                param[param.abs() < 0.005] = 0
                
                # cluster
                lib.speed_cluster_multithread(param.data_ptr(), clusters.data_ptr(), param.numel())
    
                # back to standard normal
                # param.mul_(sigma).add_(mu)
    
    # convert to FP16
    # model.half()

    torch.save(model.state_dict(), './vgg19_clustered')

    zfp_model, org_size, compressed_size = zfp_compress(model, name="generic_model_type", tolerance=1e-3)

    with torch.no_grad():
        size = 0
        for name, param in model.named_parameters():
            if 'weight' in name and name != 'classifier.6.weight' and name != 'features.0.weight':
                n = param.numel() // 2
                packed_idx = torch.empty(n + (4 -(n%4))%4, dtype=torch.torch.uint8)
                
                lib.speed_cluster_idx_already_clustered(param.data_ptr(), clusters.data_ptr(), param.numel(), packed_idx.data_ptr())
                
                packed_idx = packed_idx.view(torch.float32)
                compressed_param = zfpy.compress_numpy(packed_idx.numpy(), tolerance=1e-3)
                size += len(compressed_param)
            else:
                compressed_param = zfpy.compress_numpy(param.numpy(), tolerance=1e-3)
                size += len(compressed_param)
        print('zfp compressed size after compression methods with packed indices: ~', size, 'bytes')

    accuracy = calculate_accuracy(model, val_loader, num_batches=40)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')