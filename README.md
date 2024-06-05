# eblc-model-compression
## Project Overview

EBLC - Error Bounded Lossy Compression. EBLC refers to any compression algorithm that allows the user to limit/control the aggressiveness of lossy compression.

ZFP is an EBLC algorithm with parameters such as Tolerance and Precision to enforce control over data loss. 
In this Project, we wanted to observe how well ZFP performs in compressing the weights of various ML model architectures like:
- VGG
- ResNet
- Bert
  
We also wanted to compare it against/with similar existing techniques like Quantization, Pruning, and overall lossless compression (zlib). Most importantly, we aim to see if ZFP is a viable compression algorithm for reducing model size while minimizing the loss of performance.

## Project Goals

- Apply varying strengths of ZFP Compression to model weights
- Compare compression performance to Lossless - zlib
- Evaluate and compare against Quantization
- Observe how pruning affects ZFP
- Observe how ZFP performs on popular model architectures

## Works Cited

- Peter Lindstrom. Fixed-Rate Compressed Floating-Point Arrays. IEEE Transactions on Visualization and Computer Graphics, 20(12):2674-2683, December 2014. doi:10.1109/TVCG.2014.2346458.
- James Diffenderfer, Alyson Fox, Jeffrey Hittinger, Geoffrey Sanders, Peter Lindstrom. Error Analysis of ZFP Compression for Floating-Point Data. SIAM Journal on Scientific Computing, 41(3):A1867-A1898, June 2019. doi:10.1137/18M1168832. 

