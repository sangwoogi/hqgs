# HQGS
A minimally modified derivative of HQGS for research use

## Environmental setting
you can pull with 
'''
docker pull sangwoogi/hqgs:latest
'''
or if you want to build it on yourself then,
'''
docker build --build-arg TORCH_FLAVOR=nightly -t hqgs:latest .
docker run --gpus all -it --rm -v ~/workspace:/workspace hqgs:latest
'''

## Running codes
1. preparing datasets of COLMAP types (images&sparse/0)
2. making directory ./sr_seen and add images for training
3. running preparing_train.sh
'''
python gradient.py
python images_concat.py
'''
4. training and rendering

## License

This repository contains derivative works based on:

- **Gaussian Splatting** by Inria and the Max Planck Institute for Informatik (MPII)  
  https://github.com/graphdeco-inria/gaussian-splatting

- **HQGS** by linxin0  
  https://github.com/linxin0/HQGS

This project is distributed under the **Gaussian-Splatting Research License**.  
The software is provided **for research and non-commercial use only**.

Redistribution and modification are permitted **only under the same license**, and
all original copyright, license, and attribution notices must be preserved.

**Commercial use, including use in products, services, or monetized applications,
is strictly prohibited without prior explicit permission from Inria and MPII.**

If you wish to use this software for commercial purposes, you must contact:
stips-sophia.transfert@inria.fr
