
# MICA 

MICA output examples including .ply mesh, rendered image, and .npy FLAME parameters.



# Source
Modified repo of Gaussian Splatting and 3DSSL demo

# Installation
Either use exported environment or environment_modifed (environment_modifed might be missing one or two pip dependencies)
```shell
   conda env create -f environment_exported.yml
```

Run the steps as in the 3DSSL demo
```bash
    conda activate gaussian_splatting
    conda env config vars set CUDA_HOME=$CONDA_PREFIX
    conda activate base
    conda activate gaussian_splatting
```

Execute the demo via
```bash
    BW_IMPLEMENTATION=1 python run_gaussian_splatting.py
```
# 3DSSL-SS24_Generalizable3DHeadReconstruction
