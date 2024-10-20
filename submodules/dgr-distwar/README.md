# DISTWAR atomic reduction on Differential Gaussian Rasterization

Modified 3DGS rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields" with DISTWAR([paper here](https://arxiv.org/abs/2401.05345)) atomic reduction optimizations. We apply two different DISTWAR optimizations to the backward kernel for gaussian rasterization - serialized atomic reduction SW-S and butterfly atomic reduction SW-B. The modified backward kernels are implemented in  [cuda_rasterizer/backward.cu](https://github.com/Accelsnow/diff-gaussian-rasterization-distwar/blob/04253792aad3ad348591332e62108f422cbb3bba/cuda_rasterizer/backward.cu). 

This is a submodule of a modified 3DGS repository. For detailed instruction on how to enable different DISTWAR optimization modes, please refer to the [parent repository](https://github.com/Accelsnow/gaussian-splatting-distwar).

Default configuration: `BW_IMPLEMENTATION=0` &nbsp; `BALANCE_THRESHOLD=8`

Citation for original [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) paper:
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
