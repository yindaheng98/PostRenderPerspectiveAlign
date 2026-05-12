# PRPA: Post-Render Perspective Align

This repo is the **Python implementation of PRPA (Post-Render Perspective Align)**, a post-render view alignment algorithm for rendered 3D Gaussian Splatting images. Given a target camera with its rendered depth map and a reference camera with its colour image, PRPA warps the reference image into the target view, detects occlusion conflicts, and fills warping artifacts with an error-erosion pass.

PRPA is introduced in [**(SIGGRAPH 2026) CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting**](https://arxiv.org/abs/2605.09279). The CAGS codec implementation is maintained separately at [`ColorAdaptiveGaussianSplatting`](https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting).

## Features

* [x] Standard Python package with `pip install` support
* [x] Depth-map reconstruction from rendered target views
* [x] Perspective reprojection from target view to reference view
* [x] Reference-image warping with PyTorch `grid_sample` support
* [x] Occluded / occluding pixel masks from projected-depth competition
* [x] Error-erosion filling for occluded regions with configurable kernels and dilation
* [x] Multiple execution backends: PyTorch, Taichi CUDA, and compiled CUDA extension
* [x] Helper renderer for generating colour, depth, and camera files from Gaussian Splatting outputs

## Install

### Prerequisites

* Python >= 3
* [PyTorch](https://pytorch.org/) with CUDA support
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and a C++ compiler if building the compiled CUDA extension
* [Taichi](https://www.taichi-lang.org/) if using the `taichi` backend
* `opencv-python` and `numpy` for reading images, depth maps, and camera files
* [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) if using `render.py` to generate test data

Install common runtime dependencies first:

```shell
pip install torch torchvision
pip install opencv-python numpy taichi tqdm matplotlib
```

### PyPI Install

```shell
pip install --upgrade PostRenderPerspectiveAlign
```

or build the latest version from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/PostRenderPerspectiveAlign.git@master --no-build-isolation
```

### Development Install

```shell
git clone https://github.com/yindaheng98/PostRenderPerspectiveAlign.git
cd PostRenderPerspectiveAlign
pip install torch torchvision opencv-python numpy taichi tqdm matplotlib
pip install --upgrade --no-build-isolation -e .
```

For rendering Gaussian Splatting outputs into PRPA test data, install the packaged Gaussian Splatting dependency:

```shell
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

The `cuda` backend uses the CUDA extension declared in `setup.py`. If extension compilation fails, use the `torch` or `taichi` backend.

## Data Layout

PRPA command-line tools address each view by a file prefix. The target view needs a camera JSON file and a depth `.npz` file; the reference view needs a camera JSON file and a colour image:

```text
testdata/
|-- 00000.camera.json
|-- 00000.png
|-- 00001.camera.json
`-- 00001.depth.npz
```

Camera JSON files use the Gaussian Splatting camera fields consumed by `prpa.data.fromJSON`, including `height`, `width`, `rotation`, `position`, `fx`, and `fy`. Depth files store a compressed NumPy array named `depth`, shaped as `1 x H x W`.

## Command-Line Usage

### Align a Reference View to a Target View

```shell
python -m prpa \
    --local testdata/00001 \
    --reference testdata/00000 \
    --warped output/00001_from_00000 \
    --backend torch \
    --kernel-size 16
```

The command reads:

* `testdata/00001.camera.json` and `testdata/00001.depth.npz` as the target view
* `testdata/00000.camera.json` and `testdata/00000.png` as the reference view

It writes:

* `output/00001_from_00000.no_error_erosion.png`: raw reprojected result before error erosion
* `output/00001_from_00000.png`: final PRPA-aligned image

### Choose a Backend

```shell
python -m prpa \
    --local testdata/00001 \
    --reference testdata/00000 \
    --warped output/00001_from_00000 \
    --backend taichi
```

Available backends:

* `torch`: reference implementation using PyTorch tensor operations and `grid_sample`
* `taichi`: fused Taichi kernels for reprojection, occlusion detection, and erosion on CUDA
* `cuda`: compiled CUDA extension exposed through the Taichi kernel wrapper

The Taichi and CUDA query kernels use nearest-neighbour colour sampling. Use the `torch` backend when bilinear `grid_sample` sampling is required.

### Tune Error Erosion

```shell
python -m prpa \
    --local testdata/00001 \
    --reference testdata/00000 \
    --warped output/00001_from_00000 \
    --kernel-size 16 \
    --occluded-dilation-size 1 \
    --occlude-dilation-size 1 \
    --max-iterations 64
```

`kernel-size` controls the colour-averaging window for filling occluded regions. The dilation arguments expand the occluded / occluding masks before source pixels are selected. `max-iterations` can be used to cap the erosion loop.

### Render Your Own Test Data

```shell
python render.py \
    -s data/flame_salmon_1/frame1 \
    -d output/flame_salmon_1/frame1 \
    -t testdata \
    -i 10000 \
    --mode base \
    --device cuda
```

`render.py` loads `output/flame_salmon_1/frame1/point_cloud/iteration_10000/point_cloud.ply`, renders the cameras from the source scene, and writes `.png`, `.depth.npz`, `.depth.png`, and `.camera.json` files under `testdata/`.

## API Usage

### Run PRPA on Files

```python
import torch
from prpa import PRPA
from prpa.data import read_camera_color, read_camera_depth

with torch.device("cuda"):
    target = read_camera_depth("testdata/00001.camera.json", "testdata/00001.depth.npz")
    reference = read_camera_color("testdata/00000.camera.json", "testdata/00000.png")
    warped = PRPA(target, reference, bordermode="grid_sample", kernel_size=16)
```

`target.depth` is an `H x W` depth tensor. `reference.color` is an `H x W x C` image tensor as loaded by OpenCV.

### Select an Accelerated Backend

```python
import taichi as ti
from prpa.prpa import set_backend

set_backend("taichi", arch=ti.cuda)
```

or:

```python
from prpa.prpa import set_backend

set_backend("cuda")
```

Call `set_backend` before invoking `PRPA`. The default backend is `torch`.

### Use Lower-Level Operators

```python
from prpa import reconstruction, projection, query, warp

xyz = reconstruction(target.K, target.R, target.T, target.depth)
uv, depth = projection(reference.K, reference.R, reference.T, xyz)
warped, mask_occluded, mask_occlude = query(target, reference, reference.color)
warped = warp(warped, mask_occluded, mask_occlude, kernel_size=16)
```

These operators expose the same pipeline stages used by `PRPA` for experiments and custom post-processing.

## Design: Post-Render Perspective Align

PRPA separates perspective alignment into four stages:

```text
Target depth + target camera -> 3D reconstruction
3D points + reference camera -> reference-view projection
Reference colour + projected pixels -> warped colour + occlusion masks
Warped colour + masks -> error-eroded final image
```

### Reprojection

`reconstruction` lifts target-view pixels into 3D using the target camera intrinsics, pose, and depth map. `projection` maps those reconstructed 3D points into the reference camera, producing reference-image coordinates and projected depth.

### Occlusion Detection

`is_occlusion` compares projected depths that land on the same reference pixel. Pixels whose depth is farther than the nearest competing depth are marked as occluded, while the nearer pixels that hide them are marked as occluding.

### Error Erosion

`warp` first closes the occlusion masks morphologically, then repeatedly applies `error_erosion` to fill occluded pixels from nearby valid colour samples. This removes holes and edge artifacts introduced by perspective warping.

### Backends

The PyTorch backend is the readable reference path. The Taichi and CUDA backends fuse the expensive reprojection, occlusion, and erosion kernels for faster CUDA execution. `benchmark.py` compares Taichi and CUDA performance on local test data.

## Testing and Benchmarking

```shell
python test_cuda_vs_taichi.py
python benchmark.py
```

`test_cuda_vs_taichi.py` checks query and erosion consistency between the Taichi and CUDA kernels. `benchmark.py` reports average PRPA runtime for both accelerated backends.

## Acknowledgement

This repo is developed based on [PyTorch](https://pytorch.org/), [Taichi](https://www.taichi-lang.org/), and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Many thanks to the authors for open-sourcing their codebases.

## License

This project is released under the Apache-2.0 License.
