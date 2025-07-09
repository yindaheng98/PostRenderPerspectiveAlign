# PRPA: Post-Render Perspective Align

## PyPI Install

Install from pypi:
```sh
pip install prpa
```
or from source:
```sh
pip install git+https://github.com/yindaheng98/PostRenderPerspectiveAlign@master
```

## Render your own test data

```sh
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
python render.py -s data/flame_salmon_1/frame1 -d output/flame_salmon_1/frame1 -t testdata -i 10000
```
