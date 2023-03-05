#!/bin/bash

rm -rf site/dist
mkdir -p site/dist site/_inlcudes

cp web/stable_diffusion.html site/_includes

cp web/stable_diffusion.js site/dist
cp ../tvm/web/dist/wasm/tvmjs_runtime.wasi.js site/dist
cp ../tvm/web/dist/tvmjs.bundle.js site/dist
cp scheduler_consts.json site/dist
cp -rf ../tokenizers-wasm/pkg site/dist/tokenizers-wasm
cp build/stable_diffusion.wasm site/dist
cp web/publish.json site/stable-diffusion-config.json
ln -s ~/github/tvm/web/.ndarray_cache/sd-webgpu-v1-5 site/dist
cd site
jekyll b
jekyll serve  --skip-initial-build  --baseurl /web-stable-diffusion --port 8888
