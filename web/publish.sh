#!/bin/bash

rm -rf site/dist
mkdir -p site/dist site/_inlcudes

cp web/stable_diffusion.html site/_includes

cp web/stable_diffusion.js site/dist
cp ../tvm/web/dist/tvmjs_runtime.wasi.js site/dist
cp ../tvm/web/dist/tvmjs.bundle.js site/dist
cp scheduler_consts.json site/dist
cp ../tvm/web/dist/ site/dist
cp ../tvm/web/dist/ site/dist
cp -rf ../tokenizer-wasm/pkg site/dist/tokenizer-wasm
cp build/stable_diffusion.wasm site/dist
cp web/publish.json site/stable-diffusion-config.json

ln -fs ../tvm/web/.ndarray_cache/sd-webgpu-v1-5 site/dist/sd-webgpu-v1-5
