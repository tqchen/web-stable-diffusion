# Metal Stable Diffusion

### Build

**Using database for Metal:**
```shell
python3 build.py --target apple/m2-gpu --log-dir log_db/metal_0225 --const-params-dir const_params --output module.so
```

**Using database for WebGPU (satisfying the 256-thread constraint):**
```shell
python3 build.py --target apple/m2-gpu --log-dir log_db/webgpu_0228 --const-params-dir const_params --output module.so
```

`--const-params-dir const_params` means dumping the computed constant parameters to directory `const_params`.

### Deploy

```shell
python deploy.py --const-params-dir const_params --lib-path module.so
```
