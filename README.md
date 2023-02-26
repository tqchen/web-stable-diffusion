# Metal Stable Diffusion

### Build

```shell
python3 build.py --target apple/m2-gpu --log-dir log_db/metal_0225 --output module.so --llvm-output module_llvm.so
```

### Deploy

```shell
python3 deploy.py --lib-path module.so --lib-llvm-path module_llvm.so
```

