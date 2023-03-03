class StableDiffusionPipeline {
  constructor(tvm) {
    this.tvm = tvm;
    this.image_width = 512;
    this.image_height = 512;
    // hold output image
    this.outputImage = tvm.detachFromCurrentScope(
      tvm.empty([1, 512, 512, 3], "float32", tvm.webgpu())
    );
    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("canvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.vaeToImage = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("vae")
    );
    this.vaeParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("vae", 140)
    );
    this.imageToRGBA = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("image_to_rgba")
    );
  }

  dispose() {
    // note: tvm instance is not owned by this class
    this.outputImage.dispose();
    this.imageToRGBA.dispose();
    this.vaeToImage.dispose();
    this.vaeParams.dispose();
    this.vm.dispose();
  }

  showImage(data) {
    this.tvm.beginScope();
    this.outputImage.copyFrom(data);
    const rgbaData = this.imageToRGBA(this.outputImage);
    this.tvm.showImage(rgbaData);
    this.tvm.endScope();
  }

  async runVAEStage(data) {
    this.tvm.beginScope();
    const temp = this.tvm.empty(data.shape, data.dtype, this.tvm.webgpu());
    temp.copyFrom(data);
    const image = this.vaeToImage(temp, this.vaeParams);
    this.tvm.showImage(this.imageToRGBA(image));
    this.tvm.endScope();
  }

  clearImage() {
    this.tvm.clearCanvas();
  }
};

function onServerLoad(tvm) {
  const handler = new StableDiffusionPipeline(tvm);
  tvm.registerAsyncServerFunc("showImage", async (data) => {
     handler.showImage(data);
  });
  tvm.registerAsyncServerFunc("runVAEStage", async (data) => {
    await handler.runVAEStage(data);
  });
  tvm.registerAsyncServerFunc("clearImage", async () => {
     handler.clearImage();
  });
}

tvmjsGlobalEnv.onServerLoad = onServerLoad;
