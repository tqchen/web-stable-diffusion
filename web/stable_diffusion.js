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
    this.imageToRGBA = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("image_to_rgba")
    );
  }

  dispose() {
    // note: tvm instance is not owned by this class
    this.outputImage.dispose();
    this.imageToRGBA.dispose();
    this.vm.dispose();
  }

  async showImage(data) {
    this.tvm.beginScope();
    this.outputImage.copyFrom(data);
    const rgbaData = this.imageToRGBA(this.outputImage);
    this.tvm.showImage(rgbaData);
    this.tvm.endScope();
  }

  async clearImage() {
    this.tvm.clearCanvas();
  }
};

function onServerLoad(tvm) {
  const handler = new StableDiffusionPipeline(tvm);
  tvm.registerAsyncServerFunc("showImage", async (data) => {
    await handler.showImage(data);
  });
  tvm.registerAsyncServerFunc("clearImage", async () => {
    await handler.clearImage();
  });
}

tvmjsGlobalEnv.onServerLoad = onServerLoad;
