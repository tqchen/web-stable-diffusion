class StableDiffusionPipeline {
  constructor(tvm) {
    this.tvm = tvm;
    this.image_width = 512;
    this.image_height = 512;
    // hold output image
    this.outputImage = tvm.detachFromCurrentScope(
      tvm.empty([512, 512, 3], "float32", tvm.cpu())
    );
    this.outputTemp = tvm.detachFromCurrentScope(
      tvm.empty([512, 512], "uint32", tvm.cpu())
    );
    this.device = tvm.webgpu();
    tvm.bindCanvas(document.getElementById("canvas"));
  }

  dispose() {
    this.outputImage.dispose();
  }

  async showImage(data) {
    this.outputImage.copyFrom(data);
    await this.device.sync();
    this.tvm.beginScope();
    const imgData = this.tvm.ctx.floatNDArrayToCanvasBuffer(this.outputImage);
    this.outputTemp.copyFromRawBytes(new Uint8Array(imgData));
    const gpuArr = this.tvm.empty([512, 512], "uint32", this.tvm.webgpu());
    gpuArr.copyFrom(this.outputTemp);
    this.tvm.showImage(gpuArr);
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
  tvm.registerAsyncServerFunc("clearImage", async (data) => {
    await handler.clearImage();
  });
}

tvmjsGlobalEnv.onServerLoad = onServerLoad;
