class StableDiffusionPipeline {
  constructor(tvm) {
    this.tvm = tvm;
    this.image_width = 512;
    this.image_height = 512;
    // hold output image
    this.outputImage = tvm.detachFromCurrentScope(
      tvm.empty([512, 512, 3], "float32", tvm.cpu())
    );
    this.device = tvm.webgpu();
  }

  dispose() {
    this.outputImage.dispose();
  }

  /**
   * Given a float32 array show the image.
   *
   * @param data The input data.
   */
  async showImage(data) {
    this.outputImage.copyFrom(data);
    // sync to make sure data is ready
    await this.device.sync();
    const imgData = this.tvm.floatNDArrayToImageData(this.outputImage);
    const imageCanvas = document.getElementById("canvas");
    const imageCanvasContext = imageCanvas.getContext("2d");
    imageCanvasContext.putImageData(imgData, 0, 0);
  }
};

function onServerLoad(tvm) {
  const handler = new StableDiffusionPipeline(tvm);
  tvm.registerAsyncServerFunc("showImage", async (data) => {
    await handler.showImage(data);
  });
}

tvmjsGlobalEnv.onServerLoad = onServerLoad;
