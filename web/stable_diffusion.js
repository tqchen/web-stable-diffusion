

class StableDiffusionPipeline {
  constructor(tvm) {
    this.tvm = tvm;
    this.image_width = 512;
    this.image_height = 512;

    // hold output image
    this.outputImage = tvm.empty([4, 512, 512], "floa32", tvm.cpu());
    this.device = tvm.webgpu();
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

async function onServerLoad(tvm) {
  const handler = new StableDiffusionPipeline(tvm);
  tvm.registerAsyncServerFunction("showImage", async (data) => {
    await handler.showImage(data);
  });
}

tvmGlobalEnv.onServerLoad = onServerLoad;