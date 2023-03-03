
/**
 * Wrapper to handle PNDM scheduler
 */
class TVMPNDMScheduler {
  constructor(schedulerConsts, tvm, device, vm) {
    this.timestep = [];
    this.sampleCoeff = [];
    this.alphaDiff = [];
    this.modelOutputDenomCoeff = [];
    this.ets = [];
    this.schedulerFunc = [];
    this.currSample = undefined;
    this.tvm = tvm;
    // prebuild constants
    // principle: always detach for class members
    // to avoid recyling output scope.
    for (let t = 0; t < schedulerConsts.length; ++t) {
      this.timestep.push(
        tvm.detachFromCurrentScope(
          tvm.empty([], "int32", device).copyFrom([schedulerConsts[t][0]])
        )
      );
      this.sampleCoeff.push(
        tvm.detachFromCurrentScope(
          tvm.empty([], "float32", device).copyFrom([schedulerConsts[t][1]])
        )
      );
      this.alphaDiff.push(
        tvm.detachFromCurrentScope(
          tvm.empty([], "float32", device).copyFrom([schedulerConsts[t][2]])
        )
      );
      this.modelOutputDenomCoeff.push(
        tvm.detachFromCurrentScope(
          tvm.empty([], "float32", device).copyFrom([schedulerConsts[t][3]])
        )
      );
    }
    for (let i = 0; i < 5; ++i) {
      this.schedulerFunc.push(
        tvm.detachFromCurrentScope(
          vm.getFunction("scheduler_step_" + i.toString())
        )
      );
    }
  }

  dispose() {
    for (let t = 0; t < this.timestep.length; ++t) {
      this.timestep[t].dispose();
      this.sampleCoeff[t].dispose();
      this.alphaDiff[t].dispose();
      this.modelOutputDenomCoeff[t].dispose();
    }

    for (let i = 0; i < this.schedulerFunc.length; ++i) {
      this.schedulerFunc[i].dispose();
    }

    if (this.currSample) {
      this.currSample.dispose();
    }
    for (let i = 0; i < this.ets.length; ++i) {
      this.ets[i].dispose();
    }
  }

  step(modelOutput, sample, counter) {
    if (counter != 1) {
      // remove the recorded history
      if (self.ets.length > 4) {
        self.ets.shift().dispose();
      }
      self.ets.push(this.tvm.detachFromCurrentScope(
        modelOutput
      ));
    }

    var prevLatents;
    if (counter == 0) {
      this.currSample = this.tvm.detachFromCurrentScope(
        sample
      );
      prevLatents = this.schedulerFunc[0](
        modelOutput,
        sample,
        this.sampleCoeff[counter],
        this.alphaDiff[counter],
        this.modelOutputDenomCoeff[counter]
      );
    } else if (counter == 1) {
      sample = this.tvm.attachToCurrentScope(this.currSample);
      this.currSample = undefined;

      prevLatents = this.schedulerFunc[1](
        modelOutput,
          sample,
          this.sampleCoeff[counter],
          this.alphaDiff[counter],
          this.modelOutputDenomCoeff[counter],
          this.ets[0]
      );
    } else  {
      const reverseETS = this.ets.slice().reverse();
      const findex = counter < 5 ? counter : 4;
      prevLatents = this.schedulerFunc[findex](
        sample,
        this.sampleCoeff[counter],
        this.alphaDiff[counter],
        this.modelOutputDenomCoeff[counter],
        ...reverseETS
      );
    }
    return prevLatents;
  }
}

class StableDiffusionPipeline {
  constructor(tvm, schedulerConsts) {
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
    this.scheduler = new TVMPNDMScheduler(schedulerConsts, tvm, this.vm);
    this.clipToTextEmbeddings = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("clip")
    );
    this.clipParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("clip", 197)
    );
    this.unetLatentsToNoisePred = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unet")
    );
    this.unetParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("unet", 718)
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
    this.scheduler.dispose();
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

async function asyncOnServerLoad(tvm) {
  const schedulerConst = await(await fetch("scheduler_consts.json")).json();

  const handler = new StableDiffusionPipeline(tvm, schedulerConst);
  tvm.registerAsyncServerFunc("showImage", async (data) => {
     handler.showImage(data);
  });
  tvm.registerAsyncServerFunc("runVAEStage", async (data) => {
     handler.runVAEStage(data);
  });
  tvm.registerAsyncServerFunc("clearImage", async () => {
     handler.clearImage();
  });
}

tvmjsGlobalEnv.asyncOnServerLoad = asyncOnServerLoad;
