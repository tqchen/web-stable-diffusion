
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
      if (this.ets.length > 3) {
        this.ets.shift().dispose();
      }
      this.ets.push(this.tvm.detachFromCurrentScope(
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
      const findex = counter < 5 ? counter : 4;
      prevLatents = this.schedulerFunc[findex](
        sample,
        this.sampleCoeff[counter],
        this.alphaDiff[counter],
        this.modelOutputDenomCoeff[counter],
        ...this.ets
      );
    }
    return prevLatents;
  }
}

class StableDiffusionPipeline {
  constructor(tvm, tokenizer, schedulerConsts, progressCallback) {
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.progressCallback = progressCallback;
    this.maxTokenLength = 77;

    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("canvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.schedulerConsts = schedulerConsts;
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

  showImage(vaeOutput) {
    this.tvm.beginScope();
    const rgbaData = this.imageToRGBA(vaeOutput);
    this.tvm.showImage(rgbaData);
    this.tvm.endScope();
  }

  /**
   * Tokenize the prompt to TVMNDArray.
   * @param prompt Input prompt
   * @returns The text id NDArray.
   */
  tokenize(prompt) {
    const encoded = this.tokenizer.encode(prompt, true).input_ids;
    const inputIDs = new Int32Array(this.maxTokenLength);

    if (encoded.length < this.maxTokenLength) {
      inputIDs.set(encoded);
      const lastTok = encoded[encoded.length - 1];
      inputIDs.fill(lastTok, encoded.length, inputIDs.length);
    } else {
      inputIDs.set(encoded.slice(0, this.maxTokenLength));
    }
    return this.tvm.empty([1, this.maxTokenLength], "int32", this.device).copyFrom(inputIDs);
  }

  async runVAEStage(data) {
    this.tvm.beginScope();
    const temp = this.tvm.empty(data.shape, data.dtype, this.tvm.webgpu());
    temp.copyFrom(data);
    const image = this.vaeToImage(temp, this.vaeParams);
    this.tvm.showImage(this.imageToRGBA(image));
    this.tvm.endScope();
  }

  async runUNetStage(inputLatents, inputEmbeddings, numSteps, vaeCycle) {
    this.tvm.beginScope();
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.empty(inputLatents.shape, inputLatents.dtype, this.tvm.webgpu())
    );
    let embeddings = this.tvm.detachFromCurrentScope(
      this.tvm.empty(inputEmbeddings.shape, inputEmbeddings.dtype, this.tvm.webgpu())
    );

    latents.copyFrom(inputLatents);
    embeddings.copyFrom(inputEmbeddings);
    scheduler = new TVMPNDMScheduler(this.schedulerConsts, this.tvm, this.device, this.vm);
    this.tvm.endScope();

    let lastSync = undefined;

    for (let counter = 0; counter < numSteps; ++counter) {
      const timestep = scheduler.timestep[counter];
      // recycle noisePred, track latents manually
      const newLatents = this.tvm.withNewScope(() => {
        this.tvm.attachToCurrentScope(latents);
        const noisePred = this.unetLatentsToNoisePred(latents, timestep, embeddings, this.unetParams);
        // maintain new latents
        return this.tvm.detachFromCurrentScope(
          scheduler.step(noisePred, latents, counter)
        );
      });
      latents = newLatents;

      if (lastSync !== undefined) {
        await lastSync;
        console.log("Iter " + counter);
        if (this.progressCallback !== undefined) {
          this.progressCallback(counter, numSteps);
        }
      }
      // async event checker
      lastSync = this.device.sync();

      if ((counter + 1) % vaeCycle == 0 && (counter + 1) != numSteps) {
        this.tvm.withNewScope(() => {
          const image = this.vaeToImage(latents, this.vaeParams);
          this.tvm.showImage(this.imageToRGBA(image));
        });
        await this.device.sync();
      }
    }
    embeddings.dispose();
    this.tvm.withNewScope(() => {
      const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(image));
    });
    latents.dispose();
    scheduler.dispose();
    await this.device.sync();
  }

  async runCLIPStage(inputTextIDs, inputLatents, numSteps, vaeCycle) {
    this.tvm.beginScope();
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.empty(inputLatents.shape, inputLatents.dtype, this.tvm.webgpu())
    );
    let inputIDs = this.tvm.detachFromCurrentScope(
      this.tvm.empty(inputTextIDs.shape, inputTextIDs.dtype, this.tvm.webgpu())
    );

    latents.copyFrom(inputLatents);
    inputIDs.copyFrom(inputTextIDs);
    this.tvm.endScope();
    const embeddings = this.clipToTextEmbeddings(inputIDs, this.clipParams);
    await this.runUNetStage(latents, embeddings, numSteps, vaeCycle);
  }

  async runFullStage(prompt, numSteps, vaeCycle) {
    this.tvm.beginScope();
    const latentShape = [1, 4, 64, 64];
    // use uniform distribution with same variance.
    const scale = Math.sqrt(12) / 2;

    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );

    let inputIDs = this.tvm.detachFromCurrentScope(
      this.tokenize(prompt)
    );
    this.tvm.endScope();
    const embeddings = this.clipToTextEmbeddings(inputIDs, this.clipParams);
    await this.runUNetStage(latents, embeddings, numSteps, vaeCycle);
  }

  clearImage() {
    this.tvm.clearCanvas();
  }
};

async function asyncOnServerLoad(tvm) {
  const schedulerConst = await(await fetch("scheduler_consts.json")).json();
  const tokenizer = await tvmjsGlobalEnv.getTokenizer("openai/clip-vit-large-patch14");

  function progressCallback(counter, Steps) {

  }

  tvm.beginScope();
  const handler = new StableDiffusionPipeline(tvm, tokenizer, schedulerConst, progressCallback);

  tvm.registerAsyncServerFunc("runVAEStage", async (data) => {
    await handler.runVAEStage(data);
  });

  tvm.registerAsyncServerFunc("runUNetStage", async (latents, embeddings, numSteps, vaeCycle) => {
    await handler.runUNetStage(latents, embeddings, numSteps, vaeCycle);
  });

  tvm.registerAsyncServerFunc("runCLIPStage", async (textIDs, latents, numSteps, vaeCycle) => {
    await handler.runCLIPStage(textIDs, latents, numSteps, vaeCycle);
  });

  tvm.registerAsyncServerFunc("runFullStage", async (prompt, numSteps, vaeCycle) => {
    await handler.runFullStage(prompt, numSteps, vaeCycle);
  });

  tvm.registerAsyncServerFunc("clearImage", async () => {
     handler.clearImage();
  });
  tvm.endScope();
}

tvmjsGlobalEnv.asyncOnServerLoad = asyncOnServerLoad;
