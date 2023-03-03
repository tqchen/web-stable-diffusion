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
  constructor(tvm, tokenizer, schedulerConsts) {
    this.tvm = tvm;
    this.tokenizer = tokenizer;
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
  /**
   * Run generation pipeline.
   *
   * @param prompt Input prompt
   * @param vaeCycle optionally draw VAE result every cycle iterations.
   */
  async generate(prompt, vaeCycle = -1, progressCallback = undefined) {
    // Principle: beginScope/endScope in synchronized blocks,
    // this helps to recycle intermediate memories
    // detach states that needs to go across async boundaries.
    //--------------------------
    // Stage 0: CLIP
    //--------------------------
    this.tvm.beginScope();
    scheduler = new TVMPNDMScheduler(this.schedulerConsts, this.tvm, this.device, this.vm);
    let inputIDs = this.tvm.detachFromCurrentScope(
      this.tokenize(prompt)
    );
    if (progressCallback === undefined) {
      progressCallback("clip", 0, 1);
    }
    const embeddings = this.tvm.detachFromCurrentScope(
      this.clipToTextEmbeddings(inputIDs, this.clipParams)
    );
    // get latents
    const latentShape = [1, 4, 64, 64];
    // use uniform distribution with same variance as normal(0, 1)
    const scale = Math.sqrt(12) / 2;
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );
    this.tvm.endScope();
    //---------------------------
    // Stage 1: UNet + Scheduler
    //---------------------------
    const numSteps = 50;
    vaeCycle = vaeCycle == -1 ? numSteps: vaeCycle;
    let lastSync = undefined;

    for (let counter = 0; counter < numSteps; ++counter) {
      if (progressCallback !== undefined) {
        progressCallback("unet", counter, numSteps);
      }
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
      // use skip one sync, although likely not as useful.
      if (lastSync !== undefined) {
        await lastSync;
      }
      // async event checker
      lastSync = this.device.sync();

      // Optionally, we can draw intermediate result of VAE.
      if ((counter + 1) % vaeCycle == 0 && (counter + 1) != numSteps) {
        this.tvm.withNewScope(() => {
          const image = this.vaeToImage(latents, this.vaeParams);
          this.tvm.showImage(this.imageToRGBA(image));
        });
        await this.device.sync();
      }
    }
    embeddings.dispose();
    //-----------------------------
    // Stage 2: VAE and draw image
    //-----------------------------
    if (progressCallback !== undefined) {
      progressCallback("vae", 0, 1);
    }
    this.tvm.withNewScope(() => {
      const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(image));
    });
    latents.dispose();
    scheduler.dispose();
    await this.device.sync();
    if (progressCallback !== undefined) {
      progressCallback("vae", 1, 1);
    }
  }

  clearImage() {
    this.tvm.clearCanvas();
  }
};

async function asyncOnServerLoad(tvm) {
  const schedulerConst = await(await fetch("scheduler_consts.json")).json();
  const tokenizer = await tvmjsGlobalEnv.getTokenizer("openai/clip-vit-large-patch14");

  tvm.beginScope();
  const handler = new StableDiffusionPipeline(tvm, tokenizer, schedulerConst);

  tvm.registerAsyncServerFunc("generate", async (prompt, vaeCycle) => {
    const tstart = performance.now();

    function progressCallback(stage, counter, numSteps) {
      const totalSteps = 50 + 2;
      const timeElapsed = (performance.now() - tstart) / 1000;
      let text = "Generating ... at stage " + stage;
      if (stage == "unet") {
        counter += 1;
        text += " step [" + counter + "/" + numSteps + "]"
      }
      if (stage == "vae") {
        counter += 51;
      }
      text += ", " + Math.ceil(timeElapsed) + " secs elapsed";
      document.getElementById("progress-tracker-label").innerHTML = text;
      document.getElementById("progress-tracker-progress").value = (counter / totalSteps) * 100;
    }
    await handler.generate(prompt, vaeCycle, progressCallback);
  });

  tvm.registerAsyncServerFunc("clearImage", async () => {
     handler.clearImage();
  });
  tvm.endScope();
}

tvmjsGlobalEnv.asyncOnServerLoad = asyncOnServerLoad;
