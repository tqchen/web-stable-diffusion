globalConfig = {
  "schedulerConstUrl": "./scheduler_consts.json",
  "wasmUrl": "./stable_diffusion.wasm",
  "cacheUrl": "./sd-webgpu-v1-5",
  "tokenizer": "openai/clip-vit-large-patch14"
}

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
  constructor(tvm, tokenizer, schedulerConsts, cacheMetadata) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
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
      this.tvm.getParamsFromCache("clip", cacheMetadata.clipParamSize)
    );
    this.unetLatentsToNoisePred = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unet")
    );
    this.unetParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("unet", cacheMetadata.unetParamSize)
    );
    this.vaeToImage = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("vae")
    );
    this.vaeParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("vae", cacheMetadata.vaeParamSize)
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
   * @param prompt Input prompt.
   * @param progressCallback Callback to check progress.
   * @param vaeCycle optionally draw VAE result every cycle iterations.
   * @param beginRenderVae Begin rendering VAE after skipping these warmup runs.
   */
  async generate(prompt, progressCallback = undefined, vaeCycle = -1, beginRenderVae = 10) {
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
    if (progressCallback !== undefined) {
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
    if (vaeCycle != -1) {
      // show first frame
      this.tvm.withNewScope(() => {
        const image = this.vaeToImage(latents, this.vaeParams);
        this.tvm.showImage(this.imageToRGBA(image));
      });
      await this.device.sync();
    }
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
      if ((counter + 1) % vaeCycle == 0 &&
          (counter + 1) != numSteps &&
          counter >= beginRenderVae) {
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

  clearCanvas() {
    this.tvm.clearCanvas();
  }
};

/**
 * A instance that can be used to facilitate deployment.
 */
class StableDiffusionInstance {
  constructor() {
    this.tvm = undefined;
    this.pipeline = undefined;
    this.config = undefined;
    this.generateInProgress = false;
    this.logger = console.log;
  }
  /**
   * Initialize TVM
   * @param wasmUrl URL to wasm source.
   * @param cacheUrl URL to NDArray cache.
   * @param logger Custom logger.
   */
  async #asyncInitTVM(wasmUrl, cacheUrl) {
    if (this.tvm !== undefined) {
      return;
    }

    if (document.getElementById("log") !== undefined) {
      this.logger = function(message) {
        console.log(message);
        const d = document.createElement("div");
        d.innerHTML = message;
        document.getElementById("log").appendChild(d);
      };
    }

    const wasmSource = await (
      await fetch(wasmUrl)
    ).arrayBuffer();
    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      new EmccWASI(),
      this.logger
    );
    // intialize WebGPU
    try {
      const output = await tvmjs.detectGPUDevice();
      if (output !== undefined) {
        const label = "WebGPU - "+ output.adapterInfo.description;
        document.getElementById(
          "gpu-tracker-label").innerHTML = ("Initialize GPU device: " + label);
        this.tvm.initWebGPU(output.device);
      } else {
        document.getElementById(
          "gpu-tracker-label").innerHTML = "This browser env do not support WebGPU";
      }
      this.reset();
      throw Error("This broweser env do not support WebGPU");
    } catch(err) {
      document.getElementById("gpu-tracker-label").innerHTML = (
        "Find an error initializing the WebGPU device " + err.toString()
      );
      console.log(err.stack);
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }

    this.tvm = tvm;
    function fetchProgressCallback(report) {
      document.getElementById("progress-tracker-label").innerHTML = report.text;
      document.getElementById("progress-tracker-progress").value = (report.fetchedBytes / report.totalBytes) * 100;
    }
    tvm.registerFetchProgressCallback(fetchProgressCallback);
    if (!cacheUrl.startsWith("http")) {
      cacheUrl = new URL(cacheUrl, document.URL).href;
    }
    await tvm.fetchNDArrayCache(cacheUrl, tvm.webgpu());
  }

  /**
   * Initialize the pipeline
   *
   * @param schedulerConstUrl The scheduler constant.
   * @param tokenizerName The name of the tokenizer.
   */
  async #asyncInitPipeline(schedulerConstUrl, tokenizerName) {
    if (this.tvm == undefined) {
      throw Error("asyncInitTVM is not called");
    }
    if (this.pipeline !== undefined) return;
    const schedulerConst = await(await fetch(schedulerConstUrl)).json();
    const tokenizer = await tvmjsGlobalEnv.getTokenizer(tokenizerName);
    this.pipeline = this.tvm.withNewScope(() => {
      return new StableDiffusionPipeline(this.tvm, tokenizer, schedulerConst, this.tvm.cacheMetadata);
    });
  }

  /**
   * Async intitialize config
   */
  async #asyncInitConfig() {
    if (this.config !== undefined) return;
    this.config = await(await fetch("stable-diffusion-config.json")).json();
  }

  /**
   * Function to create progress callback tracker.
   * @returns A progress callback tracker.
   */
   #getProgressCallback() {
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
    return progressCallback;
  }

 /**
   * Async initialize instance.
   */
  async asyncInit() {
    if (this.pipeline !== undefined) return;
    await this.#asyncInitConfig();
    await this.#asyncInitTVM(this.config.wasmUrl, this.config.cacheUrl);
    await this.#asyncInitPipeline(this.config.schedulerConstUrl, this.config.tokenizer);
  }

  /**
   * Async initialize
   *
   * @param tvm The tvm instance.
   */
  async asyncInitOnRPCServerLoad(tvmInstance) {
    if (this.tvm !== undefined) {
      throw Error("Cannot reuse a loaded instance for rpc");
    }
    this.tvm = tvmInstance;

    await this.#asyncInitConfig();
    await this.#asyncInitPipeline(this.config.schedulerConstUrl, this.config.tokenizer);

    this.tvm.beginScope();
    this.tvm.registerAsyncServerFunc("generate", async (prompt, vaeCycle) => {
      document.getElementById("inputPrompt").value = prompt;
      await this.pipeline.generate(prompt, this.#getProgressCallback(), vaeCycle);
    });

    this.tvm.registerAsyncServerFunc("clearCanvas", async () => {
      this.pipeline.clearCanvas();
    });
    this.tvm.endScope();
  }

  /**
   * Run generate
   */
  async generate() {
    if (this.requestInProgress) {
      this.logger("Request in progress, generate request ignored");
      return;
    }
    this.requestInProgress = true;
    try {
      await this.asyncInit();
      const prompt = document.getElementById("inputPrompt").value;
      const vaeCycle =document.getElementById("vaeCycle").value;
      await this.pipeline.generate(prompt, this.#getProgressCallback(), vaeCycle);
    } catch (err) {
      this.logger("Generate error, " + err.toString());
      console.log(err.stack);
      this.reset();
    }
    this.requestInProgress = false;
  }

  /**
   * Reset the instance;
   */
  reset() {
    this.tvm = undefined;
    this.pipeline = undefined;
  }
}

localStableDiffusionInst = new StableDiffusionInstance();

tvmjsGlobalEnv.asyncOnGenerate = async function() {
  await localStableDiffusionInst.generate();
};

tvmjsGlobalEnv.asyncOnRPCServerLoad = async function(tvm) {
  const inst = new StableDiffusionInstance();
  await inst.asyncInitOnRPCServerLoad(tvm);
};
