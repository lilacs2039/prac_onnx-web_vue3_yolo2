<script setup lang="ts">
import { ref } from "vue";
// import { InferenceSession, Tensor } from "onnxruntime-web";
import { InferenceSession, Tensor } from "onnxruntime-web/dist/ort-web.min.js";
import modelUrl from "../yolo.onnx?url";
import ops from "ndarray-ops";
import ndarray from "ndarray";
import { yoloTransforms } from "./yolo";
import * as yolo from "./yolo";
import loadImage from "blueimp-load-image";
import img_bird_url from "../assets/bird.jpg?url";

const imageSize = 416;
const sessionBackend = ref("webgl");
const result_str = ref("");
let webcamContainer: HTMLElement;
let session: InferenceSession;
// ----------------------- main process --------------------

(async () => {
  const response = await fetch(modelUrl);
  const modelFile = await response.arrayBuffer();
  webcamContainer = document.getElementById("webcam-container") as HTMLElement;

  // var session:InferenceSession;
  console.info("Creating session...");
  try {
    session =
      sessionBackend.value == "webgl"
        ? await InferenceSession.create(modelFile, {
            executionProviders: ["webgl"],
          })
        : await await InferenceSession.create(modelFile, {
            executionProviders: ["wasm"],
          });
  } catch (e) {
    console.error(e);
    throw new Error("Error: Backend not supported. ");
  }
  console.info("Warming up model...");
  warmupModel(session, [1, 3, 416, 416]);
  console.info("Done");
  loadImageToCanvas(img_bird_url);
})();

// ------------------------- YOLO functions --------------------------------
async function warmupModel(model: InferenceSession, dims: number[]) {
  // OK. we generate a random input and call Session.run() as a warmup query
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor("float32", new Float32Array(size), dims);

  for (let i = 0; i < size; i++) {
    warmupTensor.data[i] = Math.random() * 2.0 - 1.0; // random value [-1.0, 1.0)
  }
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = warmupTensor;
    await model.run(feeds);
  } catch (e) {
    console.error(e);
  }
}

async function runModel() {
  const ctx = (
    document.getElementById("input-canvas") as HTMLCanvasElement
  ).getContext("2d") as CanvasRenderingContext2D;
  const data = preprocess(ctx);

  console.info("Running model...");
  let outputTensor: Tensor;
  let inferenceTime: number;
  [outputTensor, inferenceTime] = await _runModel(session, data);
  clearRects();
  console.info("post process...");
  postprocess(outputTensor, inferenceTime);
  console.info("Done inference!");
  console.info(outputTensor);
  return;

  function clearRects() {
    while (webcamContainer.childNodes.length > 2) {
      webcamContainer.removeChild(webcamContainer.childNodes[2]);
    }
  }

  async function _runModel(
    model: InferenceSession,
    preprocessedData: Tensor
  ): Promise<[Tensor, number]> {
    const start = new Date();
    try {
      const feeds: Record<string, Tensor> = {};
      feeds[model.inputNames[0]] = preprocessedData;
      const outputData = await model.run(feeds);
      const end = new Date();
      const inferenceTime = end.getTime() - start.getTime();
      const output = outputData[model.outputNames[0]];

      return [output, inferenceTime];
    } catch (e) {
      throw e;
    }
  }

  function preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  async function postprocess(tensor: Tensor, inferenceTime: number) {
    try {
      const originalOutput = new Tensor(
        "float32",
        tensor.data as Float32Array,
        [1, 125, 13, 13]
      );
      const outputTensor = yoloTransforms.transpose(
        originalOutput,
        [0, 2, 3, 1]
      );

      // postprocessing
      const boxes = await yolo.postprocess(outputTensor, 20);
      boxes.forEach((box) => {
        const { top, left, bottom, right, classProb, className } = box;

        result_str.value = `${className} Confidence: ${Math.round(
            classProb * 100
          )}% Time: ${inferenceTime.toFixed(1)}ms`;
        drawRect(
          left,
          top,
          right - left,
          bottom - top,
          result_str.value
        );

      });
    } catch (e) {
      alert("Model is not valid!");
      console.error(e);
    }
    return;

    function drawRect(
      x: number,
      y: number,
      w: number,
      h: number,
      text = "",
      color = "red"
    ) {
      const webcamContainerElement = document.getElementById(
        "webcam-container"
      ) as HTMLElement;
      // Depending on the display size, webcamContainerElement might be smaller than 416x416.
      const [ox, oy] = [
        (webcamContainerElement.offsetWidth - 416) / 2,
        (webcamContainerElement.offsetHeight - 416) / 2,
      ];
      const rect = document.createElement("div");
      rect.style.cssText = `top: ${y + oy}px; left: ${
        x + ox
      }px; width: ${w}px; height: ${h}px; border-color: ${color};`;
      const label = document.createElement("div");
      label.innerText = text;
      rect.appendChild(label);

      webcamContainerElement.appendChild(rect);
    }
  }
}

// ----------------------- UI functions --------------------
function handleFileChange(e: any) {
  loadImageToCanvas(e.target.files[0]);
  runModel();
  return;
}
function loadImageToCanvas(url: string) {
  if (!url) {
    clearAll();
    return;
  }
  loadImage(
    url,
    (img) => {
      if ((img as Event).type === "error") {
        console.error("Error on loading image");
      } else {
        // load image data onto input canvas
        const element = document.getElementById(
          "input-canvas"
        ) as HTMLCanvasElement;
        if (element) {
          const ctx = element.getContext("2d");
          if (ctx) {
            ctx.drawImage(img as HTMLImageElement, 0, 0);

            // this.output = [];

            // session predict
            // this.$nextTick(function () {
            //   setTimeout(() => {
            //     this.runModel();
            //   }, 10);
            // });
          }
        }
      }
    },
    {
      maxWidth: imageSize,
      maxHeight: imageSize,
      cover: true,
      crop: true,
      canvas: true,
      crossOrigin: "Anonymous",
    }
  );

  function clearAll() {
    const element = document.getElementById(
      "input-canvas"
    ) as HTMLCanvasElement;
    if (element) {
      const ctx = element.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    }

    const file = document.getElementById(
      "input-upload-image"
    ) as HTMLInputElement;
    if (file) {
      file.value = "";
    }
  }
}
</script>

<template>
  <main>
    <h1>Yolo</h1>
    <div class="webcam-container" id="webcam-container" display="none">
      <video playsinline muted id="webcam" width="416" height="416"></video>
      <canvas
        id="input-canvas"
        width="416"
        height="416"
        style="position: absolute"
      ></canvas>
    </div>

    <div>
      <span>UPLOAD IMAGE</span>
    </div>
    <!-- style="display: none" -->
    <input type="file" id="input-upload-image" @change="handleFileChange" />
    <div>
      <button type="button" id="run-button" @click="runModel()">Infer!</button>
    </div>
    <div id="result">{{ result_str }}</div>
  </main>
</template>

<style scoped>
.webcam-container {
  border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
  margin: 0 auto;
  width: 416px;
  height: 416px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.run-button {
  font-size: 24px;
}
</style>