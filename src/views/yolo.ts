// Heavily derived from YAD2K (https://github.com/ModelDepot/tfjs-yolo-tiny-demo)

// import {Tensor } from "onnxruntime-web/dist/ort-web.min.js";
import {Tensor } from "onnxruntime-web/dist/ort-web.min.js";
// import * as yolo           from './utils-yolo/yoloPostprocess';
import * as yoloTransforms from './utils-yolo/yoloPostprocess';
export { yoloTransforms};

const classNames = [
  'Aeroplane',   'Bicycle', 'Bird',  'Boat',      'Bottle', 'Bus',         'Car',   'Cat',  'Chair', 'Cow',
  'Diningtable', 'Dog',     'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
];

export const YOLO_ANCHORS = new Tensor(
    'float32', Float32Array.from([
      1.08,
      1.19,
      3.42,
      4.41,
      6.63,
      11.38,
      9.42,
      5.11,
      16.62,
      10.52,
    ]),
    [5, 2]);
const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.3;
const INPUT_DIM = 416;

export async function postprocess(outputTensor: Tensor, numClasses: number) {
  const [boxXy, boxWh, boxConfidence, boxClassProbs] = yolo_head(outputTensor, YOLO_ANCHORS, 20);
  const allBoxes = yolo_boxes_to_corners(boxXy, boxWh);
  const [outputBoxes, scores, classes] =
      await yolo_filter_boxes(allBoxes, boxConfidence, boxClassProbs, DEFAULT_FILTER_BOXES_THRESHOLD);
  // If all boxes have been filtered out
  if (outputBoxes == null) {
    return [];
  }

  const width = yoloTransforms.scalar(INPUT_DIM);
  const height = yoloTransforms.scalar(INPUT_DIM);

  const imageDims = yoloTransforms.reshape(yoloTransforms.stack([height, width, height, width]), [
    1,
    4,
  ]);

  const boxes: Tensor = yoloTransforms.mul(outputBoxes, imageDims);

  const [preKeepBoxesArr, scoresArr] = await Promise.all([
    boxes.data,
    scores.data,
  ]);

  const [keepIndx, boxesArr, keepScores] = non_max_suppression(
      preKeepBoxesArr as Float32Array | Int32Array | Uint8Array, scoresArr as Float32Array | Int32Array | Uint8Array,
      DEFAULT_IOU_THRESHOLD);

  const classesIndxArr = (await yoloTransforms.gather(classes, new Tensor('int32', keepIndx)).data) as Float32Array;

  const results: any[] = [];

  classesIndxArr.forEach((classIndx, i) => {
    const classProb = keepScores[i];
    if (classProb < DEFAULT_CLASS_PROB_THRESHOLD) {
      return;
    }

    const className = classNames[classIndx];
    let [top, left, bottom, right] = boxesArr[i];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(416, bottom);
    right = Math.min(416, right);

    const resultObj = {
      className,
      classProb,
      bottom,
      top,
      left,
      right,
    };

    results.push(resultObj);
  });
  return results;
}

export async function yolo_filter_boxes(
    boxes: Tensor, boxConfidence: Tensor, boxClassProbs: Tensor, threshold: number) {
  const boxScores = yoloTransforms.mul(boxConfidence, boxClassProbs);
  const boxClasses = yoloTransforms.argMax(boxScores, -1);
  const boxClassScores = yoloTransforms.max(boxScores, -1);
  // Many thanks to @jacobgil
  // Source: https://github.com/ModelDepot/tfjs-yolo-tiny/issues/6#issuecomment-387614801
  const predictionMask = yoloTransforms.as1D(yoloTransforms.greaterEqual(boxClassScores, yoloTransforms.scalar(threshold)));

  const N = predictionMask.size;
  // linspace start/stop is inclusive.
  const allIndices = yoloTransforms.cast(yoloTransforms.linspace(0, N - 1, N), 'int32');
  const negIndices = yoloTransforms.zeros([N], 'int32');
  const indices = yoloTransforms.where(predictionMask, allIndices, negIndices);

  return [
    yoloTransforms.gather(yoloTransforms.reshape(boxes, [N, 4]), indices),
    yoloTransforms.gather(yoloTransforms.as1D(boxClassScores), indices),
    yoloTransforms.gather(yoloTransforms.as1D(boxClasses), indices),
  ];
}

/**
 * Given XY and WH tensor outputs of yolo_head, returns corner coordinates.
 * @param {Tensor} box_xy Bounding box center XY coordinate Tensor
 * @param {Tensor} box_wh Bounding box WH Tensor
 * @returns {Tensor} Bounding box corner Tensor
 */
export function yolo_boxes_to_corners(boxXy: Tensor, boxWh: Tensor) {
  const two = new Tensor('float32', [2.0]);
  const boxMins = yoloTransforms.sub(boxXy, yoloTransforms.div(boxWh, two));
  const boxMaxes = yoloTransforms.add(boxXy, yoloTransforms.div(boxWh, two));

  const dim0 = boxMins.dims[0];
  const dim1 = boxMins.dims[1];
  const dim2 = boxMins.dims[2];
  const size = [dim0, dim1, dim2, 1];

  return yoloTransforms.concat(
      [
        yoloTransforms.slice(boxMins, [0, 0, 0, 1], size),
        yoloTransforms.slice(boxMins, [0, 0, 0, 0], size),
        yoloTransforms.slice(boxMaxes, [0, 0, 0, 1], size),
        yoloTransforms.slice(boxMaxes, [0, 0, 0, 0], size),
      ],
      3);
}

/**
 * Filters/deduplicates overlapping boxes predicted by YOLO. These
 * operations are done on CPU as AFAIK, there is no tfjs way to do it
 * on GPU yet.
 * @param {TypedArray} boxes Bounding box corner data buffer from Tensor
 * @param {TypedArray} scores Box scores data buffer from Tensor
 * @param {Number} iouThreshold IoU cutoff to filter overlapping boxes
 */
export function non_max_suppression(
    boxes: Float32Array|Int32Array|Uint8Array, scores: Float32Array|Int32Array|Uint8Array, iouThreshold: number) {
  // Zip together scores, box corners, and index
  const zipped = [];
  for (let i = 0; i < scores.length; i++) {
    zipped.push([
      scores[i],
      [boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]],
      i,
    ]);
  }
  // Sort by descending order of scores (first index of zipped array)
  const sortedBoxes = zipped.sort((a: number[], b: number[]) => b[0] - a[0]);

  const selectedBoxes: any[] = [];

  // Greedily go through boxes in descending score order and only
  // return boxes that are below the IoU threshold.
  sortedBoxes.forEach((box: any[]) => {
    let add = true;
    for (let i = 0; i < selectedBoxes.length; i++) {
      // Compare IoU of zipped[1], since that is the box coordinates arr
      // TODO: I think there's a bug in this calculation
      const curIou = box_iou(box[1], selectedBoxes[i][1]);
      if (curIou > iouThreshold) {
        add = false;
        break;
      }
    }
    if (add) {
      selectedBoxes.push(box);
    }
  });

  // Return the kept indices and bounding boxes
  return [
    selectedBoxes.map((e) => e[2]),
    selectedBoxes.map((e) => e[1]),
    selectedBoxes.map((e) => e[0]),
  ];
}

// Convert yolo output to bounding box + prob tensors
export function yolo_head(feats: Tensor, anchors: Tensor, numClasses: number) {
  const numAnchors = anchors.dims[0];

  const anchorsArray = yoloTransforms.reshape(anchors, [1, 1, numAnchors, 2]);

  const convDims = feats.dims.slice(1, 3);

  // For later use
  const convDims0 = convDims[0];
  const convDims1 = convDims[1];

  let convHeightIndex = yoloTransforms.range(0, convDims[0]);
  let convWidthIndex = yoloTransforms.range(0, convDims[1]);

  convHeightIndex = yoloTransforms.tile(convHeightIndex, [convDims[1]]);

  convWidthIndex = yoloTransforms.tile(yoloTransforms.expandDims(convWidthIndex, 0), [
    convDims[0],
    1,
  ]);
  convWidthIndex = yoloTransforms.as1D(yoloTransforms.transpose(convWidthIndex));

  let convIndex = yoloTransforms.transpose(yoloTransforms.stack([convHeightIndex, convWidthIndex]));
  convIndex = yoloTransforms.reshape(convIndex, [convDims[0], convDims[1], 1, 2]);
  convIndex = yoloTransforms.cast(convIndex, feats.type);

  feats = yoloTransforms.reshape(feats, [
    convDims[0],
    convDims[1],
    numAnchors,
    numClasses + 5,
  ]);
  const convDimsTensor = yoloTransforms.cast(yoloTransforms.reshape(new Tensor('int32', convDims), [1, 1, 1, 2]), feats.type);

  let boxXy = yoloTransforms.sigmoid(yoloTransforms.slice(feats, [0, 0, 0, 0], [convDims0, convDims1, numAnchors, 2]));
  let boxWh = yoloTransforms.exp(yoloTransforms.slice(feats, [0, 0, 0, 2], [convDims0, convDims1, numAnchors, 2]));
  const boxConfidence = yoloTransforms.sigmoid(yoloTransforms.slice(feats, [0, 0, 0, 4], [convDims0, convDims1, numAnchors, 1]));
  const boxClassProbs = yoloTransforms.softmax(yoloTransforms.slice(feats, [0, 0, 0, 5], [convDims0, convDims1, numAnchors, numClasses]));

  boxXy = yoloTransforms.div(yoloTransforms.add(boxXy, convIndex), convDimsTensor);
  boxWh = yoloTransforms.div(yoloTransforms.mul(boxWh, anchorsArray), convDimsTensor);
  // boxXy = tf.mul(tf.add(boxXy, convIndex), 32);
  // boxWh = tf.mul(tf.mul(boxWh, anchorsArray), 32);
  return [boxXy, boxWh, boxConfidence, boxClassProbs];
}

export function box_intersection(a: number[], b: number[]) {
  const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  if (w < 0 || h < 0) {
    return 0;
  }
  return w * h;
}

export function box_union(a: number[], b: number[]) {
  const i = box_intersection(a, b);
  return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

export function box_iou(a: number[], b: number[]) {
  return box_intersection(a, b) / box_union(a, b);
}
