// One perceptron, written as loops students can read.
//
// z = bias + x1 * w1 + x2 * w2 + ...
//
// One step visits every row with the same weights, then adds the average pull.
// Update for the selected activation, before that average:
//   Step:    delta = learningRate * (target - output) * input
//   Linear:  same, because the slope is 1
//   Sigmoid: multiply by output * (1 - output)
//   ReLU:    output = z when z > 0, else 0.
//            delta = learningRate * (target - output) * (z > 0 ? 1 : 0) * input
//   Softmax: output is P(output = 1) from scores z and 0.
//            The loss slope on that score is P(1) - target.
//            The weight update uses the opposite sign, so
//            factor = target - P(1). The unused score would move the other way.

const ACTIVATIONS = {
  boolean: [
    { id: "step_gt", label: "Step (z > 0)" },
    { id: "step_lt", label: "Step (z < 0)" },
    { id: "sigmoid", label: "Sigmoid" },
    { id: "relu", label: "ReLU" },
    { id: "linear", label: "Linear" },
    { id: "softmax", label: "Softmax" },
  ],
  variable: [
    { id: "step_gt", label: "Step (z > 0)" },
    { id: "step_lt", label: "Step (z < 0)" },
    { id: "sigmoid", label: "Sigmoid" },
    { id: "relu", label: "ReLU" },
    { id: "linear", label: "Linear" },
  ],
};

const FORMULAS = {
  step_gt: "output = 1 when z > 0, else 0.  delta = learningRate × (target − output) × input",
  step_lt: "output = 1 when z < 0, else 0.  delta = learningRate × (target − output) × input",
  sigmoid: "output = 1 / (1 + e^(−z)).  delta = learningRate × (target − output) × output × (1 − output) × input",
  relu: "output = z when z > 0, else 0.  delta = learningRate × (target − output) × (z > 0 ? 1 : 0) × input",
  linear: "output = z.  delta = learningRate × (target − output) × input",
  softmax: "scores are z and 0.  P(1) = e^z / (e^z + e^0).  delta = learningRate × (target − P(1)) × input",
};

function rowPauseMs() {
  const speed = Number(document.getElementById("speed").value);
  const level = Number.isFinite(speed) && speed > 0 ? speed : 3;
  return Math.round(600 / level);
}

let inputNames = [];
let rows = [];
let weights = [];
let bias = 0;
let stepCount = 0;
let lastChange = null;
let activeRow = -1;
let nextInfer = 0;
let shown = null;
let playing = false;

function round2(value) {
  return Math.round(value * 100) / 100;
}

function formatNum(value) {
  const rounded = round2(value);
  const body = Math.abs(rounded).toFixed(2);
  if (rounded < 0) return "−" + body;
  return body;
}

function formatSigned(value) {
  const rounded = round2(value);
  if (rounded > 0) return "+" + rounded.toFixed(2);
  if (rounded < 0) return "−" + Math.abs(rounded).toFixed(2);
  return "0.00";
}

function sleep(ms) {
  return new Promise(function wake(resolve) {
    setTimeout(resolve, ms);
  });
}

function learningRate() {
  const value = Number(document.getElementById("learning-rate").value);
  if (!Number.isFinite(value)) return 0.1;
  return value;
}

function requestedPasses() {
  const value = Math.floor(Number(document.getElementById("steps").value));
  if (!Number.isFinite(value) || value < 1) return 1;
  return value;
}

function activationName() {
  return document.getElementById("activation").value;
}

function outputMode() {
  return document.getElementById("output-mode").value;
}

function weightLook(weight) {
  const thickness = 1.5 + Math.min(6.5, Math.abs(weight) * 4);
  let color = "#1c1915";
  if (weight > 0.005) color = "#0f6e56";
  else if (weight < -0.005) color = "#9a3412";
  return { thickness: thickness, color: color };
}

// z = bias + each input times its weight.
function weightedSum(values) {
  let z = bias;
  for (let i = 0; i < values.length; i++) {
    z += values[i] * weights[i];
  }
  return z;
}

function sigmoid(z) {
  if (z > 20) return 1;
  if (z < -20) return 0;
  return 1 / (1 + Math.exp(-z));
}

// Softmax of two scores: z for output 1, and 0 for output 0.
function softmaxPositive(z) {
  const clipped = Math.max(-20, Math.min(20, z));
  const score1 = Math.exp(clipped);
  const score0 = Math.exp(0);
  return score1 / (score1 + score0);
}

function activate(name, z) {
  if (name === "step_gt") return z > 0 ? 1 : 0;
  if (name === "step_lt") return z < 0 ? 1 : 0;
  if (name === "sigmoid") return sigmoid(z);
  if (name === "relu") return z > 0 ? z : 0;
  if (name === "softmax") return softmaxPositive(z);
  return z;
}

// How far the output should pull z, before multiplying by the input.
function errorFactor(name, target, output, z) {
  const error = target - output;
  if (name === "sigmoid") return error * output * (1 - output);
  if (name === "relu") return error * (z > 0 ? 1 : 0);
  return error;
}

function zeroWeights() {
  weights = [];
  for (let i = 0; i < inputNames.length; i++) weights.push(0);
  bias = 0;
  stepCount = 0;
  lastChange = null;
  activeRow = -1;
  nextInfer = 0;
  shown = null;
}

function predictRow(row) {
  return activate(activationName(), weightedSum(row.values));
}

function setPlaying(on) {
  playing = on;
  const ids = [
    "step", "run", "infer", "random-weights", "reset-weights",
    "add-record", "add-input", "remove-input", "steps",
    "output-mode", "activation", "learning-rate",
  ];
  for (let i = 0; i < ids.length; i++) {
    document.getElementById(ids[i]).disabled = on;
  }
  if (!on) document.getElementById("remove-input").disabled = inputNames.length <= 1;
}

async function loadProvidedDataset() {
  const mode = outputMode();
  const file = mode === "boolean"
    ? "data/perceptron_boolean.json"
    : "data/perceptron_variable.json";
  const response = await fetch(file);
  if (!response.ok) throw new Error("Could not load " + file);
  const data = await response.json();
  inputNames = data.inputs.slice();
  rows = [];
  for (let r = 0; r < data.rows.length; r++) {
    rows.push({
      values: data.rows[r].values.slice(),
      target: data.rows[r].target,
    });
  }
  ensureActivationFitsMode();
  zeroWeights();
  render();
  document.getElementById("readout").textContent = "Weights start at 0. Press Step to train every row once.";
}

function ensureActivationFitsMode() {
  const select = document.getElementById("activation");
  const choices = ACTIVATIONS[outputMode()];
  const previous = select.value;
  select.textContent = "";
  let keep = false;
  for (let i = 0; i < choices.length; i++) {
    const option = document.createElement("option");
    option.value = choices[i].id;
    option.textContent = choices[i].label;
    if (choices[i].id === previous) keep = true;
    select.appendChild(option);
  }
  select.value = keep ? previous : choices[0].id;
}

function addRecord() {
  if (playing) return;
  const values = [];
  for (let i = 0; i < inputNames.length; i++) values.push(0);
  rows.push({ values: values, target: 0 });
  render();
}

function deleteRow(index) {
  if (playing) return;
  rows.splice(index, 1);
  if (activeRow > index) activeRow -= 1;
  if (activeRow >= rows.length) activeRow = rows.length - 1;
  if (nextInfer > index) nextInfer -= 1;
  if (nextInfer >= rows.length) nextInfer = 0;
  render();
}

function addInput() {
  if (playing) return;
  inputNames.push("x" + (inputNames.length + 1));
  weights.push(0);
  for (let r = 0; r < rows.length; r++) rows[r].values.push(0);
  lastChange = null;
  render();
}

function removeInput() {
  if (playing || inputNames.length <= 1) return;
  inputNames.pop();
  weights.pop();
  for (let r = 0; r < rows.length; r++) rows[r].values.pop();
  lastChange = null;
  render();
}

function randomWeights() {
  if (playing) return;
  for (let i = 0; i < weights.length; i++) {
    weights[i] = round2(Math.random() * 2 - 1);
  }
  bias = round2(Math.random() * 2 - 1);
  lastChange = null;
  shown = null;
  render();
  document.getElementById("readout").textContent = "Random weights are set. Press Step to train every row once.";
}

function resetWeights() {
  if (playing) return;
  zeroWeights();
  render();
  document.getElementById("readout").textContent = "Weights are 0 again. Press Step to train every row once.";
}

// One row's pull. Weights stay put until the pass averages every pull.
function rowPull(rowNumber) {
  const row = rows[rowNumber];
  const name = activationName();
  const rate = learningRate();
  const z = weightedSum(row.values);
  const output = activate(name, z);
  const factor = errorFactor(name, row.target, output, z);
  const weightDeltas = [];
  for (let i = 0; i < weights.length; i++) {
    weightDeltas.push(rate * factor * row.values[i]);
  }
  activeRow = rowNumber;
  shown = { values: row.values.slice(), z: z, output: output };
  lastChange = null;
  document.getElementById("readout").textContent = [
    "step " + stepCount + ", row " + (rowNumber + 1),
    "z = " + formatNum(z),
    "output = " + formatNum(output),
    "target = " + formatNum(row.target),
    "error = " + formatNum(row.target - output),
  ].join("\n");
  return { weightDeltas: weightDeltas, biasDelta: rate * factor };
}

function applyAverage(sumWeights, sumBias, count) {
  const oldWeights = weights.slice();
  const oldBias = bias;
  const weightDeltas = [];
  for (let i = 0; i < weights.length; i++) {
    const delta = round2(sumWeights[i] / count);
    weightDeltas.push(delta);
    weights[i] = round2(weights[i] + delta);
  }
  const biasDelta = round2(sumBias / count);
  bias = round2(bias + biasDelta);
  lastChange = {
    weightDeltas: weightDeltas,
    biasDelta: biasDelta,
    oldWeights: oldWeights,
    oldBias: oldBias,
  };
  const lines = [
    "step " + stepCount + ", average of " + count + (count === 1 ? " row" : " rows"),
  ];
  for (let i = 0; i < inputNames.length; i++) {
    lines.push(
      "w" + (i + 1) + ": " + formatNum(oldWeights[i]) + " → " + formatNum(weights[i])
      + " (" + formatSigned(weightDeltas[i]) + ")"
    );
  }
  lines.push(
    "bias: " + formatNum(oldBias) + " → " + formatNum(bias)
    + " (" + formatSigned(biasDelta) + ")"
  );
  document.getElementById("readout").textContent = lines.join("\n");
}

function targetsReached() {
  for (let r = 0; r < rows.length; r++) {
    if (round2(predictRow(rows[r])) !== round2(rows[r].target)) return false;
  }
  return true;
}

function showInferenceRow(rowNumber) {
  const row = rows[rowNumber];
  const z = weightedSum(row.values);
  const output = activate(activationName(), z);
  activeRow = rowNumber;
  shown = { values: row.values.slice(), z: z, output: output };
  lastChange = null;
  const lines = [
    "infer, row " + (rowNumber + 1),
    "z = " + formatNum(z),
    "output = " + formatNum(output),
    "target = " + formatNum(row.target),
    "weights stay the same",
  ];
  for (let i = 0; i < inputNames.length; i++) {
    lines.push(inputNames[i] + " = " + formatNum(row.values[i]));
  }
  document.getElementById("readout").textContent = lines.join("\n");
}

async function runPasses(count, stopWhenReached) {
  if (playing) return;
  if (rows.length === 0) {
    document.getElementById("readout").textContent = "Add a record before stepping.";
    return;
  }
  if (stopWhenReached && targetsReached()) {
    document.getElementById("readout").textContent = "Targets already reached. Run stopped.";
    return;
  }
  setPlaying(true);
  for (let pass = 0; pass < count; pass++) {
    stepCount += 1;
    const sumWeights = [];
    for (let i = 0; i < weights.length; i++) sumWeights.push(0);
    let sumBias = 0;
    const countRows = rows.length;
    for (let r = 0; r < countRows; r++) {
      const pull = rowPull(r);
      for (let i = 0; i < sumWeights.length; i++) sumWeights[i] += pull.weightDeltas[i];
      sumBias += pull.biasDelta;
      render();
      await sleep(rowPauseMs());
    }
    applyAverage(sumWeights, sumBias, countRows);
    render();
    if (stopWhenReached && targetsReached()) {
      const readout = document.getElementById("readout");
      readout.textContent += "\ntargets reached, run stopped";
      await sleep(rowPauseMs());
      break;
    }
    await sleep(rowPauseMs());
  }
  setPlaying(false);
  render();
}

function inferOne() {
  if (playing) return;
  if (rows.length === 0) {
    document.getElementById("readout").textContent = "Add a record before inference.";
    return;
  }
  if (nextInfer >= rows.length) nextInfer = 0;
  showInferenceRow(nextInfer);
  nextInfer += 1;
  render();
}

function renderFormula() {
  document.getElementById("formula").textContent = FORMULAS[activationName()];
}

function renderTable() {
  const head = document.getElementById("data-head");
  const body = document.getElementById("data-body");
  head.textContent = "";
  body.textContent = "";

  const header = document.createElement("tr");
  header.appendChild(cell("th", "row"));
  for (let i = 0; i < inputNames.length; i++) header.appendChild(cell("th", inputNames[i]));
  header.appendChild(cell("th", "target"));
  header.appendChild(cell("th", "prediction"));
  header.appendChild(cell("th", ""));
  head.appendChild(header);

  for (let r = 0; r < rows.length; r++) {
    const tr = document.createElement("tr");
    if (r === activeRow) tr.className = "active";
    tr.appendChild(cell("td", String(r + 1)));
    for (let c = 0; c < inputNames.length; c++) {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = String(rows[r].values[c]);
      input.disabled = playing;
      input.addEventListener("input", function onValue(event) {
        const parsed = Number(event.target.value);
        rows[r].values[c] = Number.isFinite(parsed) ? parsed : 0;
        renderDiagram();
        fillPredictions();
      });
      td.appendChild(input);
      tr.appendChild(td);
    }
    tr.appendChild(targetCell(r));
    const prediction = document.createElement("td");
    prediction.dataset.prediction = "1";
    prediction.textContent = formatNum(predictRow(rows[r]));
    tr.appendChild(prediction);
    const action = document.createElement("td");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete";
    remove.textContent = "Delete";
    remove.disabled = playing;
    remove.addEventListener("click", function onDelete() {
      deleteRow(r);
    });
    action.appendChild(remove);
    tr.appendChild(action);
    body.appendChild(tr);
  }
  if (!playing) document.getElementById("remove-input").disabled = inputNames.length <= 1;
}

function cell(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

function targetCell(rowIndex) {
  const td = document.createElement("td");
  if (outputMode() === "boolean") {
    const select = document.createElement("select");
    for (let value = 0; value <= 1; value++) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      if (Number(rows[rowIndex].target) === value) option.selected = true;
      select.appendChild(option);
    }
    select.disabled = playing;
    select.addEventListener("change", function onTarget(event) {
      rows[rowIndex].target = Number(event.target.value);
      fillPredictions();
    });
    td.appendChild(select);
  } else {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = String(rows[rowIndex].target);
    input.disabled = playing;
    input.addEventListener("input", function onTarget(event) {
      const parsed = Number(event.target.value);
      rows[rowIndex].target = Number.isFinite(parsed) ? parsed : 0;
    });
    td.appendChild(input);
  }
  return td;
}

function fillPredictions() {
  const cells = document.querySelectorAll("[data-prediction]");
  for (let r = 0; r < cells.length && r < rows.length; r++) {
    cells[r].textContent = formatNum(predictRow(rows[r]));
  }
}

function renderDiagram() {
  const svg = document.getElementById("diagram");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const svgNs = "http://www.w3.org/2000/svg";
  const rowHeight = 88;
  const top = 48;
  const width = 860;
  const yAt = function yOf(index) {
    return top + index * rowHeight;
  };
  const ySum = inputNames.length === 0
    ? top
    : (yAt(0) + yAt(inputNames.length - 1)) / 2;
  const yBias = ySum + 120;
  const height = Math.max(300, yBias + 56);
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);

  const xIn = 80;
  const xBox = 270;
  const xSum = 480;
  const xAct = 650;
  const xOut = 790;

  for (let i = 0; i < inputNames.length; i++) {
    const y = yAt(i);
    drawLine(svg, svgNs, xIn + 32, y, xBox - 46, y, 1.5, "#1c1915");
    const look = weightLook(weights[i]);
    drawLine(svg, svgNs, xBox + 46, y, xSum - 34, ySum, look.thickness, look.color);
  }
  const biasLook = weightLook(bias);
  const biasBoxHeight = lastChange ? 48 : 36;
  drawLine(svg, svgNs, xSum, yBias - biasBoxHeight / 2, xSum, ySum + 34, biasLook.thickness, biasLook.color);
  drawLine(svg, svgNs, xSum + 34, ySum, xAct - 34, ySum, 1.5, "#1c1915");
  drawLine(svg, svgNs, xAct + 34, ySum, xOut - 34, ySum, 1.5, "#1c1915");

  for (let i = 0; i < inputNames.length; i++) {
    const y = yAt(i);
    const inputValue = shown ? formatNum(shown.values[i]) : null;
    drawNode(svg, svgNs, xIn, y, inputNames[i], inputValue);
    const delta = lastChange ? lastChange.weightDeltas[i] : null;
    drawWeightBox(svg, svgNs, xBox, y, "w" + (i + 1), weights[i], delta);
  }

  drawNode(svg, svgNs, xSum, ySum, "sum", shown ? formatNum(shown.z) : null);
  drawNode(svg, svgNs, xAct, ySum, "f", shown ? formatNum(shown.output) : null);
  drawNode(svg, svgNs, xOut, ySum, "y", shown ? formatNum(shown.output) : null);
  drawWeightBox(svg, svgNs, xSum, yBias, "bias", bias, lastChange ? lastChange.biasDelta : null);
  const activationLabel = document.getElementById("activation").selectedOptions[0].textContent;
  drawText(svg, svgNs, xAct, ySum + 52, activationLabel, "#1c1915");
}

function drawLine(svg, svgNs, x1, y1, x2, y2, thickness, color) {
  const line = document.createElementNS(svgNs, "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", thickness);
  line.setAttribute("stroke-linecap", "round");
  svg.appendChild(line);
}

function drawNode(svg, svgNs, x, y, label, value) {
  const circle = document.createElementNS(svgNs, "circle");
  circle.setAttribute("cx", x);
  circle.setAttribute("cy", y);
  circle.setAttribute("r", 32);
  circle.setAttribute("fill", "#fffdf8");
  circle.setAttribute("stroke", "#1c1915");
  circle.setAttribute("stroke-width", "1.5");
  svg.appendChild(circle);
  if (value === null || value === undefined) {
    drawText(svg, svgNs, x, y + 4, label, "#1c1915");
  } else {
    drawText(svg, svgNs, x, y - 6, label, "#5c564c");
    drawText(svg, svgNs, x, y + 14, value, "#1c1915");
  }
}

function drawWeightBox(svg, svgNs, x, y, title, weight, delta) {
  const look = weightLook(weight);
  const height = delta === null || delta === undefined ? 36 : 48;
  const rect = document.createElementNS(svgNs, "rect");
  rect.setAttribute("x", x - 44);
  rect.setAttribute("y", y - height / 2);
  rect.setAttribute("width", 88);
  rect.setAttribute("height", height);
  rect.setAttribute("rx", 8);
  rect.setAttribute("fill", "#fffdf8");
  rect.setAttribute("stroke", look.color);
  rect.setAttribute("stroke-width", Math.max(1.5, look.thickness / 2));
  svg.appendChild(rect);
  drawText(svg, svgNs, x, y + (delta === null || delta === undefined ? 4 : -4), title + " " + formatNum(weight), "#1c1915");
  if (delta !== null && delta !== undefined) {
    const deltaColor = delta < 0 ? "#9a3412" : "#0f6e56";
    drawText(svg, svgNs, x, y + 16, formatSigned(delta), deltaColor);
  }
}

function drawText(svg, svgNs, x, y, label, color) {
  const text = document.createElementNS(svgNs, "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("font-size", "13");
  text.setAttribute("font-family", "ui-sans-serif, system-ui, sans-serif");
  text.setAttribute("fill", color);
  text.textContent = label;
  svg.appendChild(text);
  return text;
}

function render() {
  renderFormula();
  renderTable();
  renderDiagram();
}

document.addEventListener("DOMContentLoaded", function start() {
  ensureActivationFitsMode();
  document.getElementById("output-mode").addEventListener("change", loadProvidedDataset);
  document.getElementById("activation").addEventListener("change", render);
  document.getElementById("learning-rate").addEventListener("change", render);
  document.getElementById("add-record").addEventListener("click", addRecord);
  document.getElementById("add-input").addEventListener("click", addInput);
  document.getElementById("remove-input").addEventListener("click", removeInput);
  document.getElementById("step").addEventListener("click", function onStep() {
    runPasses(1);
  });
  document.getElementById("run").addEventListener("click", function onRun() {
    runPasses(requestedPasses(), true);
  });
  document.getElementById("infer").addEventListener("click", inferOne);
  document.getElementById("random-weights").addEventListener("click", randomWeights);
  document.getElementById("reset-weights").addEventListener("click", resetWeights);
  loadProvidedDataset().catch(function onLoadError(error) {
    document.getElementById("readout").textContent = error.message;
  });
});
