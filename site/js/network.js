// A small network students can reshape.
//
// Columns: inputs, then each hidden layer, then one output y.
// z = bias + each incoming value times its weight.
// One step scores every row with the same weights, then adds the average pull.
//
// Hidden and output use the same activation rules as the perceptron:
//   Step:    delta = learningRate * (target - output) * input
//   Linear:  same, because the slope is 1
//   Sigmoid: multiply by output * (1 - output)
//   ReLU:    output = z when z > 0, else 0
//   Softmax: output is P(output = 1) from scores z and 0

const ACTIVATIONS = [
  { id: "step_gt", label: "Step (z > 0)" },
  { id: "step_lt", label: "Step (z < 0)" },
  { id: "sigmoid", label: "Sigmoid" },
  { id: "relu", label: "ReLU" },
  { id: "linear", label: "Linear" },
  { id: "softmax", label: "Softmax" },
];

const MAX_LAYER_SIZE = 32;

let inputNames = [];
let rows = [];
let hiddenSizes = [2];
let weights = [];
let biases = [];
let stepCount = 0;
let lastChange = null;
let activeRow = -1;
let nextInfer = 0;
let shown = null;
let playing = false;
let pendingDelete = -1;
let hideValues = false;
let hideTouched = false;

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

function rowPauseMs() {
  const speed = Number(document.getElementById("speed").value);
  const level = Number.isFinite(speed) && speed > 0 ? speed : 3;
  return Math.round(600 / level);
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

function hiddenActivation() {
  return document.getElementById("hidden-activation").value;
}

function outputActivation() {
  return document.getElementById("output-activation").value;
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

function sigmoid(z) {
  if (z > 20) return 1;
  if (z < -20) return 0;
  return 1 / (1 + Math.exp(-z));
}

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

function errorFactor(name, target, output, z) {
  const error = target - output;
  if (name === "sigmoid") return error * output * (1 - output);
  if (name === "relu") return error * (z > 0 ? 1 : 0);
  return error;
}

function hiddenDerivative(name, output, z) {
  if (name === "sigmoid") return output * (1 - output);
  if (name === "relu") return z > 0 ? 1 : 0;
  return 1;
}

function blankMatrix(from, to) {
  const matrix = [];
  for (let i = 0; i < from; i++) {
    const row = [];
    for (let j = 0; j < to; j++) row.push(0);
    matrix.push(row);
  }
  return matrix;
}

function blankBias(count) {
  const bias = [];
  for (let i = 0; i < count; i++) bias.push(0);
  return bias;
}

function rebuildWeights() {
  const widths = [inputNames.length];
  for (let i = 0; i < hiddenSizes.length; i++) widths.push(hiddenSizes[i]);
  widths.push(1);
  weights = [];
  biases = [];
  for (let layer = 1; layer < widths.length; layer++) {
    weights.push(blankMatrix(widths[layer - 1], widths[layer]));
    biases.push(blankBias(widths[layer]));
  }
  lastChange = null;
  shown = null;
  stepCount = 0;
  activeRow = -1;
  nextInfer = 0;
  pendingDelete = -1;
}

function congested() {
  if (hiddenSizes.length > 3) return true;
  for (let i = 0; i < hiddenSizes.length; i++) {
    if (hiddenSizes[i] > 8) return true;
  }
  return false;
}

function valuesHidden() {
  if (hideTouched) return hideValues;
  return congested();
}

function growLayer(index) {
  if (playing || hiddenSizes[index] >= MAX_LAYER_SIZE) return;
  hiddenSizes[index] += 1;
  for (let i = 0; i < weights[index].length; i++) weights[index][i].push(0);
  biases[index].push(0);
  const next = biases[index + 1].length;
  const row = [];
  for (let j = 0; j < next; j++) row.push(0);
  weights[index + 1].push(row);
  lastChange = null;
  render();
}

function shrinkLayer(index) {
  if (playing || hiddenSizes[index] <= 1) return;
  hiddenSizes[index] -= 1;
  for (let i = 0; i < weights[index].length; i++) weights[index][i].pop();
  biases[index].pop();
  weights[index + 1].pop();
  lastChange = null;
  render();
}

function addLayer() {
  if (playing) return;
  const previous = hiddenSizes[hiddenSizes.length - 1];
  const outputBias = biases.pop()[0];
  weights.pop();
  hiddenSizes.push(2);
  weights.push(blankMatrix(previous, 2));
  biases.push(blankBias(2));
  weights.push(blankMatrix(2, 1));
  biases.push([outputBias]);
  lastChange = null;
  shown = null;
  render();
}

function deleteLayer(index) {
  if (playing || hiddenSizes.length <= 1) return;
  const previous = index === 0 ? inputNames.length : hiddenSizes[index - 1];
  const next = index === hiddenSizes.length - 1 ? 1 : hiddenSizes[index + 1];
  const merged = blankMatrix(previous, next);
  const newWeights = weights.slice(0, index);
  newWeights.push(merged);
  for (let i = index + 2; i < weights.length; i++) newWeights.push(weights[i]);
  const newBiases = biases.slice(0, index);
  newBiases.push(biases[index + 1].slice());
  for (let i = index + 2; i < biases.length; i++) newBiases.push(biases[i]);
  hiddenSizes.splice(index, 1);
  weights = newWeights;
  biases = newBiases;
  pendingDelete = -1;
  lastChange = null;
  shown = null;
  render();
}

function askDelete(index) {
  if (playing || hiddenSizes.length <= 1) return;
  pendingDelete = index;
  render();
}

function cancelDelete() {
  pendingDelete = -1;
  render();
}

function forwardPass(values) {
  const activation = [values.slice()];
  const zValues = [];
  for (let layer = 0; layer < weights.length; layer++) {
    const previous = activation[layer];
    const isOutput = layer === weights.length - 1;
    const name = isOutput ? outputActivation() : hiddenActivation();
    const zRow = [];
    const aRow = [];
    for (let unit = 0; unit < biases[layer].length; unit++) {
      let z = biases[layer][unit];
      for (let i = 0; i < previous.length; i++) {
        z += previous[i] * weights[layer][i][unit];
      }
      zRow.push(z);
      aRow.push(activate(name, z));
    }
    zValues.push(zRow);
    activation.push(aRow);
  }
  return { activation: activation, zValues: zValues, output: activation[activation.length - 1][0] };
}

function pullsForRow(values, target) {
  const pass = forwardPass(values);
  const rate = learningRate();
  const last = weights.length - 1;
  const delta = [];
  delta[last] = [rate * errorFactor(outputActivation(), target, pass.output, pass.zValues[last][0])];
  for (let layer = last; layer > 0; layer--) {
    const incoming = pass.activation[layer];
    delta[layer - 1] = [];
    for (let i = 0; i < incoming.length; i++) {
      let sum = 0;
      for (let j = 0; j < delta[layer].length; j++) {
        sum += weights[layer][i][j] * delta[layer][j];
      }
      const slope = hiddenDerivative(hiddenActivation(), incoming[i], pass.zValues[layer - 1][i]);
      delta[layer - 1].push(sum * slope);
    }
  }
  const weightPulls = [];
  const biasPulls = [];
  for (let layer = 0; layer < weights.length; layer++) {
    const previous = pass.activation[layer];
    const matrix = [];
    for (let i = 0; i < previous.length; i++) {
      const row = [];
      for (let j = 0; j < delta[layer].length; j++) row.push(delta[layer][j] * previous[i]);
      matrix.push(row);
    }
    weightPulls.push(matrix);
    biasPulls.push(delta[layer].slice());
  }
  return { weightPulls: weightPulls, biasPulls: biasPulls, pass: pass };
}

function addPull(sumWeights, sumBiases, pull) {
  for (let layer = 0; layer < pull.weightPulls.length; layer++) {
    for (let i = 0; i < pull.weightPulls[layer].length; i++) {
      for (let j = 0; j < pull.weightPulls[layer][i].length; j++) {
        sumWeights[layer][i][j] += pull.weightPulls[layer][i][j];
      }
    }
    for (let j = 0; j < pull.biasPulls[layer].length; j++) {
      sumBiases[layer][j] += pull.biasPulls[layer][j];
    }
  }
}

function zeroPull() {
  const sumWeights = [];
  const sumBiases = [];
  for (let layer = 0; layer < weights.length; layer++) {
    sumWeights.push(blankMatrix(weights[layer].length, biases[layer].length));
    sumBiases.push(blankBias(biases[layer].length));
  }
  return { sumWeights: sumWeights, sumBiases: sumBiases };
}

function applyAverage(sumWeights, sumBiases, count) {
  const oldWeights = [];
  const oldBiases = [];
  const weightDeltas = [];
  const biasDeltas = [];
  for (let layer = 0; layer < weights.length; layer++) {
    oldWeights.push(weights[layer].map(function copyRow(row) { return row.slice(); }));
    oldBiases.push(biases[layer].slice());
    const matrix = [];
    for (let i = 0; i < weights[layer].length; i++) {
      const row = [];
      for (let j = 0; j < weights[layer][i].length; j++) {
        const delta = round2(sumWeights[layer][i][j] / count);
        row.push(delta);
        weights[layer][i][j] = round2(weights[layer][i][j] + delta);
      }
      matrix.push(row);
    }
    const biasRow = [];
    for (let j = 0; j < biases[layer].length; j++) {
      const delta = round2(sumBiases[layer][j] / count);
      biasRow.push(delta);
      biases[layer][j] = round2(biases[layer][j] + delta);
    }
    weightDeltas.push(matrix);
    biasDeltas.push(biasRow);
  }
  lastChange = {
    oldWeights: oldWeights,
    oldBiases: oldBiases,
    weightDeltas: weightDeltas,
    biasDeltas: biasDeltas,
  };
  const lines = [
    "step " + stepCount + ", average of " + count + (count === 1 ? " row" : " rows"),
  ];
  for (let layer = 0; layer < weights.length; layer++) {
    for (let j = 0; j < biases[layer].length; j++) {
      for (let i = 0; i < weights[layer].length; i++) {
        lines.push(
          linkName(layer, i, j) + ": " + formatNum(oldWeights[layer][i][j])
          + " → " + formatNum(weights[layer][i][j])
          + " (" + formatSigned(weightDeltas[layer][i][j]) + ")"
        );
      }
      lines.push(
        biasName(layer, j) + ": " + formatNum(oldBiases[layer][j])
        + " → " + formatNum(biases[layer][j])
        + " (" + formatSigned(biasDeltas[layer][j]) + ")"
      );
    }
  }
  document.getElementById("readout").textContent = lines.join("\n");
}

function sourceName(layer, index) {
  if (layer === 0) return "x" + (index + 1);
  return "h" + layer + "." + (index + 1);
}

function destName(layer, index) {
  if (layer === weights.length - 1) return "y";
  return "h" + (layer + 1) + "." + (index + 1);
}

function linkName(layer, from, to) {
  return destName(layer, to) + " ← " + sourceName(layer, from);
}

function biasName(layer, index) {
  return destName(layer, index) + " bias";
}

function showRow(rowNumber, pass, title) {
  activeRow = rowNumber;
  shown = {
    values: rows[rowNumber].values.slice(),
    activation: pass.activation,
    zValues: pass.zValues,
    output: pass.output,
  };
  const row = rows[rowNumber];
  document.getElementById("readout").textContent = [
    title,
    "output = " + formatNum(pass.output),
    "target = " + formatNum(row.target),
    "error = " + formatNum(row.target - pass.output),
  ].join("\n");
}

function targetsReached() {
  for (let r = 0; r < rows.length; r++) {
    const output = forwardPass(rows[r].values).output;
    if (round2(output) !== round2(rows[r].target)) return false;
  }
  return true;
}

function setPlaying(on) {
  playing = on;
  const ids = [
    "step", "run", "infer", "random-weights", "reset-weights",
    "add-record", "add-input", "remove-input", "add-layer", "steps",
    "output-mode", "hidden-activation", "output-activation", "learning-rate",
  ];
  for (let i = 0; i < ids.length; i++) document.getElementById(ids[i]).disabled = on;
  if (!on) document.getElementById("remove-input").disabled = inputNames.length <= 1;
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
    const totals = zeroPull();
    const countRows = rows.length;
    for (let r = 0; r < countRows; r++) {
      const pull = pullsForRow(rows[r].values, rows[r].target);
      addPull(totals.sumWeights, totals.sumBiases, pull);
      lastChange = null;
      showRow(r, pull.pass, "step " + stepCount + ", row " + (r + 1));
      render();
      await sleep(rowPauseMs());
    }
    applyAverage(totals.sumWeights, totals.sumBiases, countRows);
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
  const rowNumber = nextInfer;
  nextInfer += 1;
  const pass = forwardPass(rows[rowNumber].values);
  lastChange = null;
  showRow(rowNumber, pass, "infer, row " + (rowNumber + 1));
  document.getElementById("readout").textContent += "\nweights stay the same";
  render();
}

function randomWeights() {
  if (playing) return;
  for (let layer = 0; layer < weights.length; layer++) {
    for (let i = 0; i < weights[layer].length; i++) {
      for (let j = 0; j < weights[layer][i].length; j++) {
        weights[layer][i][j] = round2(Math.random() * 2 - 1);
      }
    }
    for (let j = 0; j < biases[layer].length; j++) {
      biases[layer][j] = round2(Math.random() * 2 - 1);
    }
  }
  lastChange = null;
  shown = null;
  render();
  document.getElementById("readout").textContent = "Random weights are set. Press Step to train every row once.";
}

function resetWeights() {
  if (playing) return;
  rebuildWeights();
  render();
  document.getElementById("readout").textContent = "Weights are 0 again. Press Step to train every row once.";
}

function fillActivationSelect(id, selected) {
  const select = document.getElementById(id);
  const previous = selected || select.value;
  select.textContent = "";
  for (let i = 0; i < ACTIVATIONS.length; i++) {
    const option = document.createElement("option");
    option.value = ACTIVATIONS[i].id;
    option.textContent = ACTIVATIONS[i].label;
    select.appendChild(option);
  }
  select.value = previous || (id === "hidden-activation" ? "relu" : "linear");
}

function renderFormula() {
  document.getElementById("formula").textContent =
    "Hidden uses " + document.getElementById("hidden-activation").selectedOptions[0].textContent
    + ". Output uses " + document.getElementById("output-activation").selectedOptions[0].textContent
    + ". One step adds the average pull from every row.";
}

function renderLayers() {
  const body = document.getElementById("layer-body");
  body.textContent = "";
  for (let index = 0; index < hiddenSizes.length; index++) {
    const tr = document.createElement("tr");
    tr.appendChild(textCell(String(index + 1)));
    const size = document.createElement("td");
    const minus = document.createElement("button");
    minus.type = "button";
    minus.className = "secondary";
    minus.textContent = "−";
    minus.disabled = playing || hiddenSizes[index] <= 1;
    minus.addEventListener("click", function onMinus() { shrinkLayer(index); });
    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "secondary";
    plus.textContent = "+";
    plus.disabled = playing || hiddenSizes[index] >= MAX_LAYER_SIZE;
    plus.addEventListener("click", function onPlus() { growLayer(index); });
    const count = document.createElement("span");
    count.textContent = " " + hiddenSizes[index] + " ";
    size.appendChild(minus);
    size.appendChild(count);
    size.appendChild(plus);
    tr.appendChild(size);
    const action = document.createElement("td");
    if (hiddenSizes.length > 1 && pendingDelete === index) {
      action.appendChild(actionButton("Cancel", "secondary", cancelDelete));
      action.appendChild(document.createTextNode(" "));
      action.appendChild(actionButton("Delete", "delete", function onConfirm() { deleteLayer(index); }));
    } else if (hiddenSizes.length > 1) {
      action.appendChild(actionButton("Delete", "delete", function onAsk() { askDelete(index); }));
    }
    tr.appendChild(action);
    body.appendChild(tr);
  }
  const hideButton = document.getElementById("hide-values");
  hideButton.textContent = valuesHidden() ? "Show values" : "Hide values";
}

function actionButton(label, className, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.disabled = playing;
  button.addEventListener("click", onClick);
  return button;
}

function textCell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function renderTable() {
  const head = document.getElementById("data-head");
  const body = document.getElementById("data-body");
  head.textContent = "";
  body.textContent = "";
  const header = document.createElement("tr");
  header.appendChild(textCellHead("row"));
  for (let i = 0; i < inputNames.length; i++) header.appendChild(textCellHead(inputNames[i]));
  header.appendChild(textCellHead("target"));
  header.appendChild(textCellHead("prediction"));
  header.appendChild(textCellHead(""));
  head.appendChild(header);
  for (let r = 0; r < rows.length; r++) {
    const tr = document.createElement("tr");
    if (r === activeRow) tr.className = "active";
    tr.appendChild(textCell(String(r + 1)));
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
    prediction.textContent = formatNum(forwardPass(rows[r].values).output);
    tr.appendChild(prediction);
    const remove = actionButton("Delete", "delete", function onDelete() { deleteRow(r); });
    const action = document.createElement("td");
    action.appendChild(remove);
    tr.appendChild(action);
    body.appendChild(tr);
  }
  if (!playing) document.getElementById("remove-input").disabled = inputNames.length <= 1;
}

function textCellHead(text) {
  const th = document.createElement("th");
  th.textContent = text;
  return th;
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
    cells[r].textContent = formatNum(forwardPass(rows[r].values).output);
  }
}

function diagramScale() {
  let widest = inputNames.length;
  for (let i = 0; i < hiddenSizes.length; i++) widest = Math.max(widest, hiddenSizes[i]);
  const columns = hiddenSizes.length + 2;
  let scale = 1;
  if (widest > 4) scale = Math.max(0.42, 1 - (widest - 4) * 0.07);
  if (columns > 4) scale = Math.min(scale, Math.max(0.42, 1 - (columns - 4) * 0.08));
  return scale;
}

function renderDiagram() {
  const svg = document.getElementById("diagram");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const svgNs = "http://www.w3.org/2000/svg";
  const scale = diagramScale();
  const nodeR = Math.max(12, 26 * scale);
  const biasH = 20 * scale;
  const gapY = nodeR * 2 + biasH + 18 * scale;
  const gapX = Math.max(108, 168 * scale);
  const hidden = valuesHidden();
  const columns = hiddenSizes.length + 2;
  let tallest = inputNames.length;
  for (let i = 0; i < hiddenSizes.length; i++) tallest = Math.max(tallest, hiddenSizes[i]);
  const width = 48 + columns * gapX;
  const height = Math.max(220, 28 + tallest * gapY + 8);
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));

  function columnCount(column) {
    if (column === 0) return inputNames.length;
    if (column === columns - 1) return 1;
    return hiddenSizes[column - 1];
  }
  function xAt(column) { return 48 + column * gapX; }
  function yAt(column, unit) {
    const count = columnCount(column);
    const extra = (tallest - count) * gapY / 2;
    return 36 + extra + unit * gapY + nodeR;
  }

  for (let layer = 0; layer < weights.length; layer++) {
    for (let i = 0; i < weights[layer].length; i++) {
      for (let j = 0; j < weights[layer][i].length; j++) {
        const look = weightLook(weights[layer][i][j]);
        drawLine(svg, svgNs, xAt(layer) + nodeR, yAt(layer, i), xAt(layer + 1) - nodeR, yAt(layer + 1, j), look.thickness * Math.max(0.6, scale), look.color);
      }
    }
  }

  for (let column = 0; column < columns; column++) {
    const title = column === 0 ? "Inputs" : (column === columns - 1 ? "Output" : "Hidden " + column);
    drawText(svg, svgNs, xAt(column), 16, title, "#5c564c", Math.max(11, 13 * scale));
    for (let unit = 0; unit < columnCount(column); unit++) {
      const x = xAt(column);
      const y = yAt(column, unit);
      const label = column === 0 ? "x" + (unit + 1) : (column === columns - 1 ? "y" : "h" + column + "." + (unit + 1));
      const value = nodeValue(column, unit);
      drawNode(svg, svgNs, x, y, nodeR, hidden ? label : label, hidden ? null : value);
      const bias = column === 0 ? null : biases[column - 1][unit];
      if (column > 0) {
        const delta = lastChange && !hidden ? lastChange.biasDeltas[column - 1][unit] : null;
        drawWeightBox(svg, svgNs, x, y + nodeR + biasH * 0.65, scale, "b", bias, delta, hidden);
      }
    }
  }

  if (!hidden && scale > 0.62) {
    for (let layer = 0; layer < weights.length; layer++) {
      if (weights[layer].length * biases[layer].length > 12) continue;
      for (let i = 0; i < weights[layer].length; i++) {
        for (let j = 0; j < weights[layer][i].length; j++) {
          const x = (xAt(layer) + xAt(layer + 1)) / 2;
          const y = (yAt(layer, i) + yAt(layer + 1, j)) / 2;
          const delta = lastChange ? lastChange.weightDeltas[layer][i][j] : null;
          drawWeightBox(svg, svgNs, x, y, scale, "w", weights[layer][i][j], delta, false);
        }
      }
    }
  }
}

function nodeValue(column, unit) {
  if (!shown) return null;
  if (column === 0) return formatNum(shown.values[unit]);
  return formatNum(shown.activation[column][unit]);
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

function drawNode(svg, svgNs, x, y, radius, label, value) {
  const circle = document.createElementNS(svgNs, "circle");
  circle.setAttribute("cx", x);
  circle.setAttribute("cy", y);
  circle.setAttribute("r", radius);
  circle.setAttribute("fill", "#fffdf8");
  circle.setAttribute("stroke", "#1c1915");
  circle.setAttribute("stroke-width", "1.5");
  svg.appendChild(circle);
  const size = Math.max(9, radius * 0.46);
  if (value === null || value === undefined) {
    drawText(svg, svgNs, x, y + size * 0.35, label, "#1c1915", size);
  } else {
    drawText(svg, svgNs, x, y - size * 0.15, label, "#5c564c", size * 0.85);
    drawText(svg, svgNs, x, y + size * 0.95, value, "#1c1915", size);
  }
}

function drawWeightBox(svg, svgNs, x, y, scale, title, weight, delta, nameOnly) {
  const look = weightLook(weight);
  const height = !nameOnly && delta !== null && delta !== undefined ? 34 * scale : 24 * scale;
  const width = Math.max(36, 64 * scale);
  const rect = document.createElementNS(svgNs, "rect");
  rect.setAttribute("x", x - width / 2);
  rect.setAttribute("y", y - height / 2);
  rect.setAttribute("width", width);
  rect.setAttribute("height", height);
  rect.setAttribute("rx", 6);
  rect.setAttribute("fill", "#fffdf8");
  rect.setAttribute("stroke", look.color);
  rect.setAttribute("stroke-width", Math.max(1.25, look.thickness / 2));
  svg.appendChild(rect);
  const size = Math.max(9, 12 * scale);
  const text = nameOnly ? title : title + " " + formatNum(weight);
  drawText(svg, svgNs, x, y + (delta !== null && delta !== undefined && !nameOnly ? -2 : size * 0.3), text, "#1c1915", size);
  if (!nameOnly && delta !== null && delta !== undefined) {
    drawText(svg, svgNs, x, y + size, formatSigned(delta), delta < 0 ? "#9a3412" : "#0f6e56", size);
  }
}

function drawText(svg, svgNs, x, y, label, color, size) {
  const text = document.createElementNS(svgNs, "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("font-size", size);
  text.setAttribute("font-family", "ui-sans-serif, system-ui, sans-serif");
  text.setAttribute("fill", color);
  text.textContent = label;
  svg.appendChild(text);
}

function render() {
  renderFormula();
  renderLayers();
  renderTable();
  renderDiagram();
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
  for (let r = 0; r < rows.length; r++) rows[r].values.push(0);
  if (weights[0]) {
    const row = [];
    for (let j = 0; j < biases[0].length; j++) row.push(0);
    weights[0].push(row);
  }
  lastChange = null;
  render();
}

function removeInput() {
  if (playing || inputNames.length <= 1) return;
  inputNames.pop();
  for (let r = 0; r < rows.length; r++) rows[r].values.pop();
  if (weights[0]) weights[0].pop();
  lastChange = null;
  render();
}

async function loadProvidedDataset() {
  const file = outputMode() === "boolean"
    ? "data/perceptron_boolean.json"
    : "data/perceptron_variable.json";
  const response = await fetch(file);
  if (!response.ok) throw new Error("Could not load " + file);
  const data = await response.json();
  inputNames = data.inputs.slice();
  rows = [];
  for (let r = 0; r < data.rows.length; r++) {
    rows.push({ values: data.rows[r].values.slice(), target: data.rows[r].target });
  }
  rebuildWeights();
  render();
  document.getElementById("readout").textContent = "Weights start at 0. Press Step to train every row once.";
}

function toggleValues() {
  hideValues = !valuesHidden();
  hideTouched = true;
  render();
}

document.addEventListener("DOMContentLoaded", function start() {
  fillActivationSelect("hidden-activation", "relu");
  fillActivationSelect("output-activation", "linear");
  document.getElementById("output-mode").addEventListener("change", loadProvidedDataset);
  document.getElementById("hidden-activation").addEventListener("change", render);
  document.getElementById("output-activation").addEventListener("change", render);
  document.getElementById("learning-rate").addEventListener("change", render);
  document.getElementById("add-layer").addEventListener("click", addLayer);
  document.getElementById("add-record").addEventListener("click", addRecord);
  document.getElementById("add-input").addEventListener("click", addInput);
  document.getElementById("remove-input").addEventListener("click", removeInput);
  document.getElementById("step").addEventListener("click", function onStep() { runPasses(1, false); });
  document.getElementById("run").addEventListener("click", function onRun() { runPasses(requestedPasses(), true); });
  document.getElementById("infer").addEventListener("click", inferOne);
  document.getElementById("random-weights").addEventListener("click", randomWeights);
  document.getElementById("reset-weights").addEventListener("click", resetWeights);
  document.getElementById("hide-values").addEventListener("click", toggleValues);
  loadProvidedDataset().catch(function onLoadError(error) {
    document.getElementById("readout").textContent = error.message;
  });
});
