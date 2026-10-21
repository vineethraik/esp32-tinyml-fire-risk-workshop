// Workshop-shaped thermal-risk trainer in the browser.
//
// 20 inputs -> 12 hidden (ReLU) -> 3 Softmax classes.
// One step visits every window. Average on: mean pull. Average off: update after each window.
// Softmax + cross-entropy: d_logits = probabilities - one_hot(label).
// Stored weights keep full precision. Scores are training-set only.

const INPUT_SIZE = 20;
const HIDDEN_SIZE = 12;
const OUTPUT_SIZE = 3;
const LABELS = ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"];
const SHORT = ["NORMAL", "ELEVATED", "HIGH"];

let rawFeatures = [];
let labels = [];
let meta = [];
let mean = [];
let scale = [];
let X = [];
let w1 = [];
let b1 = [];
let w2 = [];
let b2 = [];
let stepCount = 0;
let activeRow = -1;
let nextInfer = 0;
let lastProbs = null;
let lastChangeNote = "";
let lastChange = null;
let shown = null;
let playing = false;
let stopRequested = false;
let hideValues = false;
let hideTouched = false;

function sleep(ms) {
  return new Promise(function wake(resolve) {
    setTimeout(resolve, ms);
  });
}

function rowPauseMs() {
  const pauses = [600, 300, 200, 120, 80, 50, 30, 18, 10, 5];
  const speed = Number(document.getElementById("speed").value);
  const level = Number.isFinite(speed) ? Math.round(speed) : 3;
  if (level < 1) return pauses[0];
  if (level > pauses.length) return pauses[pauses.length - 1];
  return pauses[level - 1];
}

function learningRate() {
  const value = Number(document.getElementById("learning-rate").value);
  if (!Number.isFinite(value)) return 0.05;
  return value;
}

function requestedPasses() {
  const value = Math.floor(Number(document.getElementById("steps").value));
  if (!Number.isFinite(value) || value < 1) return 1;
  return value;
}

function averageUpdates() {
  return document.getElementById("average-update").checked;
}

function formatNum(value) {
  const rounded = Math.round(value * 100) / 100;
  return rounded.toFixed(2);
}

function formatPct(value) {
  return (Math.round(value * 1000) / 10).toFixed(1) + "%";
}

function blankMatrix(rows, cols) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) row.push(0);
    matrix.push(row);
  }
  return matrix;
}

function blankBias(count) {
  const bias = [];
  for (let i = 0; i < count; i++) bias.push(0);
  return bias;
}

function copyMatrix(matrix) {
  const out = [];
  for (let i = 0; i < matrix.length; i++) out.push(matrix[i].slice());
  return out;
}

function softmax(logits) {
  let max = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(logits[i] - max);
    exps.push(value);
    sum += value;
  }
  const probs = [];
  for (let i = 0; i < exps.length; i++) probs.push(exps[i] / sum);
  return probs;
}

function normalizeFeatures() {
  mean = [];
  scale = [];
  for (let f = 0; f < INPUT_SIZE; f++) {
    let sum = 0;
    for (let r = 0; r < rawFeatures.length; r++) sum += rawFeatures[r][f];
    const m = sum / rawFeatures.length;
    mean.push(m);
    let sq = 0;
    for (let r = 0; r < rawFeatures.length; r++) {
      const d = rawFeatures[r][f] - m;
      sq += d * d;
    }
    let s = Math.sqrt(sq / rawFeatures.length);
    if (s === 0) s = 1;
    scale.push(s);
  }
  X = [];
  for (let r = 0; r < rawFeatures.length; r++) {
    const row = [];
    for (let f = 0; f < INPUT_SIZE; f++) {
      row.push((rawFeatures[r][f] - mean[f]) / scale[f]);
    }
    X.push(row);
  }
}

function zeroWeights() {
  w1 = blankMatrix(INPUT_SIZE, HIDDEN_SIZE);
  b1 = blankBias(HIDDEN_SIZE);
  w2 = blankMatrix(HIDDEN_SIZE, OUTPUT_SIZE);
  b2 = blankBias(OUTPUT_SIZE);
  stepCount = 0;
  lastChangeNote = "Weights are 0.";
  lastChange = null;
  lastProbs = null;
  shown = null;
  activeRow = -1;
  nextInfer = 0;
}

function randomWeights() {
  if (playing) return;
  // He-style scale for ReLU: std ~= sqrt(2 / fan_in)
  const s1 = Math.sqrt(2 / INPUT_SIZE);
  const s2 = Math.sqrt(2 / HIDDEN_SIZE);
  w1 = blankMatrix(INPUT_SIZE, HIDDEN_SIZE);
  b1 = blankBias(HIDDEN_SIZE);
  w2 = blankMatrix(HIDDEN_SIZE, OUTPUT_SIZE);
  b2 = blankBias(OUTPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE; i++) {
    for (let j = 0; j < HIDDEN_SIZE; j++) w1[i][j] = (Math.random() * 2 - 1) * s1;
  }
  for (let i = 0; i < HIDDEN_SIZE; i++) {
    for (let j = 0; j < OUTPUT_SIZE; j++) w2[i][j] = (Math.random() * 2 - 1) * s2;
  }
  stepCount = 0;
  lastChangeNote = "Random He-style weights set.";
  lastChange = null;
  lastProbs = null;
  shown = null;
  render();
  document.getElementById("readout").textContent = "Random weights are set. Press Step to train every window once.";
}

function resetWeights() {
  if (playing) return;
  zeroWeights();
  render();
  document.getElementById("readout").textContent = "Weights are 0 again. Press Random weights, then Step.";
}

function forward(x) {
  const z1 = [];
  const h = [];
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    let z = b1[j];
    for (let i = 0; i < INPUT_SIZE; i++) z += x[i] * w1[i][j];
    z1.push(z);
    h.push(z > 0 ? z : 0);
  }
  const logits = [];
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    let z = b2[k];
    for (let j = 0; j < HIDDEN_SIZE; j++) z += h[j] * w2[j][k];
    logits.push(z);
  }
  const probs = softmax(logits);
  let pred = 0;
  for (let k = 1; k < OUTPUT_SIZE; k++) {
    if (probs[k] > probs[pred]) pred = k;
  }
  return { z1: z1, h: h, logits: logits, probs: probs, pred: pred };
}

function pullsForRow(rowIndex) {
  const x = X[rowIndex];
  const y = labels[rowIndex];
  const pass = forward(x);
  const rate = learningRate();
  // Softmax + cross-entropy pull (add to weights): one_hot - probabilities.
  const dLogits = [];
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    dLogits.push(rate * ((k === y ? 1 : 0) - pass.probs[k]));
  }
  const dw2 = blankMatrix(HIDDEN_SIZE, OUTPUT_SIZE);
  const db2 = blankBias(OUTPUT_SIZE);
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    db2[k] = dLogits[k];
    for (let j = 0; j < HIDDEN_SIZE; j++) dw2[j][k] = dLogits[k] * pass.h[j];
  }
  const dHidden = blankBias(HIDDEN_SIZE);
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    let sum = 0;
    for (let k = 0; k < OUTPUT_SIZE; k++) sum += w2[j][k] * dLogits[k];
    dHidden[j] = pass.z1[j] > 0 ? sum : 0;
  }
  const dw1 = blankMatrix(INPUT_SIZE, HIDDEN_SIZE);
  const db1 = blankBias(HIDDEN_SIZE);
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    db1[j] = dHidden[j];
    for (let i = 0; i < INPUT_SIZE; i++) dw1[i][j] = dHidden[j] * x[i];
  }
  return { dw1: dw1, db1: db1, dw2: dw2, db2: db2, pass: pass };
}

function zeroPull() {
  return {
    dw1: blankMatrix(INPUT_SIZE, HIDDEN_SIZE),
    db1: blankBias(HIDDEN_SIZE),
    dw2: blankMatrix(HIDDEN_SIZE, OUTPUT_SIZE),
    db2: blankBias(OUTPUT_SIZE),
  };
}

function addPull(totals, pull) {
  for (let i = 0; i < INPUT_SIZE; i++) {
    for (let j = 0; j < HIDDEN_SIZE; j++) totals.dw1[i][j] += pull.dw1[i][j];
  }
  for (let j = 0; j < HIDDEN_SIZE; j++) totals.db1[j] += pull.db1[j];
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    for (let k = 0; k < OUTPUT_SIZE; k++) totals.dw2[j][k] += pull.dw2[j][k];
  }
  for (let k = 0; k < OUTPUT_SIZE; k++) totals.db2[k] += pull.db2[k];
}

function matrixNorm(matrix) {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) sum += matrix[i][j] * matrix[i][j];
  }
  return Math.sqrt(sum);
}

function biasNorm(bias) {
  let sum = 0;
  for (let i = 0; i < bias.length; i++) sum += bias[i] * bias[i];
  return Math.sqrt(sum);
}

function applyPull(totals, count, title) {
  const oldW1 = copyMatrix(w1);
  const oldB1 = b1.slice();
  const oldW2 = copyMatrix(w2);
  const oldB2 = b2.slice();
  for (let i = 0; i < INPUT_SIZE; i++) {
    for (let j = 0; j < HIDDEN_SIZE; j++) w1[i][j] += totals.dw1[i][j] / count;
  }
  for (let j = 0; j < HIDDEN_SIZE; j++) b1[j] += totals.db1[j] / count;
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    for (let k = 0; k < OUTPUT_SIZE; k++) w2[j][k] += totals.dw2[j][k] / count;
  }
  for (let k = 0; k < OUTPUT_SIZE; k++) b2[k] += totals.db2[k] / count;

  const dw1 = blankMatrix(INPUT_SIZE, HIDDEN_SIZE);
  const dw2 = blankMatrix(HIDDEN_SIZE, OUTPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE; i++) {
    for (let j = 0; j < HIDDEN_SIZE; j++) dw1[i][j] = w1[i][j] - oldW1[i][j];
  }
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    for (let k = 0; k < OUTPUT_SIZE; k++) dw2[j][k] = w2[j][k] - oldW2[j][k];
  }
  const db1 = [];
  const db2 = [];
  for (let j = 0; j < HIDDEN_SIZE; j++) db1.push(b1[j] - oldB1[j]);
  for (let k = 0; k < OUTPUT_SIZE; k++) db2.push(b2[k] - oldB2[k]);
  lastChange = {
    dw1: dw1,
    db1: db1,
    dw2: dw2,
    db2: db2,
  };
  lastChangeNote = (title || ("step " + stepCount))
    + "\n|ΔW1| = " + formatNum(matrixNorm(dw1))
    + "  |Δb1| = " + formatNum(biasNorm(db1))
    + "\n|ΔW2| = " + formatNum(matrixNorm(dw2))
    + "  |Δb2| = " + formatNum(biasNorm(db2));
}

function rememberShown(rowIndex, pass) {
  shown = {
    values: X[rowIndex].slice(),
    h: pass.h.slice(),
    probs: pass.probs.slice(),
  };
  lastProbs = pass.probs;
}

function scores() {
  let correct = 0;
  const confusion = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const guesses = [];
  for (let r = 0; r < X.length; r++) {
    const pass = forward(X[r]);
    guesses.push(pass.pred);
    confusion[labels[r]][pass.pred] += 1;
    if (pass.pred === labels[r]) correct += 1;
  }
  return {
    accuracy: X.length === 0 ? 0 : correct / X.length,
    confusion: confusion,
    guesses: guesses,
  };
}

function perfectAccuracy() {
  return scores().accuracy >= 1;
}

function updatePlayLocks() {
  const ids = [
    "step", "run", "infer", "random-weights", "reset-weights",
    "steps", "learning-rate", "average-update",
  ];
  for (let i = 0; i < ids.length; i++) document.getElementById(ids[i]).disabled = playing;
  const stop = document.getElementById("stop-run");
  if (playing) {
    stop.removeAttribute("disabled");
    stop.disabled = false;
  } else {
    stop.disabled = true;
  }
}

function setPlaying(on) {
  playing = on;
  if (!on) stopRequested = false;
  updatePlayLocks();
}

function requestStop() {
  if (!playing) return;
  stopRequested = true;
}

async function runPasses(count, stopWhenReached) {
  if (playing) return;
  if (X.length === 0) {
    document.getElementById("readout").textContent = "No windows loaded.";
    return;
  }
  if (stopWhenReached && perfectAccuracy()) {
    document.getElementById("readout").textContent = "Training accuracy already 100%. Run stopped.";
    return;
  }
  stopRequested = false;
  setPlaying(true);
  let stopped = false;
  for (let pass = 0; pass < count; pass++) {
    if (stopRequested) {
      stopped = true;
      break;
    }
    stepCount += 1;
    const average = averageUpdates();
    const totals = zeroPull();
    let finishedPass = true;
    let appliedRow = false;
    for (let r = 0; r < X.length; r++) {
      if (stopRequested) {
        finishedPass = false;
        stopped = true;
        break;
      }
      const pull = pullsForRow(r);
      activeRow = r;
      rememberShown(r, pull.pass);
      if (average) {
        addPull(totals, pull);
        lastChange = null;
        lastChangeNote = "step " + stepCount + ", window " + (r + 1);
      } else {
        applyPull(pull, 1, "step " + stepCount + ", window " + (r + 1));
        appliedRow = true;
      }
      render();
      await sleep(rowPauseMs());
    }
    if (!finishedPass) {
      if (average || !appliedRow) stepCount -= 1;
      break;
    }
    if (average) applyPull(totals, X.length, "step " + stepCount + ", average of " + X.length + " windows");
    render();
    if (stopWhenReached && perfectAccuracy()) {
      document.getElementById("readout").textContent += "\ntraining accuracy 100%, run stopped";
      await sleep(rowPauseMs());
      break;
    }
    if (stopRequested) {
      stopped = true;
      break;
    }
    await sleep(rowPauseMs());
  }
  setPlaying(false);
  render();
  if (stopped) {
    document.getElementById("readout").textContent += "\nrun stopped";
  }
}

function inferOne() {
  if (playing) return;
  if (X.length === 0) return;
  if (nextInfer >= X.length) nextInfer = 0;
  const row = nextInfer;
  nextInfer += 1;
  activeRow = row;
  const pass = forward(X[row]);
  lastChange = null;
  rememberShown(row, pass);
  lastChangeNote = "infer, window " + (row + 1) + "\nweights stay the same"
    + "\ntrue = " + LABELS[labels[row]]
    + "\nguess = " + LABELS[pass.pred];
  render();
}

function congested() {
  return INPUT_SIZE > 8 || HIDDEN_SIZE > 8 || OUTPUT_SIZE > 8;
}

function valuesHidden() {
  if (hideTouched) return hideValues;
  return congested();
}

function toggleValues() {
  hideValues = !valuesHidden();
  hideTouched = true;
  render();
}

function weightLook(weight) {
  const thickness = 1.5 + Math.min(6.5, Math.abs(weight) * 4);
  let color = "#1c1915";
  if (weight > 0.005) color = "#0f6e56";
  else if (weight < -0.005) color = "#9a3412";
  return { thickness: thickness, color: color };
}

function diagramScale() {
  const widest = Math.max(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
  let scale = 1;
  if (widest > 4) scale = Math.max(0.38, 1 - (widest - 4) * 0.045);
  return scale;
}

function renderDiagram() {
  const svg = document.getElementById("diagram");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const svgNs = "http://www.w3.org/2000/svg";
  const scale = diagramScale();
  const nodeR = Math.max(10, 22 * scale);
  const biasH = 18 * scale;
  const gapY = nodeR * 2 + biasH + 14 * scale;
  const hidden = valuesHidden();
  const columns = 3;
  const tallest = Math.max(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
  // Stretch columns across the frame. Revert the wide-diagram commit to restore fixed gapX.
  const available = diagramFrameWidth(svg, 520);
  const minGap = Math.max(120, 150 * scale);
  const gapX = Math.max(minGap, (available - 48) / columns);
  const width = Math.max(available, 48 + columns * gapX);
  const height = Math.max(220, 28 + tallest * gapY + 8);
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));

  function columnCount(column) {
    if (column === 0) return INPUT_SIZE;
    if (column === 1) return HIDDEN_SIZE;
    return OUTPUT_SIZE;
  }
  function xAt(column) { return 48 + column * gapX; }
  function yAt(column, unit) {
    const count = columnCount(column);
    const extra = (tallest - count) * gapY / 2;
    return 36 + extra + unit * gapY + nodeR;
  }

  for (let i = 0; i < INPUT_SIZE; i++) {
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      const look = weightLook(w1[i][j]);
      drawLine(svg, svgNs, xAt(0) + nodeR, yAt(0, i), xAt(1) - nodeR, yAt(1, j), look.thickness * Math.max(0.5, scale), look.color);
    }
  }
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    for (let k = 0; k < OUTPUT_SIZE; k++) {
      const look = weightLook(w2[j][k]);
      drawLine(svg, svgNs, xAt(1) + nodeR, yAt(1, j), xAt(2) - nodeR, yAt(2, k), look.thickness * Math.max(0.5, scale), look.color);
    }
  }

  const titles = ["Inputs", "Hidden ReLU", "Output Softmax"];
  for (let column = 0; column < columns; column++) {
    drawText(svg, svgNs, xAt(column), 16, titles[column], "#5c564c", Math.max(10, 12 * scale));
    for (let unit = 0; unit < columnCount(column); unit++) {
      const x = xAt(column);
      const y = yAt(column, unit);
      let label = "x" + (unit + 1);
      if (column === 1) label = "h" + (unit + 1);
      if (column === 2) label = SHORT[unit];
      const value = nodeValue(column, unit);
      drawNode(svg, svgNs, x, y, nodeR, label, hidden ? null : value);
      if (column === 1) {
        const delta = lastChange && !hidden ? lastChange.db1[unit] : null;
        drawWeightBox(svg, svgNs, x, y + nodeR + biasH * 0.65, scale, "b", b1[unit], delta, hidden);
      }
      if (column === 2) {
        const delta = lastChange && !hidden ? lastChange.db2[unit] : null;
        drawWeightBox(svg, svgNs, x, y + nodeR + biasH * 0.65, scale, "b", b2[unit], delta, hidden);
      }
    }
  }

  if (!hidden) {
    drawWeightBoxes(svg, svgNs, scale, nodeR, xAt, yAt, 0, 1, w1, lastChange ? lastChange.dw1 : null);
    drawWeightBoxes(svg, svgNs, scale, nodeR, xAt, yAt, 1, 2, w2, lastChange ? lastChange.dw2 : null);
  }

  const hideButton = document.getElementById("hide-values");
  hideButton.textContent = valuesHidden() ? "Show values" : "Hide values";
}

function drawWeightBoxes(svg, svgNs, scale, nodeR, xAt, yAt, fromCol, toCol, matrix, deltas) {
  const fromCount = matrix.length;
  const toCount = matrix[0].length;
  const edgeTotal = fromCount * toCount;
  if (edgeTotal > 48) return;
  for (let i = 0; i < fromCount; i++) {
    for (let j = 0; j < toCount; j++) {
      const x1 = xAt(fromCol) + nodeR;
      const y1 = yAt(fromCol, i);
      const x2 = xAt(toCol) - nodeR;
      const y2 = yAt(toCol, j);
      const edgeIndex = i * toCount + j;
      const t = edgeTotal <= 1 ? 0.5 : 0.28 + 0.44 * (edgeIndex / (edgeTotal - 1));
      let x = x1 + (x2 - x1) * t;
      let y = y1 + (y2 - y1) * t;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const side = edgeIndex % 2 === 0 ? 1 : -1;
      const nudge = 6 * scale * side;
      x += (-dy / len) * nudge;
      y += (dx / len) * nudge;
      const delta = deltas ? deltas[i][j] : null;
      drawWeightBox(svg, svgNs, x, y, scale, "w", matrix[i][j], delta, false);
    }
  }
}

function nodeValue(column, unit) {
  if (!shown) return null;
  if (column === 0) return formatNum(shown.values[unit]);
  if (column === 1) return formatNum(shown.h[unit]);
  return formatNum(shown.probs[unit]);
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
  const size = Math.max(8, radius * 0.42);
  if (value === null || value === undefined) {
    drawText(svg, svgNs, x, y + size * 0.35, label, "#1c1915", size);
  } else {
    drawText(svg, svgNs, x, y - size * 0.15, label, "#5c564c", size * 0.8);
    drawText(svg, svgNs, x, y + size * 0.95, value, "#1c1915", size);
  }
}

function drawWeightBox(svg, svgNs, x, y, scale, title, weight, delta, nameOnly) {
  const look = weightLook(weight);
  const height = !nameOnly && delta !== null && delta !== undefined ? 32 * scale : 22 * scale;
  const width = Math.max(32, 56 * scale);
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
  const size = Math.max(8, 11 * scale);
  const text = nameOnly ? title : title + " " + formatNum(weight);
  drawText(svg, svgNs, x, y + (delta !== null && delta !== undefined && !nameOnly ? -2 : size * 0.3), text, "#1c1915", size);
  if (!nameOnly && delta !== null && delta !== undefined) {
    drawText(svg, svgNs, x, y + size, formatNum(delta), delta < 0 ? "#9a3412" : "#0f6e56", size);
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

function renderWindows() {
  const body = document.getElementById("windows-body");
  body.textContent = "";
  const score = scores();
  document.getElementById("windows-meta").textContent =
    X.length + " windows  ·  features normalized with training mean/scale";
  for (let r = 0; r < meta.length; r++) {
    const tr = document.createElement("tr");
    if (r === activeRow) tr.className = "active";
    const cells = [
      String(meta[r].id),
      meta[r].start_index + "–" + meta[r].end_index,
      formatNum(meta[r].last_temp_c),
      formatNum(meta[r].temp_change_c),
      LABELS[labels[r]],
      SHORT[score.guesses[r]],
    ];
    for (let c = 0; c < cells.length; c++) {
      const td = document.createElement("td");
      td.textContent = cells[c];
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function renderScores() {
  const score = scores();
  document.getElementById("accuracy").textContent = formatPct(score.accuracy);
  document.getElementById("epoch").textContent = String(stepCount);

  const table = document.getElementById("confusion");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.textContent = "";
  tbody.textContent = "";
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th")).textContent = "true \\ pred";
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    const th = document.createElement("th");
    th.textContent = SHORT[k];
    head.appendChild(th);
  }
  thead.appendChild(head);
  for (let i = 0; i < OUTPUT_SIZE; i++) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = SHORT[i];
    tr.appendChild(th);
    for (let j = 0; j < OUTPUT_SIZE; j++) {
      const td = document.createElement("td");
      td.textContent = String(score.confusion[i][j]);
      if (i === j) td.className = "diag";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  const probs = document.getElementById("probs");
  probs.textContent = "";
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    const cell = document.createElement("div");
    cell.className = "prob-cell";
    const value = lastProbs ? lastProbs[k] : 0;
    const name = document.createElement("strong");
    name.textContent = SHORT[k];
    const span = document.createElement("span");
    span.textContent = lastProbs ? formatPct(value) : "—";
    cell.appendChild(name);
    cell.appendChild(span);
    probs.appendChild(cell);
  }

  document.getElementById("readout").textContent = lastChangeNote
    || "Weights start at 0. Press Random weights, then Step.";
}

function render() {
  renderDiagram();
  renderWindows();
  renderScores();
  updatePlayLocks();
}

async function loadWindows() {
  const response = await fetch("data/thermal_windows.json");
  if (!response.ok) throw new Error("Could not load data/thermal_windows.json");
  const data = await response.json();
  rawFeatures = [];
  labels = [];
  meta = [];
  for (let i = 0; i < data.windows.length; i++) {
    const row = data.windows[i];
    if (!row.features || row.features.length !== INPUT_SIZE) {
      throw new Error("Window " + (i + 1) + " must have " + INPUT_SIZE + " features");
    }
    const labelIndex = LABELS.indexOf(row.label);
    if (labelIndex < 0) throw new Error("Unknown label: " + row.label);
    rawFeatures.push(row.features.slice());
    labels.push(labelIndex);
    meta.push({
      id: row.id || (i + 1),
      start_index: row.start_index,
      end_index: row.end_index,
      last_temp_c: row.last_temp_c,
      temp_change_c: row.temp_change_c,
    });
  }
  normalizeFeatures();
  zeroWeights();
  render();
  document.getElementById("readout").textContent =
    "Loaded " + X.length + " windows. Press Random weights, then Step.";
}

document.addEventListener("DOMContentLoaded", function start() {
  document.getElementById("step").addEventListener("click", function onStep() {
    runPasses(1, false);
  });
  document.getElementById("run").addEventListener("click", function onRun() {
    runPasses(requestedPasses(), true);
  });
  document.getElementById("stop-run").addEventListener("click", requestStop);
  document.getElementById("infer").addEventListener("click", inferOne);
  document.getElementById("random-weights").addEventListener("click", randomWeights);
  document.getElementById("reset-weights").addEventListener("click", resetWeights);
  document.getElementById("hide-values").addEventListener("click", toggleValues);
  document.getElementById("average-update").addEventListener("change", function onAvg() {
    render();
  });
  watchDiagramResize(render);
  loadWindows().catch(function onError(error) {
    document.getElementById("readout").textContent = error.message;
  });
});
