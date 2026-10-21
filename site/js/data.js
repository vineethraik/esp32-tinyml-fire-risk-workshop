// Data lesson: raw readings -> 10-sample window -> 20 features -> label -> time split.
//
// Sample every sampleSeconds. One window is ten consecutive readings.
// Feature order matches the workshop: temp0, hum0, temp1, hum1, ..., temp9, hum9.
// Window starts begin at 1 so readings 0..9 are a warm-up, like prepare_labeling_windows.py.
// Labels stay in page memory only. No model training on this page.

const WINDOW_SAMPLES = 10;
const LABEL_NAMES = ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"];

let readings = [];
let sampleSeconds = 2;
let windowStart = 1;
let labelsByStart = {};

function formatNum(value) {
  const rounded = Math.round(value * 100) / 100;
  return rounded.toFixed(2);
}

function maxStart() {
  return readings.length - WINDOW_SAMPLES;
}

function possibleStarts() {
  const starts = [];
  // Start at 1: skip the first complete window as warm-up.
  for (let start = 1; start <= maxStart(); start++) starts.push(start);
  return starts;
}

function splitAt() {
  const starts = possibleStarts();
  if (starts.length === 0) return 1;
  return starts[0] + Math.floor(starts.length * 0.6);
}

function windowAt(start) {
  const rows = [];
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    const reading = readings[start + i];
    rows.push({
      index: start + i,
      time_s: (start + i) * sampleSeconds,
      temp_c: reading.temp_c,
      humidity_pct: reading.humidity_pct,
    });
  }
  return rows;
}

function featureVector(windowRows) {
  const features = [];
  for (let i = 0; i < windowRows.length; i++) {
    features.push(windowRows[i].temp_c);
    features.push(windowRows[i].humidity_pct);
  }
  return features;
}

function tempChange(windowRows) {
  return windowRows[windowRows.length - 1].temp_c - windowRows[0].temp_c;
}

function halfName(start) {
  return start < splitAt() ? "early" : "late";
}

function clampWindowStart(start) {
  const starts = possibleStarts();
  if (starts.length === 0) return 1;
  if (start < starts[0]) return starts[0];
  if (start > starts[starts.length - 1]) return starts[starts.length - 1];
  return start;
}

function renderSeriesMeta() {
  const end = windowStart + WINDOW_SAMPLES - 1;
  document.getElementById("series-meta").textContent =
    "Reading " + windowStart + "–" + end
    + "  ·  time " + (windowStart * sampleSeconds) + "–" + (end * sampleSeconds) + " s"
    + "  ·  " + readings.length + " readings total"
    + "  ·  sample every " + sampleSeconds + " s";
}

function renderWindow() {
  const rows = windowAt(windowStart);
  const body = document.getElementById("window-body");
  body.textContent = "";
  for (let i = 0; i < rows.length; i++) {
    const tr = document.createElement("tr");
    const cells = [
      String(rows[i].index),
      String(rows[i].time_s),
      formatNum(rows[i].temp_c),
      formatNum(rows[i].humidity_pct),
    ];
    for (let c = 0; c < cells.length; c++) {
      const td = document.createElement("td");
      td.textContent = cells[c];
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  document.getElementById("window-meta").textContent =
    "Δtemp = " + formatNum(tempChange(rows)) + " °C"
    + "  ·  features = " + featureVector(rows).length
    + "  ·  half = " + halfName(windowStart);
  const current = labelsByStart[windowStart];
  if (current) document.getElementById("label-choice").value = current;
  renderFeatures(rows);
}

function renderFeatures(windowRows) {
  const features = featureVector(windowRows);
  const box = document.getElementById("feature-vector");
  box.textContent = "";
  for (let i = 0; i < features.length; i++) {
    const cell = document.createElement("span");
    cell.className = "feature-cell";
    const name = i % 2 === 0
      ? "t" + Math.floor(i / 2)
      : "h" + Math.floor(i / 2);
    cell.textContent = name + " " + formatNum(features[i]);
    box.appendChild(cell);
  }
}

function renderLabels() {
  const body = document.getElementById("label-body");
  body.textContent = "";
  const starts = Object.keys(labelsByStart).map(Number).sort(function byStart(a, b) { return a - b; });
  for (let i = 0; i < starts.length; i++) {
    const start = starts[i];
    const rows = windowAt(start);
    const end = start + WINDOW_SAMPLES - 1;
    const tr = document.createElement("tr");
    if (start === windowStart) tr.className = "active";
    const cells = [
      String(i + 1),
      start + "–" + end,
      formatNum(tempChange(rows)),
      labelsByStart[start],
      halfName(start),
    ];
    for (let c = 0; c < cells.length; c++) {
      const td = document.createElement("td");
      td.textContent = cells[c];
      tr.appendChild(td);
    }
    const action = document.createElement("td");
    const del = document.createElement("button");
    del.type = "button";
    del.className = "delete";
    del.textContent = "Delete";
    del.addEventListener("click", function onDelete() {
      delete labelsByStart[start];
      render();
    });
    action.appendChild(del);
    tr.appendChild(action);
    body.appendChild(tr);
  }
}

function renderSplit() {
  const starts = possibleStarts();
  const cut = splitAt();
  let earlyLabeled = 0;
  let lateLabeled = 0;
  const keys = Object.keys(labelsByStart);
  for (let i = 0; i < keys.length; i++) {
    const start = Number(keys[i]);
    if (start < cut) earlyLabeled += 1;
    else lateLabeled += 1;
  }
  const earlySlots = [];
  const lateSlots = [];
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] < cut) earlySlots.push(starts[i]);
    else lateSlots.push(starts[i]);
  }
  document.getElementById("split-formula").textContent =
    "Keep time order. First ~60% of window starts are early (train-like)."
    + " The rest are late (hold-out-like). Do not shuffle a time series."
    + " Split starts at window start " + cut + ".";
  document.getElementById("split-counts").textContent =
    "Early slots " + earlySlots.length + " (labeled " + earlyLabeled + ")"
    + "  ·  Late slots " + lateSlots.length + " (labeled " + lateLabeled + ")";
  document.getElementById("readout").textContent = [
    "window start = " + windowStart,
    "warm-up readings = 0.." + (WINDOW_SAMPLES - 1),
    "feature count = " + (WINDOW_SAMPLES * 2),
    "labeled windows = " + keys.length,
    "early labeled = " + earlyLabeled,
    "late labeled = " + lateLabeled,
  ].join("\n");
}

function renderChart() {
  const svg = document.getElementById("series-chart");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (readings.length === 0) return;
  const svgNs = "http://www.w3.org/2000/svg";
  const width = Math.max(640, 28 + readings.length * 10);
  const height = 260;
  const left = 44;
  const right = width - 16;
  const top = 18;
  const bottom = height - 28;
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));

  let minTemp = readings[0].temp_c;
  let maxTemp = readings[0].temp_c;
  let minHum = readings[0].humidity_pct;
  let maxHum = readings[0].humidity_pct;
  for (let i = 1; i < readings.length; i++) {
    minTemp = Math.min(minTemp, readings[i].temp_c);
    maxTemp = Math.max(maxTemp, readings[i].temp_c);
    minHum = Math.min(minHum, readings[i].humidity_pct);
    maxHum = Math.max(maxHum, readings[i].humidity_pct);
  }
  if (maxTemp === minTemp) maxTemp = minTemp + 1;
  if (maxHum === minHum) maxHum = minHum + 1;

  function xAt(index) {
    return left + (right - left) * (index / Math.max(1, readings.length - 1));
  }
  function yTemp(value) {
    return bottom - (bottom - top) * ((value - minTemp) / (maxTemp - minTemp));
  }
  function yHum(value) {
    return bottom - (bottom - top) * ((value - minHum) / (maxHum - minHum));
  }

  const cut = splitAt();
  const splitX = xAt(cut);
  const splitLine = document.createElementNS(svgNs, "line");
  splitLine.setAttribute("x1", splitX);
  splitLine.setAttribute("x2", splitX);
  splitLine.setAttribute("y1", top);
  splitLine.setAttribute("y2", bottom);
  splitLine.setAttribute("stroke", "#9a3412");
  splitLine.setAttribute("stroke-width", "1.5");
  splitLine.setAttribute("stroke-dasharray", "4 3");
  svg.appendChild(splitLine);

  const winX1 = xAt(windowStart);
  const winX2 = xAt(windowStart + WINDOW_SAMPLES - 1);
  const band = document.createElementNS(svgNs, "rect");
  band.setAttribute("x", winX1);
  band.setAttribute("y", top);
  band.setAttribute("width", Math.max(2, winX2 - winX1));
  band.setAttribute("height", bottom - top);
  band.setAttribute("fill", "#e7f3ee");
  band.setAttribute("opacity", "0.85");
  svg.appendChild(band);

  function drawSeries(yFn, color, key) {
    let path = "";
    for (let i = 0; i < readings.length; i++) {
      const x = xAt(i);
      const y = yFn(readings[i][key]);
      path += (i === 0 ? "M " : " L ") + x + " " + y;
    }
    const line = document.createElementNS(svgNs, "path");
    line.setAttribute("d", path);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linejoin", "round");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  }
  drawSeries(yTemp, "#0f6e56", "temp_c");
  drawSeries(yHum, "#5c564c", "humidity_pct");

  const axis = document.createElementNS(svgNs, "line");
  axis.setAttribute("x1", left);
  axis.setAttribute("x2", right);
  axis.setAttribute("y1", bottom);
  axis.setAttribute("y2", bottom);
  axis.setAttribute("stroke", "#ddd4c4");
  axis.setAttribute("stroke-width", "1");
  svg.appendChild(axis);

  function label(x, y, text, color) {
    const node = document.createElementNS(svgNs, "text");
    node.setAttribute("x", x);
    node.setAttribute("y", y);
    node.setAttribute("fill", color);
    node.setAttribute("font-size", "11");
    node.setAttribute("font-family", "ui-sans-serif, system-ui, sans-serif");
    node.textContent = text;
    svg.appendChild(node);
  }
  label(4, top + 10, formatNum(maxTemp) + "°", "#0f6e56");
  label(4, bottom, formatNum(minTemp) + "°", "#0f6e56");
  label(right - 36, top + 10, formatNum(maxHum) + "%", "#5c564c");
  label(splitX + 4, top + 12, "split", "#9a3412");
}

function syncSlider() {
  const slider = document.getElementById("window-start");
  const starts = possibleStarts();
  if (starts.length === 0) return;
  slider.min = String(starts[0]);
  slider.max = String(starts[starts.length - 1]);
  slider.value = String(windowStart);
  document.getElementById("window-prev").disabled = windowStart <= starts[0];
  document.getElementById("window-next").disabled = windowStart >= starts[starts.length - 1];
}

function render() {
  windowStart = clampWindowStart(windowStart);
  syncSlider();
  renderSeriesMeta();
  renderChart();
  renderWindow();
  renderLabels();
  renderSplit();
}

function saveLabel() {
  const choice = document.getElementById("label-choice").value;
  if (LABEL_NAMES.indexOf(choice) < 0) return;
  labelsByStart[windowStart] = choice;
  render();
}

function clearLabel() {
  delete labelsByStart[windowStart];
  render();
}

function moveWindow(delta) {
  windowStart = clampWindowStart(windowStart + delta);
  render();
}

async function loadSeries() {
  const response = await fetch("data/thermal_series.json");
  if (!response.ok) throw new Error("Could not load data/thermal_series.json");
  const data = await response.json();
  sampleSeconds = data.sample_seconds || 2;
  readings = [];
  for (let i = 0; i < data.readings.length; i++) {
    readings.push({
      temp_c: Number(data.readings[i].temp_c),
      humidity_pct: Number(data.readings[i].humidity_pct),
    });
  }
  labelsByStart = {};
  windowStart = 1;
  render();
  document.getElementById("readout").textContent =
    "Series loaded. Slide the window. Save labels on early and late halves.";
}

document.addEventListener("DOMContentLoaded", function start() {
  document.getElementById("window-start").addEventListener("input", function onSlide(event) {
    windowStart = clampWindowStart(Number(event.target.value));
    render();
  });
  document.getElementById("window-prev").addEventListener("click", function onPrev() { moveWindow(-1); });
  document.getElementById("window-next").addEventListener("click", function onNext() { moveWindow(1); });
  document.getElementById("save-label").addEventListener("click", saveLabel);
  document.getElementById("clear-label").addEventListener("click", clearLabel);
  loadSeries().catch(function onLoadError(error) {
    document.getElementById("readout").textContent = error.message;
  });
});
