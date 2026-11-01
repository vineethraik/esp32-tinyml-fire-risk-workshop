// Run lesson: Web Serial console for 04_final_combined AT commands.
//
// Chrome / Edge. 115200 8N1. Write COMMAND + newline. Show every line the board prints.
// Close PlatformIO Serial Monitor first. Opening the port usually resets the ESP32.

const BAUD = 115200;
const MAX_LINES = 2000;

let port = null;
let reader = null;
let readLoopActive = false;
let lineBuffer = "";
let writeChain = Promise.resolve();
const lines = [];
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function serialSupported() {
  return Boolean(navigator.serial);
}

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("port-status").textContent = text;
}

function commandControls() {
  return document.querySelectorAll(".cmd, #export-range, #export-offset, #export-count");
}

function setConnected(on) {
  $("connect").disabled = on || !serialSupported();
  $("disconnect").disabled = !on;
  const controls = commandControls();
  for (let i = 0; i < controls.length; i++) controls[i].disabled = !on;
}

function monitorEl() {
  return $("monitor");
}

function stickToBottom() {
  const el = monitorEl();
  return el.scrollHeight - el.scrollTop - el.clientHeight < 32;
}

function renderMonitor() {
  const el = monitorEl();
  const stick = stickToBottom();
  el.textContent = lines.length ? lines.join("\n") : "";
  if (stick) el.scrollTop = el.scrollHeight;
}

function appendLine(text) {
  lines.push(text);
  while (lines.length > MAX_LINES) lines.shift();
  renderMonitor();
}

function clearMonitor() {
  lines.length = 0;
  lineBuffer = "";
  const el = monitorEl();
  el.textContent = "";
}

function consumeChunk(chunk) {
  lineBuffer += chunk.replace(/\r/g, "");
  const parts = lineBuffer.split("\n");
  lineBuffer = parts.pop();
  for (let i = 0; i < parts.length; i++) appendLine(parts[i]);
}

async function writeOne(command) {
  if (!port || !port.writable) return;
  const writer = port.writable.getWriter();
  try {
    await writer.write(encoder.encode(command + "\n"));
    appendLine(">> " + command);
  } catch (error) {
    appendLine("ERR+WRITE," + error.message);
  } finally {
    writer.releaseLock();
  }
}

function sendCommand(command) {
  if (!port || !port.writable) return;
  writeChain = writeChain.then(function sendNext() {
    return writeOne(command);
  }).catch(function onWriteFail() {
    // Keep the queue alive after a failed write.
  });
}

function exportRangeCommand() {
  const offset = Math.floor(Number($("export-offset").value));
  const count = Math.floor(Number($("export-count").value));
  if (!Number.isFinite(offset) || offset < 0 || !Number.isFinite(count) || count <= 0) {
    appendLine("ERR+EXPORT,arguments");
    return;
  }
  sendCommand("AT+EXPORT," + offset + "," + count);
}

async function readLoop() {
  readLoopActive = true;
  while (port && port.readable && readLoopActive) {
    reader = port.readable.getReader();
    try {
      while (true) {
        const result = await reader.read();
        if (result.done) break;
        if (result.value) consumeChunk(decoder.decode(result.value, { stream: true }));
      }
    } catch (error) {
      if (readLoopActive) appendLine("ERR+READ," + error.message);
      break;
    } finally {
      try {
        reader.releaseLock();
      } catch (error) {
        // Reader already released on cancel.
      }
      reader = null;
    }
  }
}

async function closePort(note) {
  readLoopActive = false;
  if (reader) {
    try {
      await reader.cancel();
    } catch (error) {
      // Already gone.
    }
    reader = null;
  }
  if (port) {
    try {
      await port.close();
    } catch (error) {
      // Already closed.
    }
    port.removeEventListener("disconnect", onPortDisconnect);
  }
  port = null;
  setConnected(false);
  setStatus(note || "disconnected");
}

function onPortDisconnect() {
  closePort("disconnected (cable unplugged)");
}

function portLabel(serialPort) {
  const info = serialPort.getInfo ? serialPort.getInfo() : {};
  if (info.usbVendorId && info.usbProductId) {
    const vid = info.usbVendorId.toString(16).padStart(4, "0");
    const pid = info.usbProductId.toString(16).padStart(4, "0");
    return "connected  usb:" + vid + ":" + pid + "  115200";
  }
  return "connected  115200";
}

async function connectPort() {
  if (!serialSupported()) {
    setStatus("Web Serial missing. Use Chrome or Edge.");
    return;
  }
  if (port) return;
  let picked;
  try {
    picked = await navigator.serial.requestPort();
  } catch (error) {
    if (error.name === "NotFoundError") {
      setStatus("disconnected (no port picked)");
      return;
    }
    setStatus("connect failed: " + error.message);
    return;
  }
  try {
    await picked.open({ baudRate: BAUD });
  } catch (error) {
    setStatus("open failed: " + error.message + "  Close PlatformIO Monitor and retry.");
    return;
  }
  port = picked;
  port.addEventListener("disconnect", onPortDisconnect);
  setConnected(true);
  setStatus(portLabel(port));
  appendLine("opened 115200");
  readLoop();
}

function onUnsupported() {
  setConnected(false);
  $("connect").disabled = true;
  $("disconnect").disabled = true;
  setStatus("Web Serial missing. Use Chrome or Edge on HTTPS or localhost.");
}

document.addEventListener("DOMContentLoaded", function start() {
  if (!serialSupported()) {
    onUnsupported();
  } else {
    setConnected(false);
    setStatus("disconnected");
  }

  $("connect").addEventListener("click", connectPort);
  $("disconnect").addEventListener("click", function onDisconnect() {
    closePort("disconnected");
  });
  $("clear-monitor").addEventListener("click", clearMonitor);
  $("export-range").addEventListener("click", exportRangeCommand);

  const buttons = document.querySelectorAll(".cmd");
  for (let i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener("click", function onCmd() {
      sendCommand(buttons[i].getAttribute("data-cmd"));
    });
  }
});
