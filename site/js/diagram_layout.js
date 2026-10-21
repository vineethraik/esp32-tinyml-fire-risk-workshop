// Shared horizontal stretch for lesson diagrams.
// Revert the commit that added this file (and its call sites) to restore the old fixed widths.

function diagramFrameWidth(svg, minimum) {
  const min = minimum || 640;
  const frame = svg.closest(".diagram-frame") || svg.parentElement;
  if (!frame) return min;
  const style = window.getComputedStyle(frame);
  const pad = (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0);
  return Math.max(min, Math.floor(frame.clientWidth - pad));
}

function watchDiagramResize(renderFn) {
  let ticking = false;
  window.addEventListener("resize", function onResize() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function onFrame() {
      ticking = false;
      renderFn();
    });
  });
}
