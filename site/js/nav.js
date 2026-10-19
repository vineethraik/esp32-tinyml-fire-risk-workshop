// Shared page list. Every item is a real link on every lesson page.
const SITE_PAGES = [
  { href: "index.html", label: "Home" },
  { href: "perceptron.html", label: "Perceptron" },
  { href: "neural-net.html", label: "Neural net" },
  { href: "data.html", label: "Data" },
  { href: "train.html", label: "Train" },
  { href: "run.html", label: "Run" },
];

function renderSiteNav() {
  const nav = document.querySelector("[data-site-nav]");
  if (!nav) return;
  const current = document.body.dataset.page || "";
  nav.textContent = "";
  for (let i = 0; i < SITE_PAGES.length; i++) {
    const page = SITE_PAGES[i];
    const link = document.createElement("a");
    link.href = page.href;
    link.textContent = page.label;
    if (page.href === current) link.setAttribute("aria-current", "page");
    nav.appendChild(link);
  }
}

document.addEventListener("DOMContentLoaded", renderSiteNav);
