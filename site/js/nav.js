// Shared page list. Later lessons stay here as labels until their pages exist.
const SITE_PAGES = [
  { href: "perceptron.html", label: "Perceptron", ready: true },
  { href: "neural-net.html", label: "Neural net", ready: true },
  { href: "data.html", label: "Data", ready: false },
  { href: "train.html", label: "Train", ready: false },
  { href: "run.html", label: "Run", ready: false },
];

function renderSiteNav() {
  const nav = document.querySelector("[data-site-nav]");
  if (!nav) return;
  const current = document.body.dataset.page || "";
  nav.textContent = "";
  for (let i = 0; i < SITE_PAGES.length; i++) {
    const page = SITE_PAGES[i];
    if (page.ready) {
      const link = document.createElement("a");
      link.href = page.href;
      link.textContent = page.label;
      if (page.href === current) link.setAttribute("aria-current", "page");
      nav.appendChild(link);
    } else {
      const item = document.createElement("span");
      item.textContent = page.label;
      item.title = "Later lesson";
      nav.appendChild(item);
    }
  }
}

document.addEventListener("DOMContentLoaded", renderSiteNav);
