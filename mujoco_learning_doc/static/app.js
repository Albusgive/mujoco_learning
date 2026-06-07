const searchInput = document.querySelector("#doc-search");
const sidebar = document.querySelector(".sidebar");
const links = Array.from(document.querySelectorAll(".doc-link"));
const groups = Array.from(document.querySelectorAll(".doc-group"));
const initialOpenGroups = new WeakMap();
const sidebarScrollKey = "mujoco-learning-doc-sidebar-scroll";
const groupStateKey = "mujoco-learning-doc-group-state";

function loadGroupState() {
  try {
    return JSON.parse(sessionStorage.getItem(groupStateKey) || "{}");
  } catch {
    return {};
  }
}

function saveGroupState() {
  const state = {};
  groups.forEach((group) => {
    const title = group.querySelector("summary span")?.textContent?.trim();
    if (title) {
      state[title] = group.open;
    }
  });
  sessionStorage.setItem(groupStateKey, JSON.stringify(state));
}

function saveSidebarScroll() {
  if (sidebar) {
    sessionStorage.setItem(sidebarScrollKey, String(sidebar.scrollTop));
  }
}

function restoreSidebarScroll() {
  if (!sidebar) {
    return;
  }
  const saved = Number(sessionStorage.getItem(sidebarScrollKey));
  if (Number.isFinite(saved)) {
    sidebar.scrollTop = saved;
    requestAnimationFrame(() => {
      sidebar.scrollTop = saved;
    });
  }
}

const savedGroupState = loadGroupState();
groups.forEach((group) => {
  const title = group.querySelector("summary span")?.textContent?.trim();
  if (title && Object.prototype.hasOwnProperty.call(savedGroupState, title)) {
    group.open = savedGroupState[title];
  }
  initialOpenGroups.set(group, group.open);
  group.addEventListener("toggle", () => {
    saveGroupState();
    saveSidebarScroll();
  });
});

if (sidebar) {
  sidebar.addEventListener("scroll", saveSidebarScroll, { passive: true });
  restoreSidebarScroll();
}

links.forEach((link) => {
  link.addEventListener("click", () => {
    saveGroupState();
    saveSidebarScroll();
  });
});

if (searchInput) {
  searchInput.addEventListener("input", () => {
    const query = searchInput.value.trim().toLowerCase();

    links.forEach((link) => {
      const haystack = `${link.dataset.title || ""} ${link.dataset.path || ""}`;
      link.hidden = query.length > 0 && !haystack.includes(query);
    });

    groups.forEach((group) => {
      const visibleLinks = Array.from(group.querySelectorAll(".doc-link")).some(
        (link) => !link.hidden,
      );
      group.hidden = !visibleLinks;
      if (query.length > 0 && visibleLinks) {
        group.open = true;
      }
      if (query.length === 0) {
        group.open = initialOpenGroups.get(group);
      }
    });
  });
}

window.addEventListener("beforeunload", () => {
  saveGroupState();
  saveSidebarScroll();
});
