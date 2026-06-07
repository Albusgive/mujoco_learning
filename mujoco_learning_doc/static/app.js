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

// --- Interactive Solver Visualizer ---
document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("solver-visualizer-container");
  if (!container) return;

  // Dynamically inject the visualizer HTML markup to keep the Markdown file pristine
  container.innerHTML = `
    <div class="solver-visualizer-card">
      <h3>MuJoCo 接触解算参数实时可视化工具</h3>
      <div class="visualizer-layout">
        <div class="canvas-panel">
          <canvas id="solver-canvas" style="width: 100%; height: 260px;"></canvas>
          <div class="plot-legend">
            <span class="legend-item"><span class="color-box d-curve"></span>d(r) - 实线：阻抗</span>
            <span class="legend-item"><span class="color-box k-curve"></span>k(r) - 疏虚线：刚度</span>
            <span class="legend-item"><span class="color-box b-curve"></span>b(r) - 密虚线：阻尼</span>
          </div>
        </div>
        <div class="controls-panel">
          <div class="control-group">
            <label>预设模式 (Presets):</label>
            <select id="preset-select">
              <option value="custom">自定义 (Custom)</option>
              <option value="rubber">橡胶球 (Rubber Ball)</option>
              <option value="metal">牛顿摆/金属 (Newton Cradle)</option>
              <option value="cushion">缓冲垫 (Cushioning)</option>
            </select>
          </div>
          <div class="control-group">
            <label>Y 轴缩放模式 (Y-axis Scale):</label>
            <select id="scale-mode-select">
              <option value="absolute">绝对物理值 (高度随参数发生变化)</option>
              <option value="normalized">形状归一化 (显示变化趋势)</option>
            </select>
          </div>

          <div class="tab-container">
            <button class="tab-btn active" onclick="switchTab('imp')">solimp 阻抗曲线</button>
            <button class="tab-btn" onclick="switchTab('ref')">solref 回弹参考</button>
          </div>

          <!-- solimp parameters -->
          <div id="imp-controls" class="tab-content active">
            <div class="slider-item">
              <label>dmin (d₀): <span id="val-d0">0.9</span></label>
              <input type="range" id="param-d0" min="0" max="1" step="0.01" value="0.9">
            </div>
            <div class="slider-item">
              <label>dmax (dwidth): <span id="val-dwidth">0.95</span></label>
              <input type="range" id="param-dwidth" min="0.0001" max="0.9999" step="0.01" value="0.95">
            </div>
            <div class="slider-item">
              <label>width: <span id="val-width">0.001</span></label>
              <input type="range" id="param-width" min="0.0001" max="0.01" step="0.0001" value="0.001">
            </div>
            <div class="slider-item">
              <label>midpoint: <span id="val-midpoint">0.5</span></label>
              <input type="range" id="param-midpoint" min="0.01" max="0.99" step="0.01" value="0.5">
            </div>
            <div class="slider-item">
              <label>power: <span id="val-power">2</span></label>
              <input type="range" id="param-power" min="1" max="10" step="0.5" value="2">
            </div>
          </div>

          <!-- solref parameters -->
          <div id="ref-controls" class="tab-content">
            <div class="control-group">
              <label>参考格式 (Format):</label>
              <select id="ref-format-select">
                <option value="standard">标准 (timeconst, dampratio)</option>
                <option value="direct">直接 (stiffness, damping)</option>
              </select>
            </div>
            <div id="standard-ref-inputs" style="display: flex; flex-direction: column; gap: 14px;">
              <div class="slider-item">
                <label>timeconst (τ): <span id="val-timeconst">0.02</span>s</label>
                <input type="range" id="param-timeconst" min="0.001" max="0.1" step="0.001" value="0.02">
              </div>
              <div class="slider-item">
                <label>dampratio (ζ): <span id="val-dampratio">1.0</span></label>
                <input type="range" id="param-dampratio" min="0.1" max="5.0" step="0.1" value="1.0">
              </div>
            </div>
            <div id="direct-ref-inputs" style="display: none; flex-direction: column; gap: 14px;">
              <div class="slider-item">
                <label>stiffness (k): <span id="val-stiffness">1000</span></label>
                <input type="range" id="param-stiffness" min="10" max="10000" step="10" value="1000">
              </div>
              <div class="slider-item">
                <label>damping (b): <span id="val-damping">10</span></label>
                <input type="range" id="param-damping" min="1" max="500" step="1" value="10">
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const canvas = document.getElementById("solver-canvas");
  const ctx = canvas.getContext("2d");

  // Setup High DPI Canvas
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  // Inputs
  const d0Input = document.getElementById("param-d0");
  const dwidthInput = document.getElementById("param-dwidth");
  const widthInput = document.getElementById("param-width");
  const midpointInput = document.getElementById("param-midpoint");
  const powerInput = document.getElementById("param-power");
  const presetSelect = document.getElementById("preset-select");
  const scaleModeSelect = document.getElementById("scale-mode-select");
  const formatSelect = document.getElementById("ref-format-select");

  const timeconstInput = document.getElementById("param-timeconst");
  const dampratioInput = document.getElementById("param-dampratio");
  const stiffnessInput = document.getElementById("param-stiffness");
  const dampingInput = document.getElementById("param-damping");

  // Value displays
  const d0Val = document.getElementById("val-d0");
  const dwidthVal = document.getElementById("val-dwidth");
  const widthVal = document.getElementById("val-width");
  const midpointVal = document.getElementById("val-midpoint");
  const powerVal = document.getElementById("val-power");
  const timeconstVal = document.getElementById("val-timeconst");
  const dampratioVal = document.getElementById("val-dampratio");
  const stiffnessVal = document.getElementById("val-stiffness");
  const dampingVal = document.getElementById("val-damping");

  const standardInputs = document.getElementById("standard-ref-inputs");
  const directInputs = document.getElementById("direct-ref-inputs");

  // Tab switching
  window.switchTab = (tab) => {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));
    
    if (tab === 'imp') {
      document.querySelector(".tab-btn[onclick*='imp']").classList.add("active");
      document.getElementById("imp-controls").classList.add("active");
    } else {
      document.querySelector(".tab-btn[onclick*='ref']").classList.add("active");
      document.getElementById("ref-controls").classList.add("active");
    }
  };

  // Presets mapping
  const presets = {
    rubber: { d0: 0.5, dwidth: 0.95, width: 0.005, midpoint: 0.1, power: 2.0, format: "standard", timeconst: 0.05, dampratio: 0.8 },
    metal: { d0: 0.0, dwidth: 0.6, width: 0.0002, midpoint: 0.5, power: 4.0, format: "standard", timeconst: 0.002, dampratio: 1.0 },
    cushion: { d0: 0.8, dwidth: 0.99, width: 0.008, midpoint: 0.3, power: 2.0, format: "standard", timeconst: 0.04, dampratio: 2.0 }
  };

  presetSelect.addEventListener("change", () => {
    const val = presetSelect.value;
    if (val === "custom") return;
    const p = presets[val];
    
    d0Input.value = p.d0;
    dwidthInput.value = p.dwidth;
    widthInput.value = p.width;
    midpointInput.value = p.midpoint;
    powerInput.value = p.power;

    formatSelect.value = p.format;
    if (p.format === "standard") {
      timeconstInput.value = p.timeconst;
      dampratioInput.value = p.dampratio;
      standardInputs.style.display = "flex";
      directInputs.style.display = "none";
    } else {
      stiffnessInput.value = p.stiffness;
      dampingInput.value = p.damping;
      standardInputs.style.display = "none";
      directInputs.style.display = "flex";
    }
    update();
  });

  scaleModeSelect.addEventListener("change", () => {
    update();
  });

  formatSelect.addEventListener("change", () => {
    if (formatSelect.value === "standard") {
      standardInputs.style.display = "flex";
      directInputs.style.display = "none";
    } else {
      standardInputs.style.display = "none";
      directInputs.style.display = "flex";
    }
    presetSelect.value = "custom";
    update();
  });

  const inputs = [d0Input, dwidthInput, widthInput, midpointInput, powerInput, timeconstInput, dampratioInput, stiffnessInput, dampingInput];
  inputs.forEach(input => {
    input.addEventListener("input", () => {
      presetSelect.value = "custom";
      update();
    });
  });

  function getImpedanceValue(r, d0, dwidth, width, midpoint, power) {
    if (d0 === dwidth || width <= 1e-6) {
      return 0.5 * (d0 + dwidth);
    }
    const x = Math.min(1.0, Math.max(0.0, Math.abs(r) / width));
    let y = 0;
    if (power === 1) {
      y = x;
    } else if (x <= midpoint) {
      const a = 1 / Math.pow(midpoint, power - 1);
      y = a * Math.pow(x, power);
    } else {
      const b = 1 / Math.pow(1 - midpoint, power - 1);
      y = 1 - b * Math.pow(1 - x, power);
    }
    return d0 + y * (dwidth - d0);
  }

  function update() {
    // Read values
    const d0 = parseFloat(d0Input.value);
    const dwidth = parseFloat(dwidthInput.value);
    const width = parseFloat(widthInput.value);
    const midpoint = parseFloat(midpointInput.value);
    const power = parseFloat(powerInput.value);

    const isStandard = formatSelect.value === "standard";
    const timeconst = parseFloat(timeconstInput.value);
    const dampratio = parseFloat(dampratioInput.value);
    const stiffness = parseFloat(stiffnessInput.value);
    const damping = parseFloat(dampingInput.value);
    const scaleMode = scaleModeSelect ? scaleModeSelect.value : "absolute";

    // Update value labels
    d0Val.textContent = d0.toFixed(2);
    dwidthVal.textContent = dwidth.toFixed(2);
    widthVal.textContent = width.toFixed(4);
    midpointVal.textContent = midpoint.toFixed(2);
    powerVal.textContent = power.toFixed(1);

    timeconstVal.textContent = timeconst.toFixed(3);
    dampratioVal.textContent = dampratio.toFixed(1);
    stiffnessVal.textContent = stiffness;
    dampingVal.textContent = damping;

    // Calculate Base K and B
    let K_base = 0;
    let B_base = 0;
    if (isStandard) {
      K_base = 1 / (dwidth * dwidth * timeconst * timeconst * dampratio * dampratio);
      B_base = 2 / (dwidth * timeconst);
    } else {
      K_base = stiffness / (dwidth * dwidth);
      B_base = damping / dwidth;
    }

    // Clear Canvas
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    // Draw grid and axes
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#9ca3af";

    const padding = { left: 50, right: 50, top: 40, bottom: 40 };
    const graphW = w - padding.left - padding.right;
    const graphH = h - padding.top - padding.bottom;

    // Draw horizontal grid lines (Y-axis grid)
    for (let i = 0; i <= 4; i++) {
      const yVal = i / 4;
      const py = padding.top + graphH * (1 - yVal);
      ctx.beginPath();
      ctx.moveTo(padding.left, py);
      ctx.lineTo(w - padding.right, py);
      ctx.stroke();
      if (scaleMode === "normalized") {
        ctx.fillText(yVal.toFixed(2), padding.left - 30, py + 4);
      } else {
        ctx.fillText((yVal * 100).toFixed(0) + "%", padding.left - 35, py + 4);
      }
    }

    // Draw vertical grid lines (X-axis grid)
    const maxX = width * 1.5;
    for (let i = 0; i <= 3; i++) {
      const xVal = (i / 3) * maxX;
      const px_fill = padding.left + graphW * (i / 3);
      ctx.beginPath();
      ctx.moveTo(px_fill, padding.top);
      ctx.lineTo(px_fill, h - padding.bottom);
      ctx.stroke();
      ctx.fillText(xVal.toFixed(4), px_fill - 15, h - padding.bottom + 16);
    }

    // Labels
    ctx.fillStyle = "#374151";
    ctx.fillText("渗透深度 r (m)", w / 2 - 30, h - 10);
    
    // Y-Axis titles
    ctx.save();
    ctx.translate(15, h / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#0f766e";
    if (scaleMode === "normalized") {
      ctx.fillText("阻抗 / 归一化强度", 0, 0);
    } else {
      ctx.fillText("阻抗及强度百分比 (%)", 0, 0);
    }
    ctx.restore();

    // Draw curves
    const points = 100;
    const dPoints = [];
    const kPoints = [];
    const bPoints = [];

    let maxK = 0;
    let maxB = 0;

    for (let i = 0; i <= points; i++) {
      const r = (i / points) * maxX;
      const d = getImpedanceValue(r, d0, dwidth, width, midpoint, power);
      const k = d * K_base;
      const b = d * B_base;
      
      dPoints.push({ r, val: d });
      kPoints.push({ r, val: k });
      bPoints.push({ r, val: b });

      if (k > maxK) maxK = k;
      if (b > maxB) maxB = b;
    }

    // Display peak stiffness and damping text dynamically on canvas to reflect solref changes
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.fillStyle = "#ef4444";
    ctx.fillText(`k_max (最大刚度): ${maxK.toFixed(0)} N/m`, padding.left + 10, padding.top - 15);
    ctx.fillStyle = "#3b82f6";
    ctx.fillText(`b_max (最大阻尼): ${maxB.toFixed(1)} Ns/m`, padding.left + 220, padding.top - 15);

    // Render d(r) curve
    ctx.beginPath();
    ctx.strokeStyle = "#0f766e"; // Teal
    ctx.lineWidth = 3;
    dPoints.forEach((p, idx) => {
      const px = padding.left + (p.r / maxX) * graphW;
      const py = padding.top + (1 - p.val) * graphH;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Render k(r) curve
    ctx.beginPath();
    ctx.strokeStyle = "#ef4444"; // Red
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    kPoints.forEach((p, idx) => {
      const px = padding.left + (p.r / maxX) * graphW;
      let normVal = 0;
      if (scaleMode === "normalized") {
        normVal = maxK > 0 ? p.val / maxK : 0;
      } else {
        const K_scale = Math.max(10000, maxK);
        normVal = p.val / K_scale;
      }
      const py = padding.top + (1 - normVal) * graphH;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Render b(r) curve
    ctx.beginPath();
    ctx.strokeStyle = "#3b82f6"; // Blue
    ctx.lineWidth = 2;
    ctx.setLineDash([2, 2]);
    bPoints.forEach((p, idx) => {
      const px = padding.left + (p.r / maxX) * graphW;
      let normVal = 0;
      if (scaleMode === "normalized") {
        normVal = maxB > 0 ? p.val / maxB : 0;
      } else {
        const B_scale = Math.max(500, maxB);
        normVal = p.val / B_scale;
      }
      const py = padding.top + (1 - normVal) * graphH;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]); // Reset
  }

  // Initial update
  update();
});
