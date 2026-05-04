function fmt(value, digits = 3) {
  const num = Number(value);
  if (Number.isNaN(num) || !Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function fmtMetric(value, digits = 3) {
  const num = Number(value);
  if (Number.isNaN(num) || !Number.isFinite(num)) return "-";
  const abs = Math.abs(num);
  if (abs > 0 && abs < 10 ** -digits) return num.toExponential(2);
  return num.toFixed(digits);
}

function renderDatasetStats(meta) {
  const body = document.getElementById("dataset-stats-body");
  const splits = (meta && meta.splits) || {};
  const classes = (meta && meta.classes) || ["car"];
  const resolution = (meta && meta.resolution) || "1280x720";

  body.innerHTML = "";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${resolution.replace("x", "×")}</td>
    <td>${classes.join(", ")}</td>
    <td>${splits.train != null ? splits.train : "-"}</td>
    <td>${splits.val != null ? splits.val : "-"}</td>
    <td>${splits.test != null ? splits.test : "-"}</td>
    <td>COCO+YOLO</td>
  `;
  body.appendChild(tr);
}

function getRank(value, allValues, isMinBetter = false) {
  const sorted = [...new Set(allValues)].sort((a, b) => isMinBetter ? a - b : b - a);
  const index = sorted.findIndex((v) => Math.abs(v - value) < 1e-9);
  return index === -1 ? null : index + 1;
}

function renderResultsTable(data) {
  const body = document.getElementById("results-body");
  body.innerHTML = "";

  const rows = Array.isArray(data.models) ? data.models : [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5" class="notice">No detection benchmark data found.</td>`;
    body.appendChild(tr);
    return;
  }

  const allMap50 = rows.map((r) => Number(r.mAP50 || 0));
  const allMap95 = rows.map((r) => Number(r.mAP50_95 || 0));
  const allSpeed = rows.map((r) => Number(r.inference_ms || 0));

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const map50 = Number(row.mAP50 || 0);
    const map95 = Number(row.mAP50_95 || 0);
    const speed = Number(row.inference_ms || 0);

    const rank50 = getRank(map50, allMap50, false);
    const rank95 = getRank(map95, allMap95, false);
    const rankSpeed = getRank(speed, allSpeed, true);

    const getClass = (rank) => {
      if (rank === 1) return "best";
      if (rank === 2) return "second-best";
      if (rank === 3) return "third-best";
      return "";
    };

    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${row.type}</td>
      <td class="${getClass(rank50)}">${fmt(map50, 3)}</td>
      <td class="${getClass(rank95)}">${fmt(map95, 3)}</td>
      <td class="${getClass(rankSpeed)}">${fmt(speed, 1)}</td>
    `;
    body.appendChild(tr);
  });
}

function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];

  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    if (cols.length !== headers.length) continue;
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = cols[idx].trim();
    });
    rows.push(row);
  }
  return rows;
}

function renderBackboneTable(rows) {
  const body = document.getElementById("backbone-body");
  body.innerHTML = "";

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="7" class="notice">No backbone experiment data found.</td>`;
    body.appendChild(tr);
    return;
  }

  const allMap50 = rows.map((r) => Number(r.mAP50 || 0));
  const allMap95 = rows.map((r) => Number(r.mAP50_95 || 0));
  const allPrecision = rows.map((r) => Number(r.precision || 0));
  const allRecall = rows.map((r) => Number(r.recall || 0));
  const allSpeed = rows.map((r) => Number(r.inference_ms || 0));
  const allParams = rows.map((r) => Number(r.params_M || 0));

  const getClass = (rank) => {
    if (rank === 1) return "best";
    if (rank === 2) return "second-best";
    if (rank === 3) return "third-best";
    return "";
  };

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const map50 = Number(row.mAP50 || 0);
    const map95 = Number(row.mAP50_95 || 0);
    const precision = Number(row.precision || 0);
    const recall = Number(row.recall || 0);
    const speed = Number(row.inference_ms || 0);
    const params = Number(row.params_M || 0);

    const rank50 = getRank(map50, allMap50, false);
    const rank95 = getRank(map95, allMap95, false);
    const rankPrecision = getRank(precision, allPrecision, false);
    const rankRecall = getRank(recall, allRecall, false);
    const rankSpeed = getRank(speed, allSpeed, true);
    const rankParams = getRank(params, allParams, true);

    tr.innerHTML = `
      <td>${row.model}</td>
      <td class="${getClass(rank50)}">${fmtMetric(map50, 3)}</td>
      <td class="${getClass(rank95)}">${fmtMetric(map95, 3)}</td>
      <td class="${getClass(rankPrecision)}">${fmt(precision, 3)}</td>
      <td class="${getClass(rankRecall)}">${fmt(recall, 3)}</td>
      <td class="${getClass(rankSpeed)}">${fmt(speed, 1)}</td>
      <td class="${getClass(rankParams)}">${fmt(params, 2)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderQualitative(meta) {
  const grid = document.getElementById("qualitative-grid");
  grid.innerHTML = "";

  const scenes = ((meta && meta.qualitative_scenes) || []).slice(0, 3);
  if (!scenes.length) {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = "No qualitative metadata found yet. Run evaluation first.";
    grid.appendChild(p);
    return;
  }

  scenes.forEach((stem) => {
    const row = document.createElement("div");
    row.className = "qual-row";

    const cards = [
      { label: `${stem} • RGB Input`, file: `assets/images/qualitative/input_${stem}.png` },
      { label: `${stem} • YOLOv8s`, file: `assets/images/qualitative/yolov8s/${stem}.png` },
      { label: `${stem} • RT-DETR-L`, file: `assets/images/qualitative/rtdetr_l/${stem}.png` },
      { label: `${stem} • Faster R-CNN`, file: `assets/images/qualitative/fasterrcnn_resnet50_fpn_v2/${stem}.png` },
    ];

    cards.forEach((card) => {
      const wrapper = document.createElement("div");
      wrapper.className = "qual-card";
      wrapper.innerHTML = `
        <p>${card.label}</p>
        <img src="${card.file}" alt="${card.label}" loading="lazy" />
      `;
      row.appendChild(wrapper);
    });

    grid.appendChild(row);
  });
}

async function loadResults() {
  const candidates = [
    "assets/data/results.json",
    "../test_visiondrive3d/results.json",
  ];

  try {
    let data = null;
    let lastError = null;

    for (const path of candidates) {
      try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        data = await response.json();
        break;
      } catch (err) {
        lastError = err;
      }
    }

    if (!data) throw lastError || new Error("No results source found");

    renderDatasetStats(data.meta || {});
    renderResultsTable(data);
    renderQualitative(data.meta || {});
  } catch (err) {
    console.error(err);
    document.getElementById("results-body").innerHTML =
      '<tr><td colspan="5" class="notice">results.json not found. Run pipeline first.</td></tr>';
    document.getElementById("dataset-stats-body").innerHTML =
      '<tr><td colspan="6" class="notice">Dataset stats unavailable.</td></tr>';
  }
}

async function loadBackboneCsv() {
  const candidates = [
    "assets/data/backbone_comparison.csv",
    "../test_visiondrive3d/backbone_comparison.csv",
  ];

  try {
    let rows = null;
    let lastError = null;

    for (const path of candidates) {
      try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        rows = parseCsv(text);
        break;
      } catch (err) {
        lastError = err;
      }
    }

    if (!rows) throw lastError || new Error("No CSV source found");
    renderBackboneTable(rows);
  } catch (err) {
    console.error(err);
    document.getElementById("backbone-body").innerHTML =
      '<tr><td colspan="7" class="notice">backbone_comparison.csv not found.</td></tr>';
  }
}

loadResults();
loadBackboneCsv();
