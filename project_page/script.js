function fmt(value, digits = 3) {
  const num = Number(value);
  if (Number.isNaN(num) || !Number.isFinite(num)) return "-";
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
    <td>${splits.train ?? "-"}</td>
    <td>${splits.val ?? "-"}</td>
    <td>${splits.test ?? "-"}</td>
    <td>COCO+YOLO</td>
  `;
  body.appendChild(tr);
}

function renderResultsTable(data) {
  const body = document.getElementById("results-body");
  body.innerHTML = "";

  const rows = [
    {
      model: "YOLOv8n (baseline)",
      type: "Detection",
      ...data.detection.baseline,
    },
    {
      model: "YOLOv8n (fine-tuned)",
      type: "Detection",
      ...data.detection.finetuned,
    },
    {
      model: "YOLOv8n-seg (baseline)",
      type: "Segmentation",
      ...data.segmentation.baseline,
    },
    {
      model: "YOLOv8n-seg (fine-tuned)",
      type: "Segmentation",
      ...data.segmentation.finetuned,
    },
  ];

  const bestMap50 = Math.max(...rows.map((r) => Number(r.mAP50 || 0)));
  const bestMap95 = Math.max(...rows.map((r) => Number(r.mAP50_95 || 0)));
  const bestSpeed = Math.min(...rows.map((r) => Number(r.inference_ms || Number.POSITIVE_INFINITY)));

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const map50 = Number(row.mAP50 || 0);
    const map95 = Number(row.mAP50_95 || 0);
    const speed = Number(row.inference_ms || 0);

    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${row.type}</td>
      <td class="${Math.abs(map50 - bestMap50) < 1e-9 ? "best" : ""}">${fmt(map50, 3)}</td>
      <td class="${Math.abs(map95 - bestMap95) < 1e-9 ? "best" : ""}">${fmt(map95, 3)}</td>
      <td class="${Math.abs(speed - bestSpeed) < 1e-9 ? "best" : ""}">${fmt(speed, 1)}</td>
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

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${fmt(row.mAP50, 3)}</td>
      <td>${fmt(row.mAP50_95, 3)}</td>
      <td>${fmt(row.precision, 3)}</td>
      <td>${fmt(row.recall, 3)}</td>
      <td>${fmt(row.inference_ms, 1)}</td>
      <td>${fmt(row.params_M, 2)}</td>
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
      { label: `${stem} • Baseline Det`, file: `assets/images/qualitative/detection_baseline_${stem}.png` },
      { label: `${stem} • Fine-tuned Det`, file: `assets/images/qualitative/detection_finetuned_${stem}.png` },
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
  try {
    const response = await fetch("assets/data/results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

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
  try {
    const response = await fetch("assets/data/backbone_comparison.csv");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    const rows = parseCsv(text);
    renderBackboneTable(rows);
  } catch (err) {
    console.error(err);
    document.getElementById("backbone-body").innerHTML =
      '<tr><td colspan="7" class="notice">backbone_comparison.csv not found.</td></tr>';
  }
}

loadResults();
loadBackboneCsv();
