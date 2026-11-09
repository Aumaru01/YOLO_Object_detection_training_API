/* YOLO Retraining Console — vanilla JS single-page UI for the training API. */
"use strict";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString());
const fmtPct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const fmtBytes = (b) => {
  if (b == null) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(i ? 1 : 0) + " " + u[i];
};
const fmtDuration = (sec) => {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
};
// The API stores naive UTC timestamps.
const parseTime = (s) => (s ? new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z") : null);
const fmtTime = (s) => { const d = parseTime(s); return d ? d.toLocaleString() : "–"; };

const store = {
  get(k, d = "") { try { return localStorage.getItem("yolo-ui:" + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("yolo-ui:" + k, v); } catch { /* storage unavailable */ } },
};

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers["Content-Type"] = "application/json";
    init.method = init.method || "POST";
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (body.detail && body.detail.message) {
        const err = new Error(body.detail.message);
        err.detail = body.detail;
        throw err;
      }
      else if (Array.isArray(body.detail)) msg = body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("\n");
    } catch (e) { if (e.detail) throw e; /* not JSON */ }
    throw new Error(msg);
  }
  if (opts.raw) return res;
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 9000 : 4500);
}

function setStatus(el, msg, kind = "") {
  el.className = "status-line " + kind;
  if (msg instanceof Node) { el.replaceChildren(msg); } else { el.innerHTML = msg; }
}
const spinner = (text) => `<span class="spinner"></span> ${esc(text)}`;

// Box colours: dark enough for white tag text; class names are always shown
// on the tag, so identity never relies on colour alone.
const CLASS_COLORS = ["#2a78d6", "#d4572a", "#12875f", "#a87000", "#c2457a", "#2f7d32",
  "#6250d6", "#c53b3a", "#0e8a96", "#8d5a2b", "#7b3fa0", "#4f6475"];
const classColor = (i) => CLASS_COLORS[((i % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];

// ---------------------------------------------------------------------------
// Global state + routing
// ---------------------------------------------------------------------------
const state = {
  datasets: [],
  groups: [],
  baseDir: "",
  tab: "datasets",
};
const byId = (id) => state.datasets.find((d) => d.id === id);

function parseHash() {
  const h = location.hash.replace(/^#/, "");
  const [tab, query] = h.split("?");
  return { tab: tab || "datasets", params: new URLSearchParams(query || "") };
}
function go(tab, params = {}) {
  const q = new URLSearchParams(params).toString();
  location.hash = tab + (q ? "?" + q : "");
}

function route() {
  const { tab, params } = parseHash();
  const name = ["datasets", "preview", "labelstudio", "train", "monitor"].includes(tab) ? tab : "datasets";
  state.tab = name;
  $$(".tab-panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  $$(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  Tabs[name]?.enter?.(params);
}
const Tabs = {};

// ---------------------------------------------------------------------------
// Datasets: loading + shared selects
// ---------------------------------------------------------------------------
async function loadDatasets() {
  const data = await api("/datasets");
  state.datasets = data.datasets;
  state.groups = data.groups;
  state.baseDir = data.base_dir;
  $("#ds-base").textContent = "Scanning " + data.base_dir;
  $("#group-options").innerHTML = data.groups.map((g) => `<option value="${esc(g)}">`).join("");
  renderDatasetList();
  fillDatasetSelects();
}

function groupLabel(g) { return g || "datasets/ (root)"; }

function datasetOptions(filterFn, selected) {
  const groups = new Map();
  for (const d of state.datasets.filter(filterFn)) {
    if (!groups.has(d.group)) groups.set(d.group, []);
    groups.get(d.group).push(d);
  }
  let html = "";
  for (const [g, list] of groups) {
    html += `<optgroup label="${esc(groupLabel(g))}">` + list.map((d) =>
      `<option value="${esc(d.id)}" ${d.id === selected ? "selected" : ""}>${esc(d.name)} — ${fmtInt(d.num_images)} img</option>`
    ).join("") + "</optgroup>";
  }
  return html;
}

function fillDatasetSelects() {
  const keep = (sel) => $(sel).value;
  const pv = keep("#pv-dataset"), lsd = keep("#ls-dataset"), tr = keep("#tr-dataset");
  $("#pv-dataset").innerHTML = datasetOptions(() => true, pv);
  $("#ls-dataset").innerHTML = datasetOptions(() => true, lsd);
  const trainable = datasetOptions((d) => d.trainable, tr);
  $("#tr-dataset").innerHTML = trainable || `<option value="">No prepared datasets yet — build one on the left</option>`;
  fillPipelineSelects();
}

// ---------------------------------------------------------------------------
// Tab: Datasets
// ---------------------------------------------------------------------------
const DS = { selected: null, info: null, editingClasses: false, renaming: false };

function renderDatasetList() {
  const f = $("#ds-filter").value.trim().toLowerCase();
  const list = state.datasets.filter((d) => !f || d.id.toLowerCase().includes(f));
  let html = "", lastGroup = null;
  for (const d of list) {
    if (d.group !== lastGroup) { html += `<div class="ds-group">${esc(groupLabel(d.group))}</div>`; lastGroup = d.group; }
    html += `<button class="ds-item ${d.id === DS.selected ? "active" : ""}" data-id="${esc(d.id)}">
      <span class="ds-name">${esc(d.name)}</span>
      <span class="ds-meta">
        <span class="pill ${d.kind === "split" ? "split" : ""}">${d.kind === "split" ? "Prepared" : "Source"}</span>
        ${fmtInt(d.num_images)} img · ${d.num_classes} cls
        ${d.warnings.length ? `<span class="pill warn" title="${esc(d.warnings.join("\n"))}">!</span>` : ""}
      </span></button>`;
  }
  $("#ds-list").innerHTML = html || `<p class="muted small">No datasets found${f ? " for this filter" : ""}.</p>`;
}

async function selectDataset(id, { refresh = false } = {}) {
  DS.selected = id; DS.editingClasses = false; DS.renaming = false;
  renderDatasetList();
  const d = byId(id);
  const el = $("#ds-detail");
  if (!d) { el.innerHTML = `<div class="empty">Dataset not found.</div>`; return; }
  DS.info = { ...d, classes: null };
  renderDatasetDetail();
  try {
    const info = await api(`/datasets/info?id=${encodeURIComponent(id)}${refresh ? "&refresh=1" : ""}`);
    if (DS.selected !== id) return;
    DS.info = info;
    renderDatasetDetail();
  } catch (e) {
    if (DS.selected === id) $("#ds-classes").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`;
  }
}

function renderDatasetDetail() {
  const d = DS.info;
  const el = $("#ds-detail");
  const folders = d.folders.filter((f) => !f.key.includes("/"));
  const tiles = [
    `<div class="tile"><div class="k">Images</div><div class="v">${fmtInt(d.num_images)}</div></div>`,
    `<div class="tile"><div class="k">Classes</div><div class="v">${d.num_classes}</div></div>`,
    ...(d.kind === "split" ? folders.map((f) => `<div class="tile"><div class="k">${esc(f.key)}</div><div class="v">${fmtInt(f.num_images)}</div></div>`) : []),
  ];
  if (d.classes) {
    const inst = d.classes.reduce((a, c) => a + c.instances, 0);
    tiles.push(`<div class="tile"><div class="k">Boxes</div><div class="v">${fmtInt(inst)}</div></div>`);
    tiles.push(`<div class="tile"><div class="k">No label / empty</div><div class="v">${fmtInt(d.images_without_label)} / ${fmtInt(d.background_images)}</div></div>`);
  }
  const pathWarn = d.warnings.some((w) => w.includes("Repair data.yaml path"));
  const warnings = d.warnings.map((w) => `<div class="notice warn">${esc(w)}${
    w.includes("Repair data.yaml path") ? `<br><button class="btn small" data-act="repair">Repair data.yaml path → this folder</button>` : ""}</div>`).join("");
  const unknown = d.unknown_class_ids && Object.keys(d.unknown_class_ids).length
    ? `<div class="notice bad">Labels reference class indices with no name: ${Object.entries(d.unknown_class_ids).map(([k, v]) => `#${esc(k)} (${v}×)`).join(", ")}.</div>` : "";

  el.innerHTML = `
    <div class="card">
      <div class="detail-head">
        <div class="grow">
          ${DS.renaming ? `
            <div class="rename-row">
              <input id="ds-newname" type="text" value="${esc(d.name)}" aria-label="New dataset name">
              <button class="btn primary" data-act="rename-save">Save</button>
              <button class="btn" data-act="rename-cancel">Cancel</button>
            </div>
            <p class="muted small">Renames the folder. Symlinks and preprocess records in other datasets that point at it are updated too.</p>`
          : `<h1>${esc(d.name)} <button class="btn small" data-act="rename" title="Rename dataset folder">Rename</button></h1>`}
          <div class="path">${esc(d.path)}</div>
        </div>
        <span class="pill ${d.kind === "split" ? "split" : ""}" title="${d.kind === "split"
          ? "Has data.yaml with train/val splits — trainable directly"
          : "Flat images/ + labels/ export — merge it into a prepared dataset to train"}">${d.kind === "split" ? "Prepared · trainable" : "Source (images + labels)"}</span>
      </div>
      ${warnings}${unknown}
      <div class="tiles">${tiles.join("")}</div>
      <div class="btn-row">
        <button class="btn" data-act="preview">Preview images</button>
        <button class="btn" data-act="ls">Export to Label Studio</button>
        ${d.trainable ? `<button class="btn primary" data-act="train" ${pathWarn ? 'title="Repair the data.yaml path first"' : ""}>Train on this dataset</button>` : ""}
        ${d.kind === "flat" ? `<button class="btn primary" data-act="merge">Re-index for training…</button>` : ""}
        <button class="btn" data-act="refresh" title="Recount images and labels">Recount</button>
      </div>
      <div id="ds-classes">${renderClassTable(d)}</div>
    </div>`;
}

function renderClassTable(d) {
  if (!d.classes) return `<p class="muted">${spinner("Counting labels…")}</p>`;
  const max = Math.max(1, ...d.classes.map((c) => c.instances));
  const editing = DS.editingClasses;
  const rows = d.classes.map((c) => `
    <tr>
      <td class="idx">${c.index}</td>
      <td>${editing
        ? `<input type="text" class="cls-input" data-index="${c.index}" value="${esc(c.name)}" aria-label="Name of class ${c.index}">`
        : `<span class="swatch" style="background:${classColor(c.index)}"></span>${esc(c.name)}`}</td>
      <td class="num">${fmtInt(c.instances)}</td>
      <td class="num">${fmtInt(c.images)}</td>
      <td style="width:28%"><div class="bar" title="${c.instances} boxes"><span style="width:${(c.instances / max) * 100}%"></span></div></td>
    </tr>`).join("");
  return `
    <div class="table-tools">
      <h3>Class index → data name</h3>
      ${editing
        ? `<div class="btn-row" style="margin:0"><button class="btn" data-act="cls-cancel">Cancel</button><button class="btn primary" data-act="cls-save">Save names</button></div>`
        : `<button class="btn small" data-act="cls-edit" ${d.classes.length ? "" : "disabled"}>Edit names</button>`}
    </div>
    ${editing ? `<p class="muted small">Renaming keeps every index, so label files are untouched. Updates ${d.kind === "split" ? "data.yaml" : "classes.txt / notes.json"}. Class lists are merged <b>by name</b> when preparing datasets, so use the same spelling across sources.</p>` : ""}
    <div class="table-wrap"><table>
      <thead><tr><th>Index</th><th>Data name</th><th class="num">Boxes</th><th class="num">Images</th><th>Share of boxes</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="5" class="muted">No classes defined.</td></tr>`}</tbody>
    </table></div>`;
}

$("#ds-detail").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-act]");
  if (!btn) return;
  const d = DS.info;
  const act = btn.dataset.act;
  try {
    if (act === "rename") { DS.renaming = true; renderDatasetDetail(); $("#ds-newname").select(); }
    else if (act === "rename-cancel") { DS.renaming = false; renderDatasetDetail(); }
    else if (act === "rename-save") {
      const newName = $("#ds-newname").value.trim();
      btn.disabled = true;
      const r = await api("/datasets/rename", { json: { id: d.id, new_name: newName } });
      toast(`Renamed to ${newName}` + (r.symlinks_updated ? ` · ${r.symlinks_updated} links updated` : ""), "ok");
      await loadDatasets();
      selectDataset(r.id);
    }
    else if (act === "cls-edit") { DS.editingClasses = true; $("#ds-classes").innerHTML = renderClassTable(d); }
    else if (act === "cls-cancel") { DS.editingClasses = false; $("#ds-classes").innerHTML = renderClassTable(d); }
    else if (act === "cls-save") {
      const names = $$(".cls-input", $("#ds-classes")).sort((a, b) => a.dataset.index - b.dataset.index).map((i) => i.value.trim());
      btn.disabled = true;
      const r = await api("/datasets/classes", { method: "PUT", json: { id: d.id, names } });
      toast(`Class names saved (${r.updated_files.join(", ")})`, "ok");
      DS.editingClasses = false;
      d.classes.forEach((c, i) => (c.name = names[i]));
      $("#ds-classes").innerHTML = renderClassTable(d);
      loadDatasets();
    }
    else if (act === "repair") {
      const r = await api("/datasets/repair-yaml", { json: { id: d.id } });
      toast("data.yaml path set to " + r.path, "ok");
      await loadDatasets();
      selectDataset(d.id, { refresh: true });
    }
    else if (act === "refresh") selectDataset(d.id, { refresh: true });
    else if (act === "preview") go("preview", { ds: d.id });
    else if (act === "ls") go("labelstudio", { ds: d.id });
    else if (act === "train") go("train", { ds: d.id });
    else if (act === "merge") go("train", { sources: d.id });
  } catch (e) {
    btn.disabled = false;
    toast(e.message, "err");
  }
});
$("#ds-detail").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && ev.target.id === "ds-newname") $('[data-act="rename-save"]').click();
  if (ev.key === "Escape" && ev.target.id === "ds-newname") $('[data-act="rename-cancel"]').click();
});
$("#ds-list").addEventListener("click", (ev) => {
  const b = ev.target.closest(".ds-item");
  if (b) go("datasets", { ds: b.dataset.id });
});
$("#ds-filter").addEventListener("input", renderDatasetList);
$("#ds-refresh").addEventListener("click", async () => {
  await loadDatasets().catch((e) => toast(e.message, "err"));
  if (DS.selected) selectDataset(DS.selected, { refresh: true });
});

Tabs.datasets = {
  enter(params) {
    const id = params.get("ds");
    if (id && id !== DS.selected) selectDataset(id);
    else if (!id && !DS.selected && state.datasets.length) selectDataset(state.datasets[0].id);
  },
};

// ---- Import dialog ---------------------------------------------------------
const Imp = { src: "zip" };
$("#ds-import-open").addEventListener("click", () => {
  setStatus($("#imp-status"), "");
  $("#imp-progress").hidden = true;
  $("#imp-go").disabled = false;
  $("#import-dialog").showModal();
});
$("#imp-cancel").addEventListener("click", () => $("#import-dialog").close());
$$(".seg-btn").forEach((b) => b.addEventListener("click", () => {
  Imp.src = b.dataset.src;
  $$(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
  $$("[data-src-panel]").forEach((p) => (p.hidden = p.dataset.srcPanel !== Imp.src));
}));
const suggestName = (s) => s.replace(/\.zip$/i, "").replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80);
$("#imp-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f && !$("#imp-name").value) $("#imp-name").value = suggestName(f.name);
});
$("#imp-path").addEventListener("change", (e) => {
  const last = e.target.value.replace(/\/+$/, "").split("/").pop();
  if (last && !$("#imp-name").value) $("#imp-name").value = suggestName(last);
});
$("#imp-go").addEventListener("click", async () => {
  const name = $("#imp-name").value.trim();
  const group = $("#imp-group").value.trim();
  const status = $("#imp-status");
  if (!name) { setStatus(status, "Enter a dataset name.", "err"); return; }
  const btn = $("#imp-go");
  btn.disabled = true;
  try {
    let result;
    if (Imp.src === "zip") {
      const file = $("#imp-file").files[0];
      if (!file) throw new Error("Choose a zip file.");
      const bar = $("#imp-progress");
      bar.hidden = false; bar.value = 0;
      result = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `/datasets/import/upload?name=${encodeURIComponent(name)}&group=${encodeURIComponent(group)}`);
        xhr.setRequestHeader("Content-Type", "application/zip");
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) bar.value = (e.loaded / e.total) * 100;
          setStatus(status, e.loaded < e.total ? spinner(`Uploading… ${fmtBytes(e.loaded)} / ${fmtBytes(e.total)}`) : spinner("Extracting…"));
        };
        xhr.onload = () => {
          let body = {};
          try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
          if (xhr.status >= 200 && xhr.status < 300) resolve(body);
          else reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status})`));
        };
        xhr.onerror = () => reject(new Error("Network error during upload."));
        xhr.send(file);
      });
    } else {
      const source = $("#imp-path").value.trim();
      if (!source) throw new Error("Enter the folder path on the server.");
      setStatus(status, spinner("Importing…"));
      const mode = $('input[name="imp-mode"]:checked').value;
      result = await api("/datasets/import/folder", { json: { source_path: source, name, group, mode } });
    }
    $("#import-dialog").close();
    $("#imp-name").value = ""; $("#imp-file").value = ""; $("#imp-path").value = "";
    toast(`Imported ${result.datasets.length} dataset(s) into ${result.id}`, "ok");
    await loadDatasets();
    go("datasets", { ds: result.datasets[0]?.id || result.id });
  } catch (e) {
    setStatus(status, esc(e.message), "err");
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Box overlay rendering (preview grid + lightbox)
// ---------------------------------------------------------------------------
function boxesHtml(boxes, names, showTags = true) {
  return boxes.map((b) => {
    const left = (b.x - b.w / 2) * 100, top = (b.y - b.h / 2) * 100;
    const color = classColor(b.cls);
    const name = names[b.cls] ?? `#${b.cls}`;
    return `<div class="box ${top < 7 ? "below" : ""}" style="left:${left}%;top:${top}%;width:${b.w * 100}%;height:${b.h * 100}%;border-color:${color}">${
      showTags ? `<span class="tag" style="background:${color}">${esc(name)}</span>` : ""}</div>`;
  }).join("");
}
const fileUrl = (ds, folder, path, thumb) =>
  `/datasets/file?ds=${encodeURIComponent(ds)}&folder=${encodeURIComponent(folder)}&path=${encodeURIComponent(path)}${thumb ? "&thumb=1" : ""}`;

// ---------------------------------------------------------------------------
// Tab: Preview
// ---------------------------------------------------------------------------
const PV = { ds: null, folder: null, cls: "", q: "", unlabeled: false, offset: 0, limit: 48, total: 0, items: [], lb: -1, req: 0 };

function pvSetDataset(id, folder) {
  const d = byId(id);
  if (!d) return;
  PV.ds = id;
  $("#pv-dataset").value = id;
  $("#pv-folder").innerHTML = d.folders.map((f) =>
    `<option value="${esc(f.key)}">${esc(f.key)} (${fmtInt(f.num_images)})</option>`).join("");
  PV.folder = folder && d.folders.some((f) => f.key === folder) ? folder : d.folders[0]?.key;
  $("#pv-folder").value = PV.folder || "";
  $("#pv-class").innerHTML = `<option value="">All classes</option>` +
    d.class_names.map((n, i) => `<option value="${i}">${i} · ${esc(n)}</option>`).join("");
  PV.cls = ""; PV.offset = 0;
  pvLoad();
}

async function pvLoad() {
  if (!PV.ds || !PV.folder) { $("#pv-grid").innerHTML = `<div class="empty">This dataset has no browsable folder.</div>`; return; }
  const req = ++PV.req;
  $("#pv-grid").innerHTML = `<p class="muted">${spinner("Loading images…")}</p>`;
  const q = new URLSearchParams({ id: PV.ds, folder: PV.folder, offset: PV.offset, limit: PV.limit });
  if (PV.cls !== "") q.set("class_id", PV.cls);
  if (PV.q) q.set("q", PV.q);
  if (PV.unlabeled) q.set("unlabeled", "true");
  try {
    const r = await api("/datasets/images?" + q);
    if (req !== PV.req) return;
    PV.total = r.total; PV.items = r.items;
    pvRender();
  } catch (e) {
    $("#pv-grid").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`;
  }
}

function pvRender() {
  const names = byId(PV.ds)?.class_names || [];
  const grid = $("#pv-grid");
  grid.classList.toggle("hide-boxes", !$("#pv-boxes").checked);
  grid.innerHTML = PV.items.map((it, i) => `
    <button class="thumb" data-i="${i}" title="${esc(it.path)}">
      <div class="frame"><div class="imgbox">
        <img loading="lazy" src="${fileUrl(PV.ds, PV.folder, it.path, true)}" alt="${esc(it.path)}">
        ${boxesHtml(it.boxes, names)}
      </div></div>
      <div class="cap"><span class="fn">${esc(it.path)}</span>${
        !it.has_label ? `<span class="pill warn">no label</span>` : `<span class="pill">${it.boxes.length} box${it.boxes.length === 1 ? "" : "es"}</span>`}</div>
    </button>`).join("") || `<div class="empty">No images match.</div>`;
  const from = PV.total ? PV.offset + 1 : 0, to = Math.min(PV.offset + PV.limit, PV.total);
  const pager = `<span>${fmtInt(from)}–${fmtInt(to)} of ${fmtInt(PV.total)}</span>
    <button class="btn small" data-pg="-1" ${PV.offset <= 0 ? "disabled" : ""}>← Prev</button>
    <button class="btn small" data-pg="1" ${to >= PV.total ? "disabled" : ""}>Next →</button>`;
  $("#pv-pager").innerHTML = pager;
  $("#pv-pager-top").innerHTML = pager;
}

function pvPage(dir) {
  PV.offset = Math.max(0, PV.offset + dir * PV.limit);
  return pvLoad();
}
[$("#pv-pager"), $("#pv-pager-top")].forEach((p) => p.addEventListener("click", (e) => {
  const b = e.target.closest("[data-pg]");
  if (b) { pvPage(+b.dataset.pg); if (p.id === "pv-pager") window.scrollTo({ top: 0 }); }
}));
$("#pv-dataset").addEventListener("change", (e) => go("preview", { ds: e.target.value }));
$("#pv-folder").addEventListener("change", (e) => { PV.folder = e.target.value; PV.offset = 0; pvLoad(); });
$("#pv-class").addEventListener("change", (e) => { PV.cls = e.target.value; PV.offset = 0; pvLoad(); });
let pvSearchTimer;
$("#pv-search").addEventListener("input", (e) => {
  clearTimeout(pvSearchTimer);
  pvSearchTimer = setTimeout(() => { PV.q = e.target.value.trim(); PV.offset = 0; pvLoad(); }, 300);
});
$("#pv-unlabeled").addEventListener("change", (e) => { PV.unlabeled = e.target.checked; PV.offset = 0; pvLoad(); });
$("#pv-boxes").addEventListener("change", (e) => {
  $("#pv-grid").classList.toggle("hide-boxes", !e.target.checked);
  $("#lb-frame").classList.toggle("hide-boxes", !e.target.checked);
});
$("#pv-grid").addEventListener("click", (e) => {
  const t = e.target.closest(".thumb");
  if (t) lbOpen(+t.dataset.i);
});

Tabs.preview = {
  enter(params) {
    const id = params.get("ds") || PV.ds || DS.selected || state.datasets[0]?.id;
    if (id && id !== PV.ds) pvSetDataset(id, params.get("folder"));
  },
};

// ---- Lightbox ---------------------------------------------------------------
function lbOpen(i) {
  PV.lb = i;
  const it = PV.items[i];
  if (!it) return;
  const names = byId(PV.ds)?.class_names || [];
  $("#lb-frame").classList.toggle("hide-boxes", !$("#pv-boxes").checked);
  $("#lb-frame").innerHTML = `<div class="imgbox"><img src="${fileUrl(PV.ds, PV.folder, it.path, false)}" alt="${esc(it.path)}">${boxesHtml(it.boxes, names)}</div>`;
  $("#lb-name").textContent = it.path;
  $("#lb-pos").textContent = `${fmtInt(PV.offset + i + 1)} of ${fmtInt(PV.total)} · ${PV.folder}`;
  $("#lb-boxes").innerHTML = !it.has_label
    ? `<div class="notice warn">No label file for this image.</div>`
    : it.boxes.length === 0 ? `<p class="muted">Label file is empty (background image).</p>`
      : `<ul class="box-list">${it.boxes.map((b) => `<li><span class="swatch" style="background:${classColor(b.cls)}"></span>
          <span>${b.cls} · ${esc(names[b.cls] ?? "unknown")}</span>
          <span class="coords" title="x_center y_center width height (normalized)">${b.x.toFixed(3)} ${b.y.toFixed(3)} ${b.w.toFixed(3)} ${b.h.toFixed(3)}</span></li>`).join("")}</ul>`;
  if (!$("#lightbox").open) $("#lightbox").showModal();
}
async function lbStep(dir) {
  let i = PV.lb + dir;
  if (i < 0) {
    if (PV.offset <= 0) return;
    await pvPage(-1); i = PV.items.length - 1;
  } else if (i >= PV.items.length) {
    if (PV.offset + PV.limit >= PV.total) return;
    await pvPage(1); i = 0;
  }
  lbOpen(i);
}
$("#lb-close").addEventListener("click", () => $("#lightbox").close());
$("#lb-prev").addEventListener("click", () => lbStep(-1));
$("#lb-next").addEventListener("click", () => lbStep(1));
$("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("#lightbox").close(); });
$("#lightbox").addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { e.preventDefault(); lbStep(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); lbStep(1); }
  else if (e.key.toLowerCase() === "b") { $("#pv-boxes").click(); }
});

// ---------------------------------------------------------------------------
// Tab: Label Studio
// ---------------------------------------------------------------------------
function lsSetDataset(id) {
  const d = byId(id);
  if (!d) return;
  $("#ls-dataset").value = id;
  $("#ls-folders").innerHTML = d.folders.filter((f) => !f.key.includes("/")).map((f) =>
    `<label class="check"><input type="checkbox" value="${esc(f.key)}" checked> ${esc(f.key)} <span class="muted">(${fmtInt(f.num_images)})</span></label>`).join("");
  $("#ls-title").value = d.name.slice(0, 50);
  $("#ls-storage").placeholder = d.folders[0]?.path || d.path;
}
function lsMode() { return $('input[name="ls-mode"]:checked').value; }
$$('input[name="ls-mode"]').forEach((r) => r.addEventListener("change", () => {
  $$("[data-mode]").forEach((el) => (el.hidden = el.dataset.mode !== lsMode()));
}));
$("#ls-dataset").addEventListener("change", (e) => lsSetDataset(e.target.value));
function lsBody() {
  const folders = $$("#ls-folders input:checked").map((i) => i.value);
  if (!folders.length) throw new Error("Select at least one folder.");
  const body = {
    id: $("#ls-dataset").value,
    folders,
    mode: lsMode(),
    as_predictions: $('input[name="ls-as"]:checked').value === "predictions",
    include_unlabeled: $("#ls-unlabeled").checked,
  };
  if (body.mode === "url") body.api_base_url = $("#ls-base").value.trim() || location.origin;
  else {
    body.document_root = $("#ls-docroot").value.trim();
    if (!body.document_root) throw new Error("Enter Label Studio's LOCAL_FILES_DOCUMENT_ROOT.");
  }
  store.set("ls-base", $("#ls-base").value.trim());
  store.set("ls-docroot", $("#ls-docroot").value.trim());
  return body;
}
$("#ls-download").addEventListener("click", async () => {
  const st = $("#ls-download-status"), btn = $("#ls-download");
  try {
    const body = lsBody();
    btn.disabled = true;
    setStatus(st, spinner("Building tasks (reads every image's size)…"));
    const res = await api("/datasets/labelstudio/export", { json: body, raw: true });
    const blob = await res.blob();
    const name = (res.headers.get("content-disposition") || "").match(/filename="([^"]+)"/)?.[1] || "labelstudio_export.zip";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    setStatus(st, `Downloaded ${esc(name)} — ${fmtInt(res.headers.get("x-task-count"))} tasks.`, "ok");
  } catch (e) { setStatus(st, esc(e.message), "err"); }
  finally { btn.disabled = false; }
});
$("#ls-push").addEventListener("click", async () => {
  const st = $("#ls-push-status"), btn = $("#ls-push");
  try {
    const body = lsBody();
    body.ls_url = $("#ls-url").value.trim();
    body.token = $("#ls-token").value.trim();
    body.title = $("#ls-title").value.trim();
    if (!body.ls_url || !body.token || !body.title) throw new Error("Label Studio URL, API token and project title are required.");
    if (body.mode === "local_files") body.storage_path = $("#ls-storage").value.trim() || $("#ls-storage").placeholder;
    store.set("ls-url", body.ls_url);
    btn.disabled = true;
    setStatus(st, spinner("Creating project and importing tasks…"));
    const r = await api("/datasets/labelstudio/push", { json: body });
    const frag = document.createElement("div");
    frag.innerHTML = `Imported ${fmtInt(r.tasks)} tasks into <a href="${esc(r.project_url)}" target="_blank" rel="noopener">project #${esc(r.project_id)}</a>.` +
      (r.warnings || []).map((w) => `<div class="notice warn">${esc(w)}</div>`).join("");
    setStatus(st, frag, "ok");
  } catch (e) { setStatus(st, esc(e.message), "err"); }
  finally { btn.disabled = false; }
});

Tabs.labelstudio = {
  enter(params) {
    if (!$("#ls-base").value) $("#ls-base").value = store.get("ls-base") || location.origin;
    if (!$("#ls-docroot").value) $("#ls-docroot").value = store.get("ls-docroot") || state.baseDir;
    if (!$("#ls-url").value) $("#ls-url").value = store.get("ls-url");
    const id = params.get("ds") || $("#ls-dataset").value || DS.selected;
    if (id) lsSetDataset(id);
  },
};

// ---------------------------------------------------------------------------
// Tab: Train
// ---------------------------------------------------------------------------
const TR = { selectedSources: new Set(), defaultsLoaded: false };

// Sensible defaults for the 1.labeled -> 2.re_index_label -> 3.preprocessed layout.
const looksReindex = (d) => d.kind === "flat" && /re_?index/i.test(d.id);
const looksLabeledGroup = (d) => d.kind === "flat" && !looksReindex(d);

function fillPipelineSelects() {
  const flat = (d) => d.kind === "flat";
  const pick = (sel, fallback) => { const v = $(sel).value; return v && byId(v) ? v : fallback; };
  const riDefault = pick("#ri-existing", state.datasets.find(looksReindex)?.id);
  $("#ri-existing").innerHTML = datasetOptions(flat, riDefault) || `<option value="">No flat folders yet — create a new one</option>`;
  const spSource = pick("#sp-source", $("#ri-existing").value || state.datasets.find(looksReindex)?.id);
  $("#sp-source").innerHTML = datasetOptions(flat, spSource) || `<option value="">No flat folders</option>`;
  const spTarget = pick("#sp-existing", state.datasets.find((d) => d.preprocess_target)?.id);
  $("#sp-existing").innerHTML = datasetOptions((d) => d.preprocess_target, spTarget) ||
    `<option value="">None built by splitting yet — create a new one</option>`;
  if (!state.datasets.some((d) => d.preprocess_target)) { $('input[name="sp-target"][value="new"]').click(); }
  renderSourceChecklist();
}

function renderSourceChecklist() {
  const f = $("#ri-filter").value.trim().toLowerCase();
  const target = $('input[name="ri-target"]:checked').value === "existing" ? $("#ri-existing").value : null;
  const list = state.datasets.filter((d) => d.kind === "flat" && d.id !== target && (!f || d.id.toLowerCase().includes(f)));
  let html = "", last = null;
  for (const d of list) {
    if (d.group !== last) { html += `<div class="group-label">${esc(groupLabel(d.group))}</div>`; last = d.group; }
    html += `<label class="check"><input type="checkbox" value="${esc(d.id)}" ${TR.selectedSources.has(d.id) ? "checked" : ""}>
      <span>${esc(d.name)}</span><span class="meta">${fmtInt(d.num_images)} img · ${d.num_classes} cls</span></label>`;
  }
  $("#ri-sources").innerHTML = html || `<p class="muted small">No labeled folders (images/ + labels/) found.</p>`;
  $("#ri-count").textContent = TR.selectedSources.size ? `${TR.selectedSources.size} selected` : "";
}
$("#ri-filter").addEventListener("input", renderSourceChecklist);
$("#ri-existing").addEventListener("change", () => { TR.selectedSources.delete($("#ri-existing").value); renderSourceChecklist(); });
$("#ri-sources").addEventListener("change", (e) => {
  if (e.target.type !== "checkbox") return;
  if (e.target.checked) TR.selectedSources.add(e.target.value); else TR.selectedSources.delete(e.target.value);
  $("#ri-count").textContent = TR.selectedSources.size ? `${TR.selectedSources.size} selected` : "";
});
for (const [name, attr] of [["ri-target", "riTarget"], ["sp-target", "spTarget"]]) {
  $$(`input[name="${name}"]`).forEach((r) => r.addEventListener("change", () => {
    const v = $(`input[name="${name}"]:checked`).value;
    $$(`[data-${name}]`).forEach((el) => (el.hidden = el.dataset[attr] !== v));
    if (name === "ri-target") renderSourceChecklist();
  }));
}

$("#ri-run").addEventListener("click", async () => {
  const st = $("#ri-status"), btn = $("#ri-run");
  try {
    if (!TR.selectedSources.size) throw new Error("Select at least one labeled folder.");
    const body = { sources: [...TR.selectedSources] };
    if ($('input[name="ri-target"]:checked').value === "existing") {
      body.output_id = $("#ri-existing").value;
      if (!body.output_id) throw new Error("Choose the re-index folder, or create a new one.");
      body.sources = body.sources.filter((s) => s !== body.output_id);
    } else {
      body.output_name = $("#ri-name").value.trim();
      body.output_group = $("#ri-group").value.trim();
      if (!body.output_name) throw new Error("Enter a name for the new folder.");
    }
    btn.disabled = true;
    setStatus(st, spinner("Copying images and remapping labels…"));
    const r = await api("/datasets/reindex", { json: body });
    const rows = r.sources.map((x) => `<tr><td>${esc(x.source)}</td><td class="num">${fmtInt(x.added)}</td><td class="num">${fmtInt(x.already_merged)}</td><td class="num">${fmtInt(x.no_label)}</td></tr>`).join("");
    setStatus(st, `<div class="notice good"><b>${esc(r.id)}</b>: added ${fmtInt(r.added)} images · now ${r.class_names.length} classes` +
      (r.new_classes.length ? `<br>New classes appended: ${r.new_classes.map(esc).join(", ")}` : "") + `</div>
      <div class="table-wrap"><table><thead><tr><th>Source</th><th class="num">Added</th><th class="num">Already merged</th><th class="num">No label</th></tr></thead><tbody>${rows}</tbody></table></div>`);
    TR.selectedSources.clear();
    await loadDatasets();
    $("#ri-existing").value = r.id;
    $("#sp-source").value = r.id;
    renderSourceChecklist();
  } catch (e) { setStatus(st, esc(e.message), "err"); }
  finally { btn.disabled = false; }
});

async function runSplit(extra = {}) {
  const st = $("#sp-status"), btn = $("#sp-run");
  const source = $("#sp-source").value;
  const body = { sources: [source], val_ratio: +$("#sp-val").value || 0, test_ratio: +$("#sp-test").value || 0, ...extra };
  try {
    if (!source) throw new Error("Choose the re-indexed folder to split.");
    if ($('input[name="sp-target"]:checked').value === "existing") {
      body.output_id = $("#sp-existing").value;
      if (!body.output_id) throw new Error("Choose a prepared dataset, or create a new one.");
    } else {
      body.output_name = $("#sp-name").value.trim();
      body.output_group = $("#sp-group").value.trim();
      if (!body.output_name) throw new Error("Enter a name for the new dataset.");
    }
    btn.disabled = true;
    setStatus(st, spinner("Splitting…"));
    const r = await api("/datasets/preprocess", { json: body });
    const a = r.added;
    setStatus(st, `<div class="notice good"><b>${esc(r.id)}</b>: added ${fmtInt(a.train)} train / ${fmtInt(a.val)} val / ${fmtInt(a.test)} test images. Now ${r.num_classes} classes.` +
      (a.new_classes.length ? `<br>New classes: ${a.new_classes.map(esc).join(", ")}` : "") +
      (a.skipped_no_label.length ? `<br>${a.skipped_no_label.length} image(s) skipped (no label file).` : "") +
      (a.train + a.val + a.test === 0 ? `<br>No new images — everything was already split.` : "") + `</div>`);
    await loadDatasets();
    $("#tr-dataset").value = r.id;
    trDatasetInfo();
  } catch (e) {
    if (e.detail?.code === "stale_state") {
      const box = document.createElement("div");
      box.className = "notice warn";
      box.innerHTML = `${esc(e.message)}<br><button class="btn small">Repoint ${fmtInt(e.detail.count)} records to ${esc(source)}, then split</button>
        <span class="muted small">(a backup of the records file is kept)</span>`;
      $("button", box).addEventListener("click", async (ev) => {
        ev.target.disabled = true;
        try {
          const r = await api("/datasets/repoint-state", { json: { target_id: body.output_id, source_id: source } });
          toast(`Repointed ${r.repointed} records`, "ok");
          runSplit();
        } catch (err) { setStatus(st, esc(err.message), "err"); }
      });
      setStatus(st, box);
    } else setStatus(st, esc(e.message), "err");
  } finally { btn.disabled = false; }
}
$("#sp-run").addEventListener("click", () => runSplit());

function trDatasetInfo() {
  const d = byId($("#tr-dataset").value);
  const el = $("#tr-dataset-info");
  if (!d) { el.innerHTML = ""; return; }
  el.innerHTML = `${d.num_classes} classes · ` + d.folders.map((f) => `${esc(f.key)} ${fmtInt(f.num_images)}`).join(" · ") +
    ` · <a href="#datasets?ds=${encodeURIComponent(d.id)}">details</a>` +
    d.warnings.map((w) => `<div class="notice warn">${esc(w)}</div>`).join("");
}
$("#tr-dataset").addEventListener("change", trDatasetInfo);

const newJobName = () => {
  const p = (n) => String(n).padStart(2, "0");
  const d = new Date();
  return `job_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};

async function trLoadDefaults() {
  if (TR.defaultsLoaded) return;
  TR.defaultsLoaded = true;
  try {
    const [models, sys] = await Promise.all([api("/models/available"), api("/system/status")]);
    $("#model-options").innerHTML = models.models.map((m) =>
      `<option value="${esc(m.value)}">${esc(m.kind === "trained" ? "trained: " + m.label : m.kind === "local" ? "local: " + m.label : "download: " + m.label)}</option>`).join("");
    $("#tr-device").innerHTML = sys.gpus.map((g) => `<option value="cuda:${g.index}">cuda:${g.index} · ${esc(g.name)}</option>`).join("") +
      `<option value="cpu">cpu</option>`;
  } catch (e) {
    $("#tr-device").innerHTML = `<option>cuda:0</option><option>cpu</option>`;
  }
}

$("#tr-submit").addEventListener("click", async () => {
  const st = $("#tr-status"), btn = $("#tr-submit");
  try {
    const d = byId($("#tr-dataset").value);
    if (!d) throw new Error("Choose a prepared dataset (build one on the left if needed).");
    const job = $("#tr-job").value.trim();
    if (!/^[A-Za-z0-9_-]+$/.test(job)) throw new Error("Job name may only contain letters, digits, '_' and '-'.");
    const num = (id) => Number($(id).value);
    const training = {
      model: $("#tr-model").value.trim() || "yolov8m.pt",
      epochs: num("#tr-epochs"), img_size: num("#tr-imgsz"), batch_size: num("#tr-batch"), patience: num("#tr-patience"),
      optimizer: $("#tr-optimizer").value, lr0: num("#tr-lr0"), scale: num("#tr-scale"),
      mosaic: num("#tr-mosaic"), mixup: num("#tr-mixup"), copy_paste: num("#tr-copypaste"),
      plots: $("#tr-plots").checked, cache: $("#tr-cache").checked, device: $("#tr-device").value,
    };
    const q = new URLSearchParams({ job_name: job, dataset_path: d.path });
    const out = $("#tr-output").value.trim();
    if (out) q.set("output_path", out);
    btn.disabled = true;
    setStatus(st, spinner("Validating dataset and queueing…"));
    const r = await api("/train_local_data?" + q, { json: { training } });
    setStatus(st, `Queued <b>${esc(r.job_name)}</b> (position ${r.position}).`, "ok");
    toast(`Job ${r.job_name} queued`, "ok");
    $("#tr-job").value = newJobName();
    go("monitor", { job: r.job_id });
  } catch (e) { setStatus(st, esc(e.message), "err"); }
  finally { btn.disabled = false; }
});

Tabs.train = {
  enter(params) {
    trLoadDefaults();
    if (!$("#tr-job").value) $("#tr-job").value = newJobName();
    const ds = params.get("ds");
    if (ds && byId(ds)?.trainable) $("#tr-dataset").value = ds;
    const sources = params.get("sources");
    if (sources) { sources.split(",").forEach((s) => TR.selectedSources.add(s)); renderSourceChecklist(); }
    $("#tab-train").dataset.ready = "1";
    trDatasetInfo();
  },
};

// ---------------------------------------------------------------------------
// Charts (SVG line chart with crosshair tooltip)
// ---------------------------------------------------------------------------
function niceTicks(min, max, count = 4) {
  if (min === max) { max = min + 1; }
  const span = max - min;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || mag * 10;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

const tickFmt = (v) => String(Number(v.toPrecision(4)));

function lineChart(el, { series, data, fmt = (v) => v.toFixed(3), zeroBased = false }) {
  const pts = series.map((s) => data.map((d) => ({ x: d.epoch, y: d[s.key] })).filter((p) => p.y != null && isFinite(p.y)));
  if (!pts.some((p) => p.length)) { el.innerHTML = `<div class="chart-empty">Waiting for the first epoch…</div>`; return; }
  const W = Math.max(280, el.clientWidth || 480), H = 200, m = { l: 46, r: 12, t: 10, b: 24 };
  const ys = pts.flat().map((p) => p.y);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (zeroBased) yMin = 0;
  const yt = niceTicks(yMin, yMax);
  yMin = yt[0]; yMax = yt[yt.length - 1];
  const xs = data.map((d) => d.epoch);
  const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);
  const X = (v) => m.l + ((v - xMin) / (xMax - xMin)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
  const xt = niceTicks(xMin, xMax, 5).filter((v) => v >= xMin && v <= xMax && Number.isInteger(v));

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(series.map((s) => s.label).join(", "))} by epoch">`;
  for (const v of yt) svg += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="axis-text" x="${m.l - 6}" y="${Y(v) + 3.5}" text-anchor="end">${tickFmt(v)}</text>`;
  for (const v of xt) svg += `<text class="axis-text" x="${X(v)}" y="${H - 6}" text-anchor="middle">${v}</text>`;
  series.forEach((s, i) => {
    const p = pts[i];
    if (!p.length) return;
    svg += `<path class="line" stroke="var(${s.color})" d="${p.map((q, j) => `${j ? "L" : "M"}${X(q.x).toFixed(1)},${Y(q.y).toFixed(1)}`).join("")}"/>`;
    if (p.length === 1) svg += `<circle class="dot" r="4" cx="${X(p[0].x)}" cy="${Y(p[0].y)}" fill="var(${s.color})"/>`;
  });
  svg += `<g class="hover" visibility="hidden"><line class="xhair" y1="${m.t}" y2="${H - m.b}"/>${series.map((s) => `<circle class="dot" r="4" fill="var(${s.color})"/>`).join("")}</g>`;
  svg += `<rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" class="hit"/></svg>`;
  el.innerHTML = svg + `<div class="tooltip" hidden></div>`;

  const svgEl = $("svg", el), hover = $(".hover", el), tip = $(".tooltip", el);
  const move = (ev) => {
    const r = svgEl.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    let best = data[0];
    for (const d of data) if (Math.abs(X(d.epoch) - px) < Math.abs(X(best.epoch) - px)) best = d;
    const x = X(best.epoch);
    hover.setAttribute("visibility", "visible");
    $("line", hover).setAttribute("x1", x); $("line", hover).setAttribute("x2", x);
    $$("circle", hover).forEach((c, i) => {
      const v = best[series[i].key];
      c.setAttribute("visibility", v == null ? "hidden" : "visible");
      if (v != null) { c.setAttribute("cx", x); c.setAttribute("cy", Y(v)); }
    });
    tip.hidden = false;
    tip.innerHTML = `<div class="muted">Epoch ${best.epoch}</div>` + series.map((s) =>
      `<div class="t-row"><i class="swatch" style="background:var(${s.color});width:10px;height:2px"></i>${esc(s.label)}<b>${best[s.key] == null ? "–" : fmt(best[s.key])}</b></div>`).join("");
    const left = (x / W) * r.width;
    tip.style.top = "36px";
    tip.style.left = left > r.width / 2 ? "" : `${left + 26}px`;
    tip.style.right = left > r.width / 2 ? `${r.width - left + 2}px` : "";
  };
  svgEl.addEventListener("pointermove", move);
  svgEl.addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); tip.hidden = true; });
}

// ---------------------------------------------------------------------------
// Tab: Monitor
// ---------------------------------------------------------------------------
const MON = { jobs: [], selected: null, detail: null, log: "", logOffset: -1, histLen: -1, timer: null, sysTimer: null, autoscroll: true, renderedFor: null };

const STATUS_LABEL = { queued: "Queued", started: "Running", finished: "Finished", failed: "Failed", canceled: "Canceled" };
const STAGE_LABEL = {
  starting: "Starting worker", downloading: "Downloading dataset", loading: "Loading model", preparing: "Preparing data",
  training: "Training", validating: "Validating", trained: "Training done", evaluating: "Final evaluation", finished: "Complete",
};

async function monRefreshQueue() {
  try {
    const q = await api("/queue");
    MON.jobs = q.jobs.sort((a, b) => (parseTime(b.created_at) || 0) - (parseTime(a.created_at) || 0));
    const active = q.started + q.queued;
    const badge = $("#queue-badge");
    badge.hidden = !active;
    badge.textContent = active;
    badge.title = `${q.started} running, ${q.queued} queued`;
    $("#queue-summary").textContent = `${q.started} running · ${q.queued} queued`;
    if (state.tab === "monitor") monRenderList();
  } catch { /* API unreachable; retry next tick */ }
}

function monRenderList() {
  $("#job-list").innerHTML = MON.jobs.map((j) => `
    <button class="job-item ${j.job_id === MON.selected ? "active" : ""}" data-id="${esc(j.job_id)}">
      <div class="jn">${esc(j.job_name || j.job_id)}</div>
      <div class="jm"><span class="status ${j.status}">${STATUS_LABEL[j.status] || j.status}</span><span>${fmtTime(j.created_at)}</span></div>
    </button>`).join("") || `<p class="muted small">No jobs yet.</p>`;
}
$("#job-list").addEventListener("click", (e) => {
  const b = e.target.closest(".job-item");
  if (b) go("monitor", { job: b.dataset.id });
});

function monSelect(id) {
  if (MON.selected === id) return;
  MON.selected = id; MON.detail = null; MON.log = ""; MON.logOffset = -1; MON.histLen = -1; MON.renderedFor = null;
  monRenderList();
  $("#job-detail").innerHTML = `<p class="muted">${spinner("Loading job…")}</p>`;
  monTick();
}

function cleanLog(raw) {
  return raw.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").split("\n").map((line) => {
    if (!line.includes("\r")) return line;
    const parts = line.split("\r").filter((p) => p.length);
    return parts.length ? parts[parts.length - 1] : "";
  }).join("\n");
}

async function monTick() {
  const id = MON.selected;
  if (!id) return;
  try {
    const d = await api(`/jobs/${encodeURIComponent(id)}/monitor`);
    if (MON.selected !== id) return;
    MON.detail = d;
    monRenderDetail();
    if (d.log_size > MON.logOffset || MON.logOffset < 0) {
      const r = await api(`/jobs/${encodeURIComponent(id)}/logs?offset=${MON.logOffset}&max_bytes=400000`);
      if (MON.selected !== id) return;
      if (MON.logOffset < 0 && r.offset > 0) MON.log = `… (showing the last ${fmtBytes(r.next_offset - r.offset)}; earlier output in logs/jobs/${id}/train.log)\n`;
      MON.log += r.text;
      if (MON.log.length > 1_500_000) MON.log = MON.log.slice(-1_200_000);
      MON.logOffset = r.next_offset;
      monRenderLog();
    }
  } catch (e) {
    if (MON.selected === id && !MON.detail) $("#job-detail").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`;
  }
}

function monRenderLog() {
  const pre = $("#job-log");
  if (!pre) return;
  const text = cleanLog(MON.log);
  const nearBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
  pre.textContent = text || "(no output yet)";
  if (MON.autoscroll && (nearBottom || pre.dataset.init !== "1")) pre.scrollTop = pre.scrollHeight;
  pre.dataset.init = "1";
}

function monSkeleton(d) {
  $("#job-detail").innerHTML = `
    <div class="card">
      <div class="detail-head">
        <div class="grow"><h1 id="jd-name"></h1><div class="path" id="jd-dataset"></div></div>
        <span id="jd-status"></span>
        <button class="btn danger" id="jd-cancel" hidden>Cancel job</button>
      </div>
      <div id="jd-notice"></div>
      <div class="tiles" id="jd-tiles"></div>
      <div id="jd-progress"></div>
      <div class="losses" id="jd-losses"></div>
      <div class="charts">
        <div class="chart-card"><div class="chart-title">Training loss</div><div class="legend" id="lg-loss"></div><div class="chart" id="ch-loss"></div></div>
        <div class="chart-card"><div class="chart-title">Validation mAP</div><div class="legend" id="lg-map"></div><div class="chart" id="ch-map"></div></div>
      </div>
      <details id="jd-table"><summary class="muted">Epoch table</summary><div class="table-wrap" style="max-height:320px;overflow:auto;margin-top:8px"></div></details>
      <div id="jd-result"></div>
      <div class="console-head">
        <h3>Worker output</h3>
        <label class="check"><input type="checkbox" id="jd-autoscroll" ${MON.autoscroll ? "checked" : ""}> Follow</label>
        <button class="btn small" id="jd-fulllog">Load full log</button>
      </div>
      <pre class="console" id="job-log">(loading…)</pre>
      <details style="margin-top:14px"><summary class="muted">Job configuration</summary><dl class="kv" id="jd-config" style="margin-top:8px"></dl></details>
    </div>`;
  $("#jd-cancel").addEventListener("click", async () => {
    const btn = $("#jd-cancel");
    if (btn.dataset.confirm !== "1") { btn.dataset.confirm = "1"; btn.textContent = "Click again to cancel"; setTimeout(() => { btn.dataset.confirm = ""; btn.textContent = "Cancel job"; }, 4000); return; }
    btn.disabled = true;
    try { const r = await api(`/jobs/${encodeURIComponent(d.job_id)}`, { method: "DELETE" }); toast(r.message, "ok"); monTick(); monRefreshQueue(); }
    catch (e) { toast(e.message, "err"); btn.disabled = false; }
  });
  $("#jd-autoscroll").addEventListener("change", (e) => { MON.autoscroll = e.target.checked; });
  $("#jd-fulllog").addEventListener("click", async () => {
    try {
      const r = await api(`/jobs/${encodeURIComponent(d.job_id)}/logs?offset=0&max_bytes=5000000`);
      MON.log = (r.next_offset < r.size ? "… (first 5 MB only)\n" : "") + r.text;
      MON.logOffset = r.next_offset;
      $("#job-log").dataset.init = "";
      monRenderLog();
    } catch (e) { toast(e.message, "err"); }
  });
  for (const [id, series] of [["#lg-loss", LOSS_SERIES], ["#lg-map", MAP_SERIES]]) {
    $(id).innerHTML = series.map((s) => `<span><i style="background:var(${s.color})"></i>${esc(s.label)}</span>`).join("");
  }
}

const LOSS_SERIES = [
  { key: "train/box_loss", label: "box", color: "--series-1" },
  { key: "train/cls_loss", label: "cls", color: "--series-2" },
  { key: "train/dfl_loss", label: "dfl", color: "--series-3" },
];
const MAP_SERIES = [
  { key: "metrics/mAP50(B)", label: "mAP50", color: "--series-1" },
  { key: "metrics/mAP50-95(B)", label: "mAP50-95", color: "--series-2" },
];

function monRenderDetail() {
  const d = MON.detail;
  if (MON.renderedFor !== d.job_id) { monSkeleton(d); MON.renderedFor = d.job_id; }
  const p = d.progress || {};
  const hist = p.history || [];
  const running = d.status === "started";

  $("#jd-name").textContent = d.job_name || d.job_id;
  $("#jd-dataset").textContent = d.dataset_path ? "dataset: " + d.dataset_path : d.source === "roboflow" ? "dataset: Roboflow download" : "";
  $("#jd-status").innerHTML = `<span class="status ${d.status}">${STATUS_LABEL[d.status] || d.status}</span>`;
  $("#jd-cancel").hidden = !(d.status === "queued" || d.status === "started");

  // Notices
  let notice = "";
  if (d.status === "queued") notice = `<div class="notice">Waiting in queue${d.position ? ` — position ${d.position}` : ""}. Jobs run one at a time.</div>`;
  if (d.status === "failed" && d.error) notice = `<div class="notice bad"><b>Job failed.</b><pre class="err-pre">${esc(d.error.trim().split("\n").slice(-25).join("\n"))}</pre></div>`;
  if (d.status === "canceled") notice = `<div class="notice warn">Job was canceled.</div>`;
  $("#jd-notice").innerHTML = notice;

  // Tiles
  const started = parseTime(d.started_at), ended = parseTime(d.ended_at);
  const elapsed = started ? ((ended || new Date()) - started) / 1000 : null;
  const last = hist[hist.length - 1];
  let eta = null;
  if (running && p.train_started_at && last && p.epochs) {
    const perEpoch = (last.time - p.train_started_at) / hist.length;
    eta = perEpoch * (p.epochs - hist.length) - (Date.now() / 1000 - last.time);
  }
  const best = hist.reduce((b, h) => (h["metrics/mAP50-95(B)"] ?? -1) > (b?.["metrics/mAP50-95(B)"] ?? -1) ? h : b, null);
  const stage = running || d.status === "finished" ? (STAGE_LABEL[p.stage] || p.stage || "Starting") : (STATUS_LABEL[d.status] || d.status);
  const proc = d.process || {};
  $("#jd-tiles").innerHTML = [
    ["Stage", stage, true],
    ["Epoch", p.epochs ? `${p.epoch || 0} / ${p.epochs}` : "–"],
    ["Elapsed", fmtDuration(elapsed)],
    ["ETA (max)", running ? fmtDuration(eta) : "–"],
    ["Best mAP50-95", best ? `${fmtPct(best["metrics/mAP50-95(B)"])}` : "–", false, best ? `epoch ${best.epoch}` : ""],
    ...(p.final_val ? [["best.pt mAP50 / 50-95", `${fmtPct(p.final_val["metrics/mAP50(B)"])} / ${fmtPct(p.final_val["metrics/mAP50-95(B)"])}`, true, "final validation"]] : []),
    ["Precision / Recall", last ? `${fmtPct(last["metrics/precision(B)"], 0)} / ${fmtPct(last["metrics/recall(B)"], 0)}` : "–", true],
    ["Process", proc.pid ? `PID ${proc.pid}` : "–", true, proc.alive ? "alive" : proc.exitcode != null ? `exit code ${proc.exitcode}` : ""],
  ].map(([k, v, sm, sub]) => `<div class="tile"><div class="k">${k}</div><div class="v ${sm ? "sm" : ""}">${esc(v)}</div>${sub ? `<div class="k">${esc(sub)}</div>` : ""}</div>`).join("");

  // Progress bars
  let prog = "";
  if (p.epochs) {
    const done = Math.min(hist.length, p.epochs);
    const frac = Math.min(1, (done + (p.stage === "training" && p.batches ? (p.batch || 0) / p.batches : 0)) / p.epochs);
    prog += `<div class="progress-block"><div class="progress-label"><span>Overall · ${done} of ${p.epochs} epochs complete${p.stage_detail ? " · " + esc(p.stage_detail) : ""}</span><span>${(frac * 100).toFixed(1)}%</span></div>
      <div class="progress"><span style="width:${frac * 100}%"></span></div></div>`;
    if (running && p.batches && (p.stage === "training" || p.stage === "validating")) {
      const bf = Math.min(1, (p.batch || 0) / p.batches);
      prog += `<div class="progress-block"><div class="progress-label"><span>Epoch ${p.epoch} · batch ${p.batch} / ${p.batches}${p.stage === "validating" ? " · validating" : ""}</span><span>${(bf * 100).toFixed(0)}%</span></div>
        <div class="progress thin"><span style="width:${bf * 100}%"></span></div></div>`;
    }
  } else if (running) {
    prog = `<p class="muted">${spinner(p.stage_detail || STAGE_LABEL[p.stage] || "Starting worker…")}</p>`;
  }
  $("#jd-progress").innerHTML = prog;
  const losses = p.losses || {};
  $("#jd-losses").innerHTML = running && Object.keys(losses).length
    ? "current " + Object.entries(losses).map(([k, v]) => `<span>${esc(k.replace("train/", ""))}=${v.toFixed(4)}</span>`).join("") : "";

  // Charts + table only when history changes (or the chart is empty/resized)
  if (hist.length !== MON.histLen) {
    MON.histLen = hist.length;
    lineChart($("#ch-loss"), { series: LOSS_SERIES, data: hist });
    lineChart($("#ch-map"), { series: MAP_SERIES, data: hist, zeroBased: true, fmt: (v) => v.toFixed(3) });
    const cols = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"];
    const heads = ["Epoch", "box", "cls", "dfl", "P", "R", "mAP50", "mAP50-95"];
    $("#jd-table .table-wrap").innerHTML = hist.length ? `<table><thead><tr>${heads.map((h) => `<th class="num">${h}</th>`).join("")}</tr></thead><tbody>${
      hist.slice().reverse().map((h) => `<tr>${cols.map((c) => `<td class="num">${h[c] == null ? "–" : c === "epoch" ? h[c] : h[c].toFixed(4)}</td>`).join("")}</tr>`).join("")}</tbody></table>`
      : `<p class="muted" style="padding:8px">No epochs yet.</p>`;
  }

  // Result
  const res = d.result;
  $("#jd-result").innerHTML = d.status === "finished" && res ? `
    <div class="notice good"><b>Training complete.</b> Weights and plots are in <code>${esc(res.model_dir)}</code>.</div>
    <dl class="kv">${Object.entries(res).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : v)}</dd>`).join("")}</dl>` : "";

  // Config
  const cfg = { job_id: d.job_id, source: d.source, dataset_path: d.dataset_path, output_path: d.output_path || "(default models/)",
    created: fmtTime(d.created_at), started: fmtTime(d.started_at), ended: fmtTime(d.ended_at), ...(d.training || {}) };
  $("#jd-config").innerHTML = Object.entries(cfg).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v ?? "–")}</dd>`).join("");
}

async function monSystem() {
  try {
    const s = await api("/system/status");
    const meter = (label, frac, text) => `<div class="meter"><span>${label}</span><div class="progress thin"><span style="width:${Math.min(100, (frac || 0) * 100)}%"></span></div><span>${text}</span></div>`;
    let html = "";
    for (const g of s.gpus) {
      html += `<div title="${esc(g.name)}"><b>GPU ${g.index}</b> <span class="muted">${esc(g.name)}${g.temp_c != null ? ` · ${g.temp_c}°C` : ""}</span></div>`;
      html += meter("util", (g.util || 0) / 100, `${g.util ?? "–"}%`);
      html += meter("memory", g.mem_used_mb / g.mem_total_mb, `${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(0)} GB`);
    }
    if (!s.gpus.length) html += `<div class="muted">No NVIDIA GPU detected.</div>`;
    if (s.memory.total) html += meter("RAM", 1 - s.memory.available / s.memory.total, `${fmtBytes(s.memory.total - s.memory.available)} / ${fmtBytes(s.memory.total)}`);
    html += meter("disk", 1 - s.disk.free / s.disk.total, `${fmtBytes(s.disk.free)} free`);
    $("#sys-status").innerHTML = html;
  } catch { /* ignore */ }
}

Tabs.monitor = {
  enter(params) {
    monRenderList();
    const id = params.get("job");
    if (id) monSelect(id);
    else if (!MON.selected) {
      const pick = MON.jobs.find((j) => j.status === "started") || MON.jobs[0];
      if (pick) go("monitor", { job: pick.job_id });
    }
    monSystem();
  },
};

// Polling loops: fast while watching an active job, slow otherwise.
function scheduleMonitor() {
  clearTimeout(MON.timer);
  const watching = state.tab === "monitor" && !document.hidden;
  const active = MON.detail && ["queued", "started"].includes(MON.detail.status);
  MON.timer = setTimeout(async () => {
    await monRefreshQueue();
    if (state.tab === "monitor") {
      if (!MON.selected && MON.jobs.length) Tabs.monitor.enter(parseHash().params);
      else await monTick();
    }
    scheduleMonitor();
  }, watching ? (active || !MON.detail ? 2000 : 6000) : 10000);
}
setInterval(() => { if (state.tab === "monitor" && !document.hidden) monSystem(); }, 5000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) scheduleMonitor(); });
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { MON.histLen = -1; if (MON.detail) monRenderDetail(); }, 200); });

// ---------------------------------------------------------------------------
// Dataset root
// ---------------------------------------------------------------------------
const ROOT = { settings: null };

function renderRoots() {
  const s = ROOT.settings;
  $("#root-select").innerHTML = s.roots.map((r) =>
    `<option value="${esc(r.path)}" ${r.path === s.dataset_root ? "selected" : ""}>${esc(r.path)}${r.exists ? "" : " (unreachable)"}</option>`).join("") +
    `<option value="__manage">Add / manage folders…</option>`;
  $("#root-list").innerHTML = s.roots.map((r) => `
    <div class="root-row ${r.path === s.dataset_root ? "current" : ""}">
      <span class="p">${esc(r.path)}${r.default ? ' <span class="muted">(project default)</span>' : ""}${r.exists ? "" : ' <span class="pill warn">unreachable</span>'}</span>
      ${r.path === s.dataset_root ? `<span class="pill split">current</span>` : `<button class="btn small" data-use="${esc(r.path)}">Use</button>`}
      ${r.default ? "" : `<button class="btn small danger" data-forget="${esc(r.path)}" title="Remove from this list (files are not touched)">Remove</button>`}
    </div>`).join("");
}

async function switchRoot(path) {
  const st = $("#root-status");
  try {
    ROOT.settings = await api("/settings/root", { method: "PUT", json: { path } });
    renderRoots();
    // Everything selected so far belongs to the previous root.
    DS.selected = null; DS.info = null; PV.ds = null; TR.selectedSources.clear();
    $("#ds-detail").innerHTML = `<div class="empty">Select a dataset on the left, or import one.</div>`;
    await loadDatasets();
    toast("Dataset root: " + ROOT.settings.dataset_root, "ok");
    setStatus(st, "");
    $("#root-dialog").close();
    go(state.tab);
    route();
  } catch (e) {
    setStatus(st, esc(e.message), "err");
    renderRoots();
    if (!$("#root-dialog").open) toast(e.message, "err");
  }
}

$("#root-select").addEventListener("change", (e) => {
  if (e.target.value === "__manage") {
    renderRoots();
    setStatus($("#root-status"), "");
    $("#root-dialog").showModal();
    $("#root-new").focus();
  } else switchRoot(e.target.value);
});
$("#root-close").addEventListener("click", () => $("#root-dialog").close());
$("#root-add").addEventListener("click", () => {
  const v = $("#root-new").value.trim();
  if (!v) { setStatus($("#root-status"), "Enter a folder path.", "err"); return; }
  switchRoot(v).then(() => { if (!$("#root-dialog").open) $("#root-new").value = ""; });
});
$("#root-new").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#root-add").click(); });
$("#root-list").addEventListener("click", async (e) => {
  const use = e.target.closest("[data-use]"), forget = e.target.closest("[data-forget]");
  if (use) switchRoot(use.dataset.use);
  if (forget) {
    try {
      const wasCurrent = forget.dataset.forget === ROOT.settings.dataset_root;
      ROOT.settings = await api(`/settings/root?path=${encodeURIComponent(forget.dataset.forget)}`, { method: "DELETE" });
      renderRoots();
      if (wasCurrent) { DS.selected = null; PV.ds = null; await loadDatasets(); route(); }
    } catch (err) { setStatus($("#root-status"), esc(err.message), "err"); }
  }
});
$("#root-dialog").addEventListener("close", () => { if (ROOT.settings) renderRoots(); });

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async function boot() {
  window.addEventListener("hashchange", route);
  try { ROOT.settings = await api("/settings"); renderRoots(); } catch (e) { toast("Could not load settings: " + e.message, "err"); }
  try { await loadDatasets(); } catch (e) { toast("Could not load datasets: " + e.message, "err"); }
  await monRefreshQueue();
  route();
  scheduleMonitor();
})();
