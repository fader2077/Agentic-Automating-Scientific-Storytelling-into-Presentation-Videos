const settingsForm = document.getElementById("settings-form");
const taskForm = document.getElementById("task-form");
const settingsStatus = document.getElementById("settings-status");
const gpuStatus = document.getElementById("gpu-status");
const uploadStatus = document.getElementById("upload-status");
const taskMeta = document.getElementById("task-meta");
const taskStatusPill = document.getElementById("task-status-pill");
const progressFill = document.getElementById("progress-fill");
const stepsRoot = document.getElementById("steps");
const taskLog = document.getElementById("task-log");
const sampleVideo = document.getElementById("sample-video");
const artifactLinks = document.getElementById("artifact-links");
const testOllamaButton = document.getElementById("test-ollama");
const toolCatalogRoot = document.getElementById("tool-catalog");
const agentCatalogRoot = document.getElementById("agent-catalog");
const ocrSummary = document.getElementById("ocr-summary");
const ocrAssetsRoot = document.getElementById("ocr-assets");
const queueSummary = document.getElementById("queue-summary");
const textModelSelect = document.getElementById("text_model");
const visionModelSelect = document.getElementById("vision_model");
const slideStyleSelect = document.getElementById("preferred_slide_style");
const stylePreviewGrid = document.getElementById("style-preview-grid");
const skillsEditor = document.getElementById("skills-editor");
const skillsEditorTitle = document.getElementById("skills-editor-title");
const skillsEditorText = document.getElementById("skills-editor-text");
const skillsEditorStatus = document.getElementById("skills-editor-status");
const skillsEditorSave = document.getElementById("skills-editor-save");
const skillsEditorClose = document.getElementById("skills-editor-close");

const STEP_KEYS = ["ingest", "planner", "slides", "script", "tts", "cursor", "compose"];

let uploadedPaper = null;
let activeTaskId = null;
let pollHandle = null;
let currentSettings = {};
let slideStyles = [];
let activeSkillsAgent = null;

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function requestText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.text();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formToObject(form) {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function getStepTicksFromForm() {
  const ticks = {};
  STEP_KEYS.forEach((key) => {
    ticks[key] = Number(document.getElementById(`step_ticks_${key}`).value);
  });
  return ticks;
}

function setSelectOptions(select, values, selectedValue) {
  const unique = [...new Set(values.filter(Boolean))];
  if (selectedValue && !unique.includes(selectedValue)) {
    unique.unshift(selectedValue);
  }
  select.innerHTML = "";
  unique.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  if (selectedValue) {
    select.value = selectedValue;
  }
}

function applySettings(settings) {
  currentSettings = settings;
  Object.entries(settings).forEach(([key, value]) => {
    const field = document.getElementById(key);
    if (field && typeof value !== "object") {
      field.value = value;
    }
  });
  STEP_KEYS.forEach((key) => {
    const field = document.getElementById(`step_ticks_${key}`);
    if (field) {
      field.value = settings.step_ticks?.[key] ?? "";
    }
  });
}

function renderSlideStyles(items = []) {
  slideStyles = items;
  setSelectOptions(slideStyleSelect, items.map((item) => item.value), slideStyleSelect.value || items[0]?.value);
  stylePreviewGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "style-template-card";
    card.dataset.value = item.value;
    card.style.setProperty("--accent", item.accent || "#126b9a");
    const preview = item.preview || {};
    card.innerHTML = `
      <div class="slide-preview">
        <div class="preview-bar"></div>
        <div class="preview-title">${escapeHtml(preview.headline || item.title)}</div>
        <div class="preview-body">
          <span></span><span></span><span></span>
        </div>
        <div class="preview-visual">${escapeHtml(preview.visual || "OCR visual")}</div>
      </div>
      <div class="style-template-title">${escapeHtml(item.title)}</div>
      <p>${escapeHtml(preview.notes || item.value)}</p>
    `;
    card.addEventListener("click", () => {
      slideStyleSelect.value = item.value;
      updateStyleSelection();
    });
    stylePreviewGrid.appendChild(card);
  });
  updateStyleSelection();
}

function updateStyleSelection() {
  [...stylePreviewGrid.querySelectorAll(".style-template-card")].forEach((card) => {
    card.classList.toggle("selected", card.dataset.value === slideStyleSelect.value);
  });
}

function renderToolCatalog(items = []) {
  toolCatalogRoot.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "tool-card";
    card.innerHTML = `
      <div class="tool-card-top">
        <h3>${escapeHtml(item.title)}</h3>
        <span class="tool-kind">${escapeHtml(item.kind)}</span>
      </div>
      <div class="tool-key">${escapeHtml(item.key)}</div>
      <div class="tool-entry">${escapeHtml(item.entrypoint || "")}</div>
      <p class="tool-purpose">${escapeHtml(item.purpose)}</p>
      ${item.uses_gpu ? `<div class="tool-gpu">GPU</div>` : ""}
    `;
    toolCatalogRoot.appendChild(card);
  });
}

function renderAgentCatalog(items = []) {
  agentCatalogRoot.innerHTML = "";
  items.forEach((item) => {
    const tools = (item.tools || []).map((tool) => tool.title || tool.key || tool).join(", ");
    const card = document.createElement("article");
    card.className = "agent-card";
    card.innerHTML = `
      <div class="tool-card-top">
        <h3>${escapeHtml(item.title || item.key)}</h3>
        <div class="agent-actions">
          <a class="skills-link" href="${escapeHtml(item.skills_md_url)}" target="_blank">Open</a>
          <button type="button" class="skills-edit-button" data-agent-key="${escapeHtml(item.key)}">Edit skills.md</button>
        </div>
      </div>
      <div class="tool-key">${escapeHtml(item.key)}</div>
      <div class="agent-row"><strong>Skills</strong><span>${escapeHtml((item.skills || []).join(", "))}</span></div>
      <div class="agent-row"><strong>Tools</strong><span>${escapeHtml(tools)}</span></div>
    `;
    agentCatalogRoot.appendChild(card);
  });
}

async function openSkillsEditor(agentKey) {
  activeSkillsAgent = agentKey;
  skillsEditorTitle.textContent = `${agentKey} skills.md`;
  skillsEditorStatus.textContent = "Loading...";
  skillsEditor.classList.remove("hidden");
  skillsEditor.setAttribute("aria-hidden", "false");
  try {
    skillsEditorText.value = await requestText(`/api/agents/${encodeURIComponent(agentKey)}/skills.md`);
    skillsEditorStatus.textContent = "Loaded.";
  } catch (error) {
    skillsEditorStatus.textContent = error.message;
  }
}

function closeSkillsEditor() {
  activeSkillsAgent = null;
  skillsEditor.classList.add("hidden");
  skillsEditor.setAttribute("aria-hidden", "true");
}

async function saveSkillsEditor() {
  if (!activeSkillsAgent) {
    return;
  }
  skillsEditorStatus.textContent = "Saving...";
  try {
    await request(`/api/agents/${encodeURIComponent(activeSkillsAgent)}/skills.md`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: skillsEditorText.value }),
    });
    skillsEditorStatus.textContent = "Saved.";
    await loadAgentCatalog();
  } catch (error) {
    skillsEditorStatus.textContent = error.message;
  }
}

function renderOcrAssets(manifest = {}) {
  const counts = manifest.counts || {};
  const assets = manifest.assets || [];
  const countText = Object.entries(counts).map(([key, value]) => `${key}: ${value}`).join(" | ");
  ocrSummary.textContent = countText || "No OCR visual assets found for this run.";
  ocrAssetsRoot.innerHTML = "";
  assets.slice(0, 24).forEach((asset) => {
    const card = document.createElement("article");
    card.className = "ocr-card";
    const preview = asset.image_url
      ? `<img src="${escapeHtml(asset.image_url)}" alt="${escapeHtml(asset.kind)} from page ${escapeHtml(asset.page)}" loading="lazy" />`
      : `<div class="ocr-text-preview">${escapeHtml(asset.body || asset.caption || "Text-only OCR asset")}</div>`;
    card.innerHTML = `
      ${preview}
      <div class="ocr-card-body">
        <div class="ocr-kind">${escapeHtml(asset.kind)} | page ${escapeHtml(asset.page)}</div>
        <p>${escapeHtml(asset.caption || asset.body || "No caption extracted.")}</p>
      </div>
    `;
    ocrAssetsRoot.appendChild(card);
  });
}

function renderArtifacts(artifacts = {}) {
  artifactLinks.innerHTML = "";
  Object.entries(artifacts).forEach(([key, value]) => {
    const link = document.createElement("a");
    link.href = value;
    link.textContent = key.replaceAll("_", " ");
    link.target = "_blank";
    artifactLinks.appendChild(link);
  });
}

function renderTools(tools = []) {
  if (!tools.length) return "";
  return `
    <div class="tool-run-list">
      ${tools.map((tool) => `
        <div class="tool-run">
          <div class="tool-run-top">
            <div>
              <div class="tool-run-title">${escapeHtml(tool.title)}</div>
              <div class="tool-run-purpose">${escapeHtml(tool.purpose)}</div>
            </div>
            <span class="tool-run-state ${escapeHtml(tool.status)}">${escapeHtml(tool.status)}</span>
          </div>
          <div class="tool-run-count">Count ${escapeHtml(tool.tick_progress || 0)} / ${escapeHtml(tool.tick_total || 0)}</div>
          <div class="tool-run-call">${escapeHtml(tool.recent_call || "Waiting for dispatch.")}</div>
        </div>`).join("")}
    </div>
  `;
}

function renderSteps(steps = []) {
  stepsRoot.innerHTML = "";
  steps.forEach((step) => {
    const card = document.createElement("article");
    card.className = "step-card";
    card.innerHTML = `
      <div class="step-head">
        <h3>${escapeHtml(step.title)}</h3>
        <span class="step-state ${escapeHtml(step.status)}">${escapeHtml(step.status)}</span>
      </div>
      <div class="step-meta">${escapeHtml(step.agent)}</div>
      <p class="step-detail">${escapeHtml(step.detail || "")}</p>
      <div class="step-count">Count ${escapeHtml(step.tick_progress || 0)} / ${escapeHtml(step.tick_total || 0)}</div>
      ${renderTools(step.tools || [])}
    `;
    stepsRoot.appendChild(card);
  });
}

function renderQueue(queue) {
  if (!queue) {
    queueSummary.textContent = "Queue is idle.";
    return;
  }
  const active = queue.active_task_id ? `Active ${queue.active_task_id}` : "No active job";
  const pending = queue.queued_count ? `Queued ${queue.queued_count}` : "No queued jobs";
  queueSummary.textContent = `${active} | ${pending} | Worker ${queue.worker_id}`;
}

function renderTask(task) {
  activeTaskId = task.id;
  const counts = task.progress_counts || { completed: 0, total: 0 };
  const currentStep = (task.current_step ?? -1) + 1;
  const totalSteps = (task.steps || []).length;
  const queuePos = task.job?.queue_position ? `Queue ${task.job.queue_position}` : task.job?.state || "queued";
  taskMeta.textContent = `${task.upload_name} | ${task.desired_minutes} min target | Stage ${Math.max(currentStep, 0)} / ${totalSteps} | Count ${counts.completed} / ${counts.total} | ${queuePos}`;
  taskStatusPill.textContent = task.status;
  progressFill.style.width = `${Math.round((task.progress || 0) * 100)}%`;
  renderSteps(task.steps || []);
  taskLog.textContent = (task.log || []).join("\n");
  renderArtifacts(task.artifacts || {});
  if (task.status === "completed" && task.sample_video_url) {
    sampleVideo.src = task.sample_video_url;
    sampleVideo.load();
  }
}

async function loadOcrAssets(taskId) {
  if (!taskId) return;
  try {
    renderOcrAssets(await request(`/api/tasks/${taskId}/ocr-assets`));
  } catch (error) {
    ocrSummary.textContent = error.message;
  }
}

async function loadQueue() {
  renderQueue(await request("/api/queue"));
}

async function loadHealth() {
  const health = await request("/api/health");
  const gpu = health.gpu || {};
  gpuStatus.textContent = gpu.cuda_available
    ? `GPU ready: ${gpu.device_name} | torch ${gpu.torch} | CUDA ${gpu.cuda_build}`
    : `GPU unavailable for Python runtime: ${gpu.error || "CUDA not detected"}`;
}

async function pollTask(taskId) {
  if (pollHandle) clearInterval(pollHandle);
  const refresh = async () => {
    try {
      const [task, queue] = await Promise.all([request(`/api/tasks/${taskId}`), request("/api/queue")]);
      renderTask(task);
      renderQueue(queue);
      await loadOcrAssets(taskId);
      if (task.status === "completed" || task.status === "failed") {
        clearInterval(pollHandle);
        pollHandle = null;
      }
    } catch (error) {
      taskLog.textContent = `${taskLog.textContent}\nPolling error: ${error.message}`;
      clearInterval(pollHandle);
      pollHandle = null;
    }
  };
  await refresh();
  pollHandle = setInterval(refresh, 1000);
}

async function loadSettings() {
  applySettings(await request("/api/settings"));
}

async function loadModelOptions() {
  try {
    const payload = await request("/api/ollama/models");
    const models = payload.models || [];
    setSelectOptions(textModelSelect, models, currentSettings.text_model);
    setSelectOptions(visionModelSelect, models, currentSettings.vision_model);
  } catch (error) {
    setSelectOptions(textModelSelect, [currentSettings.text_model], currentSettings.text_model);
    setSelectOptions(visionModelSelect, [currentSettings.vision_model], currentSettings.vision_model);
    settingsStatus.textContent = error.message;
  }
}

async function loadSlideStyles() {
  renderSlideStyles(await request("/api/slide-styles"));
}

async function loadToolCatalog() {
  renderToolCatalog(await request("/api/tool-catalog"));
}

async function loadAgentCatalog() {
  renderAgentCatalog(await request("/api/agent-catalog"));
}

async function uploadPaper(file) {
  const data = new FormData();
  data.append("file", file);
  const payload = await request("/api/upload", { method: "POST", body: data });
  uploadedPaper = payload;
  uploadStatus.textContent = `Uploaded ${payload.file_name} (${payload.size_bytes} bytes).`;
}

document.getElementById("paper_file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  uploadStatus.textContent = `Uploading ${file.name}...`;
  try {
    await uploadPaper(file);
  } catch (error) {
    uploadStatus.textContent = error.message;
  }
});

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  settingsStatus.textContent = "Saving settings...";
  try {
    const raw = formToObject(settingsForm);
    raw.temperature = Number(raw.temperature);
    raw.top_p = Number(raw.top_p);
    raw.max_tokens = Number(raw.max_tokens);
    raw.tick_seconds = Number(currentSettings.tick_seconds || 1);
    raw.step_ticks = getStepTicksFromForm();
    const saved = await request("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(raw),
    });
    applySettings(saved);
    await loadModelOptions();
    settingsStatus.textContent = "Settings saved.";
  } catch (error) {
    settingsStatus.textContent = error.message;
  }
});

testOllamaButton.addEventListener("click", async () => {
  settingsStatus.textContent = "Checking local Ollama...";
  try {
    const raw = formToObject(settingsForm);
    const result = await request("/api/settings/test-ollama", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ollama_url: raw.ollama_url, text_model: raw.text_model, vision_model: raw.vision_model }),
    });
    await loadModelOptions();
    const suffix = result.reachable ? " Endpoint responded." : " Endpoint is configured.";
    settingsStatus.textContent = `${result.message}${suffix}`;
  } catch (error) {
    settingsStatus.textContent = error.message;
  }
});

taskForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!uploadedPaper) {
    uploadStatus.textContent = "Upload a paper before starting the pipeline.";
    return;
  }
  const raw = formToObject(taskForm);
  taskMeta.textContent = "Creating task...";
  try {
    const task = await request("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        upload_id: uploadedPaper.id,
        goal_prompt: raw.goal_prompt,
        desired_minutes: Number(raw.desired_minutes),
        preferred_slide_style: raw.preferred_slide_style,
      }),
    });
    renderTask(task);
    await loadQueue();
    await pollTask(task.id);
  } catch (error) {
    taskMeta.textContent = error.message;
  }
});

slideStyleSelect.addEventListener("change", updateStyleSelection);
agentCatalogRoot.addEventListener("click", (event) => {
  const button = event.target.closest(".skills-edit-button");
  if (button) {
    openSkillsEditor(button.dataset.agentKey);
  }
});
skillsEditorClose.addEventListener("click", closeSkillsEditor);
skillsEditorSave.addEventListener("click", saveSkillsEditor);
skillsEditor.addEventListener("click", (event) => {
  if (event.target === skillsEditor) {
    closeSkillsEditor();
  }
});

async function initialize() {
  await loadSettings();
  await Promise.all([loadModelOptions(), loadSlideStyles(), loadToolCatalog(), loadAgentCatalog(), loadQueue(), loadHealth()]);
}

initialize().catch((error) => {
  settingsStatus.textContent = error.message;
});
