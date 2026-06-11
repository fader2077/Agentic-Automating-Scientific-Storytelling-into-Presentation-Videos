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

const STEP_KEYS = ["ingest", "planner", "slides", "script", "tts", "cursor", "compose"];

let uploadedPaper = null;
let activeTaskId = null;
let pollHandle = null;
let currentSettings = {};

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
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

function applySettings(settings) {
  currentSettings = settings;
  Object.entries(settings).forEach(([key, value]) => {
    const field = document.getElementById(key);
    if (field && typeof value !== "object") {
      field.value = value;
    }
  });

  const stepTicks = settings.step_ticks || {};
  STEP_KEYS.forEach((key) => {
    const field = document.getElementById(`step_ticks_${key}`);
    if (field) {
      field.value = stepTicks[key] ?? "";
    }
  });
}

function renderToolCatalog(items = []) {
  toolCatalogRoot.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "tool-card";
    card.innerHTML = `
      <div class="tool-card-top">
        <h3>${item.title}</h3>
        <span class="tool-kind">${item.kind}</span>
      </div>
      <div class="tool-key">${item.key}</div>
      <div class="tool-entry">${item.entrypoint || ""}</div>
      <p class="tool-purpose">${item.purpose}</p>
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
        <h3>${item.title || item.key}</h3>
        <a class="skills-link" href="${item.skills_md_url}" target="_blank">skills.md</a>
      </div>
      <div class="tool-key">${item.key}</div>
      <div class="agent-row"><strong>Skills</strong><span>${(item.skills || []).join(", ")}</span></div>
      <div class="agent-row"><strong>Tools</strong><span>${tools}</span></div>
    `;
    agentCatalogRoot.appendChild(card);
  });
}

function renderOcrAssets(manifest = {}) {
  const counts = manifest.counts || {};
  const assets = manifest.assets || [];
  const countText = Object.entries(counts)
    .map(([key, value]) => `${key}: ${value}`)
    .join(" | ");
  ocrSummary.textContent = countText || "No OCR visual assets found for this run.";
  ocrAssetsRoot.innerHTML = "";
  assets.slice(0, 24).forEach((asset) => {
    const card = document.createElement("article");
    card.className = "ocr-card";
    const preview = asset.image_url
      ? `<img src="${asset.image_url}" alt="${asset.kind} from page ${asset.page}" loading="lazy" />`
      : `<div class="ocr-text-preview">${asset.body || asset.caption || "Text-only OCR asset"}</div>`;
    card.innerHTML = `
      ${preview}
      <div class="ocr-card-body">
        <div class="ocr-kind">${asset.kind} | page ${asset.page}</div>
        <p>${asset.caption || asset.body || "No caption extracted."}</p>
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
  if (!tools.length) {
    return "";
  }

  return `
    <div class="tool-run-list">
      ${tools
        .map(
          (tool) => `
          <div class="tool-run">
            <div class="tool-run-top">
              <div>
                <div class="tool-run-title">${tool.title}</div>
                <div class="tool-run-purpose">${tool.purpose}</div>
              </div>
              <span class="tool-run-state ${tool.status}">${tool.status}</span>
            </div>
            <div class="tool-run-count">Count ${tool.tick_progress || 0} / ${tool.tick_total || 0}</div>
            <div class="tool-run-call">${tool.recent_call || "Waiting for dispatch."}</div>
          </div>
        `,
        )
        .join("")}
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
        <h3>${step.title}</h3>
        <span class="step-state ${step.status}">${step.status}</span>
      </div>
      <div class="step-meta">${step.agent}</div>
      <p class="step-detail">${step.detail || ""}</p>
      <div class="step-count">Count ${step.tick_progress || 0} / ${step.tick_total || 0}</div>
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
  const waiting = queue.queued_count || 0;
  const active = queue.active_task_id ? `Active ${queue.active_task_id}` : "No active job";
  const pending = waiting ? `Queued ${waiting}` : "No queued jobs";
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
  if (!taskId) {
    return;
  }
  try {
    const manifest = await request(`/api/tasks/${taskId}/ocr-assets`);
    renderOcrAssets(manifest);
  } catch (error) {
    ocrSummary.textContent = error.message;
  }
}

async function loadQueue() {
  const queue = await request("/api/queue");
  renderQueue(queue);
}

async function loadHealth() {
  const health = await request("/api/health");
  const gpu = health.gpu || {};
  gpuStatus.textContent = gpu.cuda_available
    ? `GPU ready: ${gpu.device_name} | torch ${gpu.torch} | CUDA ${gpu.cuda_build}`
    : `GPU unavailable for Python runtime: ${gpu.error || "CUDA not detected"}`;
}

async function pollTask(taskId) {
  if (pollHandle) {
    clearInterval(pollHandle);
  }

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
  const settings = await request("/api/settings");
  applySettings(settings);
}

async function loadToolCatalog() {
  const tools = await request("/api/tool-catalog");
  renderToolCatalog(tools);
}

async function loadAgentCatalog() {
  const agents = await request("/api/agent-catalog");
  renderAgentCatalog(agents);
}

async function uploadPaper(file) {
  const data = new FormData();
  data.append("file", file);
  const payload = await request("/api/upload", {
    method: "POST",
    body: data,
  });
  uploadedPaper = payload;
  uploadStatus.textContent = `Uploaded ${payload.file_name} (${payload.size_bytes} bytes).`;
}

document.getElementById("paper_file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
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
      body: JSON.stringify({
        ollama_url: raw.ollama_url,
        text_model: raw.text_model,
        vision_model: raw.vision_model,
      }),
    });
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

Promise.all([loadSettings(), loadToolCatalog(), loadAgentCatalog(), loadQueue(), loadHealth()]).catch((error) => {
  settingsStatus.textContent = error.message;
});
