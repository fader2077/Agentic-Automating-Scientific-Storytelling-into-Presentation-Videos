const replayTitle = document.getElementById("replay-title");
const replayPosition = document.getElementById("replay-position");
const replaySummary = document.getElementById("replay-summary");
const replaySteps = document.getElementById("replay-steps");
const snapshotLabel = document.getElementById("snapshot-label");
const snapshotMeta = document.getElementById("snapshot-meta");
const snapshotCalls = document.getElementById("snapshot-calls");

const firstButton = document.getElementById("replay-first");
const prevButton = document.getElementById("replay-prev");
const nextButton = document.getElementById("replay-next");
const lastButton = document.getElementById("replay-last");

let timeline = [];
let pointer = 0;
let replayTaskId = window.location.pathname.split("/").filter(Boolean).pop();

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderTools(tools = []) {
  return `
    <div class="tool-run-list">
      ${tools
        .map(
          (tool) => `
            <div class="tool-run">
              <div class="tool-run-top">
                <div class="tool-run-title">${tool.title}</div>
                <span class="tool-run-state ${tool.status}">${tool.status}</span>
              </div>
              <div class="tool-run-count">Count ${tool.tick_progress} / ${tool.tick_total}</div>
              <div class="tool-run-call">${tool.recent_call || "No call recorded at this snapshot."}</div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderSnapshot(index) {
  const snapshot = timeline[index];
  if (!snapshot) {
    return;
  }

  replayPosition.textContent = `${index + 1} / ${timeline.length}`;
  replaySummary.textContent = `Task ${replayTaskId} | Status ${snapshot.status} | Count ${snapshot.progress_counts.completed} / ${snapshot.progress_counts.total}`;
  snapshotLabel.textContent = snapshot.label;
  snapshotMeta.textContent = `Step ${Math.max((snapshot.current_step ?? -1) + 1, 0)} | Progress ${Math.round((snapshot.progress || 0) * 100)}%`;

  replaySteps.innerHTML = "";
  const recentCalls = [];
  snapshot.steps.forEach((step) => {
    const card = document.createElement("article");
    card.className = "step-card";
    card.innerHTML = `
      <div class="step-head">
        <h3>${step.title}</h3>
        <span class="step-state ${step.status}">${step.status}</span>
      </div>
      <div class="step-count">Count ${step.tick_progress} / ${step.tick_total}</div>
      ${renderTools(step.tools || [])}
    `;
    replaySteps.appendChild(card);

    (step.tools || []).forEach((tool) => {
      if (tool.recent_call) {
        recentCalls.push(`${tool.title}: ${tool.recent_call}`);
      }
    });
  });

  snapshotCalls.textContent = recentCalls.length ? recentCalls.join("\n") : "No tool call recorded at this snapshot.";
}

function clampPointer(nextValue) {
  if (!timeline.length) {
    pointer = 0;
    return;
  }
  pointer = Math.max(0, Math.min(nextValue, timeline.length - 1));
  renderSnapshot(pointer);
}

async function loadReplay() {
  const payload = await request(`/api/tasks/${replayTaskId}/timeline`);
  timeline = payload.timeline || [];
  replayTitle.textContent = payload.upload_name || replayTaskId;
  clampPointer(timeline.length ? timeline.length - 1 : 0);
}

firstButton.addEventListener("click", () => clampPointer(0));
prevButton.addEventListener("click", () => clampPointer(pointer - 1));
nextButton.addEventListener("click", () => clampPointer(pointer + 1));
lastButton.addEventListener("click", () => clampPointer(timeline.length - 1));

loadReplay().catch((error) => {
  replaySummary.textContent = error.message;
});
