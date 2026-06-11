const historyQueue = document.getElementById("history-queue");
const historyList = document.getElementById("history-list");

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function formatStatus(task) {
  const jobState = task.job?.state || task.status;
  const queuePos = task.job?.queue_position ? `Queue ${task.job.queue_position}` : jobState;
  return `${jobState} | Count ${task.progress_counts.completed} / ${task.progress_counts.total} | ${queuePos}`;
}

function renderHistory(queue, tasks) {
  const active = queue.active_task_id ? `Active ${queue.active_task_id}` : "No active job";
  const pending = queue.queued_count ? `Queued ${queue.queued_count}` : "No queued jobs";
  historyQueue.textContent = `${active} | ${pending} | Worker ${queue.worker_id}`;

  historyList.innerHTML = "";
  tasks.forEach((task) => {
    const card = document.createElement("article");
    card.className = "history-card";
    card.innerHTML = `
      <div class="history-card-top">
        <div>
          <h3>${task.upload_name}</h3>
          <div class="step-meta">${task.goal_prompt}</div>
        </div>
        <span class="step-state ${task.status}">${task.status}</span>
      </div>
      <div class="history-card-grid">
        <div class="status-card">${formatStatus(task)}</div>
        <div class="status-card">Timeline ${task.timeline_length}</div>
        <div class="status-card">Task ${task.id}</div>
      </div>
      <div class="actions">
        <a class="button-link" href="/replay/${task.id}">Open Replay</a>
      </div>
    `;
    historyList.appendChild(card);
  });
}

async function loadHistory() {
  const [queue, tasks] = await Promise.all([request("/api/queue"), request("/api/tasks")]);
  renderHistory(queue, tasks);
}

loadHistory().catch((error) => {
  historyQueue.textContent = error.message;
});
setInterval(() => {
  loadHistory().catch(() => {});
}, 2000);
