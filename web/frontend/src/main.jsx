import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, Database, FileUp, GitBranch, History, Play, RotateCcw, UserRound } from "lucide-react";
import "./styles.css";

const emptySpec = {
  course_title: "AIMOOC: Agentic AI Course Builder",
  audience: "graduate students and engineers",
  learning_objectives: "Understand the core sources\nBuild a module-level syllabus\nGenerate lessons, scripts, quizzes, and avatar manifests",
  requirements: "Use uploaded sources as primary material. Keep lessons practical and concise.",
  total_minutes: 30,
  module_count: 2,
  lessons_per_module: 2,
  preferred_style: "teaching_walkthrough",
  language: "zh-TW",
  difficulty: "intermediate",
  include_quizzes: true,
  include_avatar: true,
  avatar_mode: "presenter_card",
  agentic_framework: "langgraph",
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function Section({ icon: Icon, title, children }) {
  return (
    <section className="rounded-lg border border-line bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center gap-2">
        <Icon size={18} className="text-blue-700" />
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function App() {
  const [uploads, setUploads] = useState([]);
  const [projects, setProjects] = useState([]);
  const [frameworks, setFrameworks] = useState([]);
  const [selected, setSelected] = useState([]);
  const [spec, setSpec] = useState(emptySpec);
  const [activeProject, setActiveProject] = useState(null);
  const [feedback, setFeedback] = useState("請把第一節改得更直覺，補一個例子。");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    const [uploadData, projectData, frameworkData] = await Promise.all([
      api("/api/uploads"),
      api("/api/aimooc/projects"),
      api("/api/agent-frameworks"),
    ]);
    setUploads(uploadData);
    setProjects(projectData);
    setFrameworks(frameworkData);
  }

  useEffect(() => {
    refresh().catch((err) => setError(String(err)));
  }, []);

  const selectedRows = useMemo(
    () =>
      selected.map((id, index) => ({
        source_id: id,
        role: index === 0 ? "primary" : "reference",
        priority: index === 0 ? 1 : 3,
      })),
    [selected],
  );

  async function uploadFiles(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    setBusy(true);
    setError("");
    try {
      await api("/api/uploads/batch", { method: "POST", body: form });
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function createProject() {
    setBusy(true);
    setError("");
    try {
      const payload = {
        ...spec,
        learning_objectives: spec.learning_objectives.split("\n").map((line) => line.trim()).filter(Boolean),
        sources: selectedRows,
        total_minutes: Number(spec.total_minutes),
        module_count: Number(spec.module_count),
        lessons_per_module: Number(spec.lessons_per_module),
      };
      const project = await api("/api/aimooc/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setActiveProject(project);
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function reviseProject() {
    if (!activeProject) return;
    setBusy(true);
    setError("");
    try {
      await api(`/api/aimooc/projects/${activeProject.project_id}/revise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_version: "v001_initial",
          feedback: [
            {
              target_type: "lesson",
              target_id: "module_01_lesson_01",
              severity: "normal",
              instruction: feedback,
              preferred_action: "revise",
            },
          ],
        }),
      });
      setActiveProject(await api(`/api/aimooc/projects/${activeProject.project_id}`));
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen">
      <header className="border-b border-line bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">AIMOOC</p>
            <h1 className="text-2xl font-semibold">Agentic AI Course Builder</h1>
          </div>
          <a href="/" className="rounded-md border border-line px-3 py-2 text-sm hover:bg-panel">Single PDF mode</a>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-5 py-5 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Section icon={FileUp} title="Sources">
            <input multiple type="file" onChange={uploadFiles} className="w-full rounded-md border border-line bg-panel p-2 text-sm" />
            <div className="mt-3 max-h-72 space-y-2 overflow-auto">
              {uploads.map((upload) => (
                <label key={upload.id} className="flex items-start gap-2 rounded-md border border-line p-2 text-sm">
                  <input
                    type="checkbox"
                    checked={selected.includes(upload.id)}
                    onChange={(event) =>
                      setSelected((prev) => event.target.checked ? [...prev, upload.id] : prev.filter((id) => id !== upload.id))
                    }
                  />
                  <span>
                    <span className="block font-medium">{upload.file_name}</span>
                    <span className="text-xs text-slate-500">{Math.round((upload.size_bytes || 0) / 1024)} KB</span>
                  </span>
                </label>
              ))}
            </div>
          </Section>

          <Section icon={GitBranch} title="Mode and Framework">
            <label className="text-xs font-semibold">Function</label>
            <select className="mb-3 mt-1 w-full rounded-md border border-line p-2" value="aimooc" disabled>
              <option value="aimooc">Multi-source AIMOOC</option>
            </select>
            <label className="text-xs font-semibold">Agentic framework</label>
            <select
              className="mt-1 w-full rounded-md border border-line p-2"
              value={spec.agentic_framework}
              onChange={(event) => setSpec({ ...spec, agentic_framework: event.target.value })}
            >
              {frameworks.map((item) => (
                <option key={item.key} value={item.key}>{item.title} {item.installed ? "" : "(adapter)"}</option>
              ))}
            </select>
          </Section>
        </div>

        <div className="space-y-4">
          <Section icon={BookOpen} title="Course Spec">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <input className="rounded-md border border-line p-2" value={spec.course_title} onChange={(e) => setSpec({ ...spec, course_title: e.target.value })} />
              <input className="rounded-md border border-line p-2" value={spec.audience} onChange={(e) => setSpec({ ...spec, audience: e.target.value })} />
              <textarea className="min-h-24 rounded-md border border-line p-2 lg:col-span-2" value={spec.learning_objectives} onChange={(e) => setSpec({ ...spec, learning_objectives: e.target.value })} />
              <textarea className="min-h-20 rounded-md border border-line p-2 lg:col-span-2" value={spec.requirements} onChange={(e) => setSpec({ ...spec, requirements: e.target.value })} />
              <input type="number" className="rounded-md border border-line p-2" value={spec.total_minutes} onChange={(e) => setSpec({ ...spec, total_minutes: e.target.value })} />
              <input type="number" className="rounded-md border border-line p-2" value={spec.module_count} onChange={(e) => setSpec({ ...spec, module_count: e.target.value })} />
              <input type="number" className="rounded-md border border-line p-2" value={spec.lessons_per_module} onChange={(e) => setSpec({ ...spec, lessons_per_module: e.target.value })} />
              <select className="rounded-md border border-line p-2" value={spec.avatar_mode} onChange={(e) => setSpec({ ...spec, avatar_mode: e.target.value })}>
                <option value="none">No avatar</option>
                <option value="presenter_card">Presenter card</option>
              </select>
            </div>
            <button disabled={busy || selected.length === 0} onClick={createProject} className="mt-4 inline-flex items-center gap-2 rounded-md bg-blue-700 px-4 py-2 text-white disabled:opacity-50">
              <Play size={16} /> Build course package
            </button>
          </Section>

          {error && <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

          <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Section icon={Database} title="Projects">
              <div className="space-y-2">
                {projects.map((project) => (
                  <button key={project.project_id} onClick={() => api(`/api/aimooc/projects/${project.project_id}`).then(setActiveProject)} className="w-full rounded-md border border-line p-3 text-left hover:bg-panel">
                    <span className="block font-medium">{project.title}</span>
                    <span className="text-xs text-slate-500">{project.framework} | {project.source_count} sources | {project.status}</span>
                  </button>
                ))}
              </div>
            </Section>

            <Section icon={History} title="Feedback and Versions">
              <textarea className="min-h-24 w-full rounded-md border border-line p-2" value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <button disabled={busy || !activeProject} onClick={reviseProject} className="mt-3 inline-flex items-center gap-2 rounded-md border border-line px-3 py-2 disabled:opacity-50">
                <RotateCcw size={16} /> Create revision
              </button>
            </Section>
          </section>

          {activeProject && (
            <Section icon={UserRound} title="Active Package">
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
                <a className="rounded-md border border-line p-3 hover:bg-panel" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/course_plan.json`}>course_plan.json</a>
                <a className="rounded-md border border-line p-3 hover:bg-panel" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/source_manifest.json`}>source_manifest.json</a>
                <a className="rounded-md border border-line p-3 hover:bg-panel" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/course_package_manifest.json`}>course_package_manifest.json</a>
              </div>
              <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-100">{JSON.stringify(activeProject.course_plan, null, 2)}</pre>
            </Section>
          )}
        </div>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
