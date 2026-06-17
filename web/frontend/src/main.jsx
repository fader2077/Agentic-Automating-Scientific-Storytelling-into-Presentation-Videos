import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  Boxes,
  Database,
  FileUp,
  Film,
  GitBranch,
  History,
  Layers3,
  Play,
  RotateCcw,
  Sparkles,
  UserRound,
} from "lucide-react";
import "./styles.css";

const emptySpec = {
  course_title: "AIMOOC: Agentic AI Course Builder",
  audience: "graduate students and engineers",
  learning_objectives:
    "Understand the uploaded sources\nBuild a module-level syllabus\nGenerate lessons, scripts, quizzes, and avatar videos",
  requirements: "Use uploaded sources as primary material. Keep lessons practical and concise.",
  total_minutes: 30,
  module_count: 2,
  lessons_per_module: 2,
  target_slide_count: 22,
  preferred_style: "teaching_walkthrough",
  language: "zh-TW",
  difficulty: "intermediate",
  include_quizzes: true,
  include_avatar: true,
  avatar_mode: "presenter_card",
  agentic_framework: "langgraph",
  render_videos: false,
  lesson_video_task_id: "",
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    try {
      const payload = JSON.parse(text);
      const detail = payload.detail;
      if (Array.isArray(detail)) {
        throw new Error(
          detail
            .map((item) => {
              const loc = Array.isArray(item?.loc) ? item.loc.join(".") : "";
              return loc ? `${loc}: ${item?.msg || JSON.stringify(item)}` : item?.msg || JSON.stringify(item);
            })
            .join("\n"),
        );
      }
      if (detail && typeof detail === "object") throw new Error(JSON.stringify(detail, null, 2));
      throw new Error(detail || text);
    } catch (err) {
      if (err instanceof Error && err.message) throw err;
      throw new Error(text);
    }
  }
  return response.json();
}

function Card({ icon: Icon, title, eyebrow, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          {eyebrow && <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-sky-700">{eyebrow}</p>}
          <h2 className="mt-1 text-base font-semibold text-slate-950">{title}</h2>
        </div>
        {Icon && (
          <div className="rounded-lg bg-sky-50 p-2 text-sky-700">
            <Icon size={18} />
          </div>
        )}
      </div>
      {children}
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-semibold text-slate-600">{label}</span>
      {children}
    </label>
  );
}

function App() {
  const [uploads, setUploads] = useState([]);
  const [projects, setProjects] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [frameworks, setFrameworks] = useState([]);
  const [selected, setSelected] = useState([]);
  const [spec, setSpec] = useState(emptySpec);
  const [activeProject, setActiveProject] = useState(null);
  const [feedback, setFeedback] = useState("Make the first lesson more intuitive and add one concrete example.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    const [uploadData, projectData, taskData, frameworkData] = await Promise.all([
      api("/api/uploads"),
      api("/api/aimooc/projects"),
      api("/api/tasks"),
      api("/api/agent-frameworks"),
    ]);
    setUploads(uploadData);
    setProjects(projectData);
    setTasks(taskData.filter((task) => task.status === "completed" && task.artifact_paths?.video));
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

  const courseStats = useMemo(() => {
    const lessons = Number(spec.module_count || 0) * Number(spec.lessons_per_module || 0);
    return [
      { label: "Sources", value: selected.length },
      { label: "Lessons", value: lessons },
      { label: "Minutes", value: spec.total_minutes },
      { label: "Framework", value: spec.agentic_framework.replace("_adapter", "") },
    ];
  }, [selected.length, spec]);

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
        learning_objectives: spec.learning_objectives
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
        sources: selectedRows,
        total_minutes: Number(spec.total_minutes),
        module_count: Number(spec.module_count),
        lessons_per_module: Number(spec.lessons_per_module),
        target_slide_count: Number(spec.target_slide_count || 22),
        render_videos: Boolean(spec.render_videos),
        lesson_video_task_id: spec.render_videos && spec.lesson_video_task_id ? spec.lesson_video_task_id : null,
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

  const lessons = activeProject?.package_manifest?.lessons || [];
  const courseVideoArtifacts = activeProject?.course_video_artifacts || {};
  const courseVideo = activeProject?.package_manifest?.course_video || null;

  return (
    <main className="min-h-screen bg-[#eef3f9] text-slate-900">
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-sky-700">AIMOOC Control Room</p>
            <h1 className="text-2xl font-semibold">Agentic AI Course Builder</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <a href="/" className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold hover:bg-slate-50">
              Single PDF video
            </a>
            <a href="/aimooc" className="rounded-lg bg-slate-950 px-3 py-2 text-sm font-semibold text-white">
              Multi-source course
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-5 px-5 py-5 xl:grid-cols-[330px_1fr]">
        <aside className="space-y-5">
          <Card icon={Layers3} title="Workspace Mode" eyebrow="Function">
            <div className="grid gap-2">
              <a className="rounded-lg border border-slate-200 p-3 text-sm font-semibold hover:bg-slate-50" href="/">
                Single uploaded PDF to video
              </a>
              <div className="rounded-lg border border-sky-200 bg-sky-50 p-3 text-sm font-semibold text-sky-800">
                Multi-source AIMOOC package
              </div>
            </div>
          </Card>

          <Card icon={FileUp} title="Sources" eyebrow="Upload">
            <input multiple type="file" onChange={uploadFiles} className="w-full rounded-lg border border-slate-200 bg-slate-50 p-2 text-sm" />
            <div className="mt-3 max-h-72 space-y-2 overflow-auto">
              {uploads.map((upload) => (
                <label key={upload.id} className="flex items-start gap-2 rounded-lg border border-slate-200 bg-white p-2 text-sm hover:bg-slate-50">
                  <input
                    type="checkbox"
                    checked={selected.includes(upload.id)}
                    onChange={(event) =>
                      setSelected((prev) => (event.target.checked ? [...prev, upload.id] : prev.filter((id) => id !== upload.id)))
                    }
                  />
                  <span>
                    <span className="block font-medium">{upload.file_name}</span>
                    <span className="text-xs text-slate-500">{Math.round((upload.size_bytes || 0) / 1024)} KB</span>
                  </span>
                </label>
              ))}
            </div>
          </Card>

          <Card icon={GitBranch} title="Agentic Runtime" eyebrow="Framework">
            <Field label="Agentic framework">
              <select
                className="w-full rounded-lg border border-slate-200 p-2"
                value={spec.agentic_framework}
                onChange={(event) => setSpec({ ...spec, agentic_framework: event.target.value })}
              >
                {frameworks.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.title} {item.installed ? "" : "(adapter)"}
                  </option>
                ))}
              </select>
            </Field>
            <p className="mt-2 text-xs text-slate-500">
              LangGraph remains native. Hermes and OpenClaw are selectable adapter traces for course agents.
            </p>
          </Card>
        </aside>

        <section className="space-y-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {courseStats.map((stat) => (
              <div key={stat.label} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{stat.label}</p>
                <p className="mt-2 truncate text-xl font-semibold">{stat.value}</p>
              </div>
            ))}
          </div>

          <Card icon={BookOpen} title="Course Specification" eyebrow="Planner input">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <Field label="Course title">
                <input className="w-full rounded-lg border border-slate-200 p-2" value={spec.course_title} onChange={(e) => setSpec({ ...spec, course_title: e.target.value })} />
              </Field>
              <Field label="Audience">
                <input className="w-full rounded-lg border border-slate-200 p-2" value={spec.audience} onChange={(e) => setSpec({ ...spec, audience: e.target.value })} />
              </Field>
              <Field label="Learning objectives">
                <textarea className="min-h-28 w-full rounded-lg border border-slate-200 p-2" value={spec.learning_objectives} onChange={(e) => setSpec({ ...spec, learning_objectives: e.target.value })} />
              </Field>
              <Field label="Requirements">
                <textarea className="min-h-28 w-full rounded-lg border border-slate-200 p-2" value={spec.requirements} onChange={(e) => setSpec({ ...spec, requirements: e.target.value })} />
              </Field>
              <Field label="Total minutes">
                <input type="number" className="w-full rounded-lg border border-slate-200 p-2" value={spec.total_minutes} onChange={(e) => setSpec({ ...spec, total_minutes: e.target.value })} />
              </Field>
              <Field label="Modules">
                <input type="number" className="w-full rounded-lg border border-slate-200 p-2" value={spec.module_count} onChange={(e) => setSpec({ ...spec, module_count: e.target.value })} />
              </Field>
              <Field label="Lessons per module">
                <input type="number" className="w-full rounded-lg border border-slate-200 p-2" value={spec.lessons_per_module} onChange={(e) => setSpec({ ...spec, lessons_per_module: e.target.value })} />
              </Field>
              <Field label="Target slides">
                <input type="number" className="w-full rounded-lg border border-slate-200 p-2" value={spec.target_slide_count} onChange={(e) => setSpec({ ...spec, target_slide_count: e.target.value })} />
              </Field>
              <Field label="Avatar presenter">
                <select className="w-full rounded-lg border border-slate-200 p-2" value={spec.avatar_mode} onChange={(e) => setSpec({ ...spec, avatar_mode: e.target.value, include_avatar: e.target.value !== "none" })}>
                  <option value="presenter_card">Presenter card</option>
                  <option value="talking_head">Talking-head hook</option>
                  <option value="none">No avatar</option>
                </select>
              </Field>
            </div>
            <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <label className="flex items-center gap-2 text-sm font-semibold">
                <input type="checkbox" checked={spec.render_videos} onChange={(e) => setSpec({ ...spec, render_videos: e.target.checked })} />
                Render video media for lessons
              </label>
              <p className="mt-2 text-xs text-slate-500">
                Leave the task selector empty to merge selected PDFs and run the full OCR-to-video pipeline. Choose a completed task only when reusing an existing single-PDF video.
              </p>
              <Field label="Video source mode">
                <select
                  className="mt-2 w-full rounded-lg border border-slate-200 p-2"
                  value={spec.lesson_video_task_id}
                  onChange={(e) => setSpec({ ...spec, lesson_video_task_id: e.target.value })}
                  disabled={!spec.render_videos}
                >
                  <option value="">Run multi-source PDF video generation</option>
                  {tasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.upload_name} | {task.id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <button disabled={busy || selected.length === 0} onClick={createProject} className="mt-4 inline-flex items-center gap-2 rounded-lg bg-slate-950 px-4 py-2 text-white disabled:opacity-50">
              <Play size={16} /> Build course package
            </button>
          </Card>

          {error && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

          <section className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <Card icon={Database} title="Projects" eyebrow="History">
              <div className="space-y-2">
                {projects.map((project) => (
                  <button key={project.project_id} onClick={() => api(`/api/aimooc/projects/${project.project_id}`).then(setActiveProject)} className="w-full rounded-lg border border-slate-200 p-3 text-left hover:bg-slate-50">
                    <span className="block font-medium">{project.title}</span>
                    <span className="text-xs text-slate-500">
                      {project.framework} | {project.source_count} sources | {project.status}
                    </span>
                  </button>
                ))}
              </div>
            </Card>

            <Card icon={History} title="Feedback and Versions" eyebrow="Revision loop">
              <textarea className="min-h-24 w-full rounded-lg border border-slate-200 p-2" value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <button disabled={busy || !activeProject} onClick={reviseProject} className="mt-3 inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 disabled:opacity-50">
                <RotateCcw size={16} /> Create revision
              </button>
            </Card>
          </section>

          {activeProject && (
            <Card icon={Boxes} title="Active Package" eyebrow="Artifacts">
              {courseVideo && (
                <div className="mb-4 rounded-xl border border-sky-200 bg-sky-50 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-sky-950">Multi-source course video</p>
                      <p className="text-xs text-sky-700">
                        {courseVideo.slide_count || "?"} slides | {Math.round(courseVideo.duration || 0)} seconds | {courseVideo.bundle?.source_count || 0} sources
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs">
                      {Object.entries(courseVideoArtifacts).map(([key, rel]) => (
                        <a key={key} className="rounded-md bg-white px-2 py-1 font-semibold ring-1 ring-sky-200" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/${rel}`}>
                          {key.replaceAll("_", " ")}
                        </a>
                      ))}
                    </div>
                  </div>
                  {courseVideoArtifacts.video && (
                    <video className="mt-3 aspect-video w-full rounded-lg bg-black" controls preload="metadata" src={`/api/aimooc/projects/${activeProject.project_id}/artifacts/${courseVideoArtifacts.video}`} />
                  )}
                </div>
              )}
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
                <a className="rounded-lg border border-slate-200 p-3 hover:bg-slate-50" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/course_plan.json`}>
                  course_plan.json
                </a>
                <a className="rounded-lg border border-slate-200 p-3 hover:bg-slate-50" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/source_manifest.json`}>
                  source_manifest.json
                </a>
                <a className="rounded-lg border border-slate-200 p-3 hover:bg-slate-50" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/course_package_manifest.json`}>
                  package_manifest.json
                </a>
              </div>
              <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
                {lessons.map((lesson) => (
                  <div key={lesson.lesson_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="font-semibold">{lesson.title}</p>
                      <Film size={16} className="text-slate-500" />
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs">
                      {(lesson.artifacts || []).map((artifact) => (
                        <a key={artifact} className="rounded-md bg-white px-2 py-1 ring-1 ring-slate-200" href={`/api/aimooc/projects/${activeProject.project_id}/artifacts/${lesson.lesson_id}/${artifact}`}>
                          {artifact}
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <pre className="mt-4 max-h-96 overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-100">
                {JSON.stringify(activeProject.course_plan, null, 2)}
              </pre>
            </Card>
          )}

          <Card icon={Sparkles} title="Design Notes" eyebrow="UI">
            <p className="text-sm text-slate-600">
              The layout follows modern React and Tailwind dashboard patterns: clear mode switch, persistent source panel,
              compact runtime controls, and artifact-first review cards.
            </p>
          </Card>
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
