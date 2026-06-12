import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
import types
from pathlib import Path

os.environ["WEB_DISABLE_WORKER_THREAD"] = "1"

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "whisperx" not in sys.modules:
    whisperx_stub = types.ModuleType("whisperx")
    whisperx_stub.load_model = lambda *args, **kwargs: None
    whisperx_stub.load_align_model = lambda *args, **kwargs: (None, None)
    whisperx_stub.align = lambda *args, **kwargs: {"segments": []}
    sys.modules["whisperx"] = whisperx_stub

if "f5_tts.api" not in sys.modules:
    f5_pkg = types.ModuleType("f5_tts")
    f5_api = types.ModuleType("f5_tts.api")

    class F5TTSStub:
        def infer(self, *args, **kwargs):
            return None

    f5_api.F5TTS = F5TTSStub
    sys.modules["f5_tts"] = f5_pkg
    sys.modules["f5_tts.api"] = f5_api
from web.app import (
    CreateTaskPayload,
    apply_pipeline_event,
    app,
    build_steps,
    finalize_task_artifacts,
    initialize_task_runtime,
    load_settings,
    resolve_agent_skills_md,
    resolve_pipeline_python,
    write_task,
)

from src.cursor_overlay import render_cursor_overlay_timeline
from src.real_pipeline import (
    audit_asset_caption,
    ensure_reference_audio,
    expand_speaker_text,
    compact_slide_caption,
    compact_slide_captions,
    split_subtitle_cues,
    target_content_slide_count,
    target_speaker_words,
    tts_pacing_for_minutes,
)
from src.speech_synth import split_narration_chunks
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def write_fixture_result(result_dir: Path) -> dict:
    if result_dir.exists():
        shutil.rmtree(result_dir)
    asset_dir = result_dir / "latex_proj" / "ocr_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "image_01.jpg"
    image_path.write_bytes(PNG_1X1)

    files = {
        "ocr_markdown": result_dir / "ocr.md",
        "slides_pdf": result_dir / "slides.pdf",
        "slides_tex": result_dir / "slides.tex",
        "script": result_dir / "subtitle_w_cursor.txt",
        "cursor": result_dir / "cursor.json",
        "subtitles": result_dir / "subtitles.srt",
        "video": result_dir / "3_merage.mp4",
    }
    files["ocr_markdown"].write_text("# Fixture OCR\n\nA figure and equation were extracted.", encoding="utf-8")
    files["slides_pdf"].write_bytes(b"%PDF-1.4\n% fixture\n")
    files["slides_tex"].write_text("\\documentclass{beamer}\n", encoding="utf-8")
    files["script"].write_text("Fixture narration. | center of slide\n", encoding="utf-8")
    files["cursor"].write_text(json.dumps({"points": []}), encoding="utf-8")
    files["subtitles"].write_text("1\n00:00:00,000 --> 00:00:01,000\nFixture narration.\n", encoding="utf-8")
    files["video"].write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    ocr_assets = result_dir / "ocr_assets.json"
    ocr_assets.write_text(
        json.dumps(
            {
                "source": "fixture",
                "asset_dir": str(asset_dir),
                "counts": {"image": 1},
                "assets": [
                    {
                        "id": "image_01",
                        "kind": "image",
                        "page": 1,
                        "caption": "Fixture visual asset",
                        "image": str(image_path),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files["ocr_assets"] = ocr_assets

    metadata = {
        "mode": "fixture",
        "model": "qwen3.6:27b",
        "ollama_url": "http://127.0.0.1:11434",
        "paper_pdf": "fixture.pdf",
        "steps": {
            "mineru_ocr": {"seconds": 0.01, "assets": 1},
            "ollama_plan": {"seconds": 0.01, "slides": 4},
            "beamer": {"seconds": 0.01, "slide_count": 4},
            "script": {"seconds": 0.01},
            "tts": {"seconds": 0.01, "audio_files": 4},
            "cursor": {"seconds": 0.01},
            "video": {"seconds": 0.01, "duration": 12.0},
        },
        "total_seconds": 0.07,
        "artifacts": {key: str(path) for key, path in files.items()},
    }
    (result_dir / "sat.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (result_dir / "token.json").write_text(json.dumps({"mode": "fixture"}), encoding="utf-8")
    return metadata


def main() -> None:
    fixture_dir = ROOT / "result" / "test_api_fixture"
    metadata = write_fixture_result(fixture_dir)
    task_id = "test-" + uuid.uuid4().hex
    upload_id = None
    speech_skills_path = resolve_agent_skills_md("SpeechAgent")
    speech_skills_bytes = speech_skills_path.read_bytes()

    try:
        with TestClient(app) as client:
            health = client.get("/api/health")
            assert health.status_code == 200
            health_payload = health.json()
            assert health_payload["ollama_url"] == "http://127.0.0.1:11434"
            assert health_payload["gpu"]["cuda_available"] is True
            assert Path(health_payload["pipeline_python"]).exists()
            assert resolve_pipeline_python().exists()

            settings = {
                "ollama_url": "http://127.0.0.1:11434",
                "text_model": "qwen3.6:27b",
                "vision_model": "qwen2.5vl:latest",
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 8192,
                "system_prompt": "Coordinate a real academic presentation-video pipeline with faithful paper coverage.",
                "tick_seconds": 1,
                "step_ticks": {
                    "ingest": 16,
                    "planner": 18,
                    "slides": 22,
                    "script": 18,
                    "tts": 20,
                    "cursor": 16,
                    "compose": 18,
                },
            }
            save = client.post("/api/settings", json=settings)
            assert save.status_code == 200
            assert save.json()["text_model"] == "qwen3.6:27b"

            ollama = client.post(
                "/api/settings/test-ollama",
                json={
                    "ollama_url": settings["ollama_url"],
                    "text_model": settings["text_model"],
                    "vision_model": settings["vision_model"],
                },
            )
            assert ollama.status_code == 200
            assert ollama.json()["ok"] is True
            models = client.get("/api/ollama/models")
            assert models.status_code == 200
            assert "models" in models.json()

            styles = client.get("/api/slide-styles")
            assert styles.status_code == 200
            styles_payload = styles.json()
            assert len(styles_payload) >= 3
            assert all(item.get("key") and item.get("title") and item.get("value") and item.get("preview") for item in styles_payload)

            catalog = client.get("/api/tool-catalog")
            assert catalog.status_code == 200
            assert len(catalog.json()) >= 8
            assert any(item["key"] == "ocr_asset_manifest" for item in catalog.json())
            assert any(item["uses_gpu"] is True for item in catalog.json())

            tools_manifest = client.get("/api/tools/manifest")
            assert tools_manifest.status_code == 200
            assert tools_manifest.json()["manifest_path"].replace("\\", "/").endswith("src/tools/manifest.json")

            agents = client.get("/api/agent-catalog")
            assert agents.status_code == 200
            agents_payload = agents.json()
            assert any(item["key"] == "SpeechAgent" for item in agents_payload)
            assert any(tool["key"] == "f5_queue" for item in agents_payload for tool in item["tools"])
            agent_graph = client.get("/api/agent-graph")
            assert agent_graph.status_code == 200
            graph_payload = agent_graph.json()
            assert graph_payload["framework"] == "langgraph"
            assert graph_payload["compiled"] is True
            assert graph_payload["execution_model"] == "supervisor + conditional edges + parallel fanout + repair cycles"
            assert graph_payload["visited_check"][0] == "SupervisorAgent"
            assert graph_payload["visited_check"][-1] == "RenderAgent"
            assert any(edge["type"] == "parallel_fanout" for edge in graph_payload["edges"])
            assert any(call.startswith("PlannerAgent.") for call in graph_payload["tool_call_check"])
            skills = client.get("/api/agents/SpeechAgent/skills.md")
            assert skills.status_code == 200
            assert "F5TTS synthesis" in skills.text
            original_skills = skills.text
            edited_skills = original_skills.rstrip() + "\n\n<!-- api edit roundtrip -->\n"
            update_skills = client.put("/api/agents/SpeechAgent/skills.md", json={"content": edited_skills})
            assert update_skills.status_code == 200
            assert update_skills.json()["ok"] is True
            assert "api edit roundtrip" in client.get("/api/agents/SpeechAgent/skills.md").text
            restore_skills = client.put("/api/agents/SpeechAgent/skills.md", json={"content": original_skills})
            assert restore_skills.status_code == 200

            assert target_content_slide_count(6) <= 14
            assert target_speaker_words(10) >= 60
            long_speaker = expand_speaker_text("This slide introduces the method.", "Method", ["contrastive training", "backdoor robustness"], 10)
            assert len(long_speaker.split()) >= 60
            assert tts_pacing_for_minutes(6)["voice_speed"] <= 0.65
            assert tts_pacing_for_minutes(10)["voice_speed"] <= 0.9
            assert tts_pacing_for_minutes(10)["sentence_pause"] >= 1.0
            assert len(split_narration_chunks("One sentence. " * 40, max_chars=80)) > 1
            subtitle_cues = split_subtitle_cues("This is a long subtitle sentence that should not cover the slide content. " * 4)
            assert len(subtitle_cues) > 2
            assert max(len(cue.replace("\n", " ")) for cue in subtitle_cues) <= 60
            compact_caption = compact_slide_caption("This is a long subtitle sentence that should not cover the slide content.")
            assert 3 <= len(compact_caption.split()) <= 7
            compact_captions = compact_slide_captions("This is a long subtitle sentence. A second caption should stay readable and aligned.")
            assert 1 <= len(compact_captions) <= 2
            assert audit_asset_caption("%%%% $$$$ @@@@ \u03b1\u03b2\u03b3", "chart", 7) == "OCR chart from page 7"

            fallback_ref = ensure_reference_audio(str(fixture_dir / "missing_reference.wav"), fixture_dir)
            assert fallback_ref.exists()
            assert fallback_ref.suffix == ".wav"
            cursor_input = fixture_dir / "cursor_input.mp4"
            cursor_output = fixture_dir / "cursor_output.mp4"
            cursor_json = fixture_dir / "cursor_overlay.json"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=white:s=320x180:d=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-shortest",
                    str(cursor_input),
                ],
                check=True,
            )
            cursor_json.write_text(json.dumps([{"start": 0.0, "cursor": [160, 90]}]), encoding="utf-8")
            render_cursor_overlay_timeline(str(cursor_input), str(cursor_output), str(cursor_json), cursor_size=12)
            assert cursor_output.exists()
            assert cursor_output.stat().st_size > 0


            upload = client.post(
                "/api/upload",
                files={"file": ("fixture.pdf", b"%PDF-1.4\n% fixture\n", "application/pdf")},
            )
            assert upload.status_code == 200
            upload_payload = upload.json()
            upload_id = upload_payload["id"]

            payload = CreateTaskPayload(
                upload_id=upload_id,
                goal_prompt="Create a rigorous academic video presentation.",
                desired_minutes=4,
                preferred_slide_style="clean beamer academic deck",
            )
            task = {
                "id": task_id,
                "status": "queued",
                "created_at": 0.0,
                "goal_prompt": payload.goal_prompt,
                "desired_minutes": payload.desired_minutes,
                "preferred_slide_style": payload.preferred_slide_style,
                "upload_id": upload_id,
                "upload_name": upload_payload["file_name"],
                "upload_path": upload_payload["saved_path"],
                "result_dir": str(fixture_dir.resolve()),
                "settings": load_settings(),
                "steps": build_steps(payload, load_settings()),
            }
            initialize_task_runtime(task)
            task["status"] = "running"
            task["job"]["state"] = "running"
            apply_pipeline_event(task, {"kind": "start", "step": "mineru_ocr", "message": "MinerU OCR started.", "data": {}})
            apply_pipeline_event(task, {"kind": "done", "step": "mineru_ocr", "message": "MinerU OCR completed.", "data": metadata["steps"]["mineru_ocr"]})
            finalize_task_artifacts(task, metadata)
            task["status"] = "completed"
            task["job"]["state"] = "completed"
            write_task(task)

            task_response = client.get(f"/api/tasks/{task_id}")
            assert task_response.status_code == 200
            task_payload = task_response.json()
            assert task_payload["artifact_paths"]["video"].endswith("3_merage.mp4")
            assert task_payload["steps"][0]["status"] == "completed"

            timeline = client.get(f"/api/tasks/{task_id}/timeline")
            assert timeline.status_code == 200
            assert len(timeline.json()["timeline"]) >= 3

            ocr_assets = client.get(f"/api/tasks/{task_id}/ocr-assets")
            assert ocr_assets.status_code == 200
            ocr_payload = ocr_assets.json()
            assert ocr_payload["counts"]["image"] == 1
            image_asset = next(item for item in ocr_payload["assets"] if item.get("image_url"))
            ocr_image = client.get(image_asset["image_url"])
            assert ocr_image.status_code == 200
            assert ocr_image.headers["content-type"].startswith("image/")

            for artifact_name in ["ocr_markdown", "ocr_assets", "slides_pdf", "slides_tex", "script", "cursor", "subtitles", "video", "sat", "token"]:
                artifact = client.get(f"/api/tasks/{task_id}/artifacts/{artifact_name}")
                assert artifact.status_code == 200, artifact_name

            root = client.get("/")
            assert root.status_code == 200
            assert "Task History" in root.text
            assert "Agentic graph" in root.text
            assert "agent-graph" in root.text
            assert client.get("/history").status_code == 200
            replay_page = client.get(f"/replay/{task_id}")
            assert replay_page.status_code == 200
            assert "Timeline Replay" in replay_page.text

            print("web_api_test_ok", task_id)
    finally:
        task_file = ROOT / "web" / "data" / "tasks" / f"{task_id}.json"
        task_file.unlink(missing_ok=True)
        if upload_id:
            for upload_file in (ROOT / "web" / "data" / "uploads").glob(f"{upload_id}.*"):
                upload_file.unlink(missing_ok=True)
        speech_skills_path.write_bytes(speech_skills_bytes)
        if fixture_dir.exists():
            shutil.rmtree(fixture_dir)


if __name__ == "__main__":
    main()









