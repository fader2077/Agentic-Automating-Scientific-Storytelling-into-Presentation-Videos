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
    build_pipeline_command,
    build_steps,
    finalize_task_artifacts,
    initialize_task_runtime,
    load_settings,
    resolve_agent_skills_md,
    resolve_pipeline_python,
    write_task,
)

from src.aimooc_schema import AvatarConfig
from src.avatar_renderer import render_avatar_manifest
from src.cursor_overlay import render_cursor_overlay_timeline
from src.real_pipeline import (
    audit_asset_caption,
    build_srt_from_audio_transcript,
    build_srt_from_speech_manifest,
    ensure_reference_audio,
    expand_speaker_text,
    resolve_reference_voice,
    sanitize_reference_text,
    compact_slide_caption,
    compact_slide_captions,
    is_usable_ocr_asset,
    assign_visual_assets,
    select_visual_asset,
    visual_asset_display_kind,
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
        "speech_manifest": result_dir / "speech_manifest.json",
        "audio_transcript": result_dir / "audio_transcript.json",
        "agentic_pacing": result_dir / "agentic_pacing.json",
        "cursor": result_dir / "cursor.json",
        "subtitles": result_dir / "subtitles.srt",
        "video": result_dir / "3_merage.mp4",
    }
    files["ocr_markdown"].write_text("# Fixture OCR\n\nA figure and equation were extracted.", encoding="utf-8")
    files["slides_pdf"].write_bytes(b"%PDF-1.4\n% fixture\n")
    files["slides_tex"].write_text("\\documentclass{beamer}\n", encoding="utf-8")
    files["script"].write_text("Fixture narration. | center of slide\n", encoding="utf-8")
    files["speech_manifest"].write_text(json.dumps({"slides": []}), encoding="utf-8")
    files["audio_transcript"].write_text(json.dumps({"slides": []}), encoding="utf-8")
    files["agentic_pacing"].write_text(json.dumps({"total_slides": 12, "content_slides": 11}), encoding="utf-8")
    files["cursor"].write_text(json.dumps({"points": []}), encoding="utf-8")
    files["subtitles"].write_text("1\n00:00:00,000 --> 00:00:01,000\nFixture narration.\n", encoding="utf-8")
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
            str(files["video"]),
        ],
        check=True,
    )

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
    upload_id_2 = None
    aimooc_project_id = None
    aimooc_project_ids = []
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
            assert "aimooc" in graph_payload["flows"]
            assert "hermes_adapter" in graph_payload["flows"]["aimooc"]["frameworks"]
            assert "openclaw_adapter" in graph_payload["flows"]["aimooc"]["frameworks"]
            assert "CoursePlannerAgent" in graph_payload["aimooc_visited_check"]
            frameworks = client.get("/api/agent-frameworks")
            assert frameworks.status_code == 200
            assert any(item["key"] == "langgraph" for item in frameworks.json())
            assert any(item["key"] == "hermes_adapter" for item in frameworks.json())
            assert any(item["key"] == "openclaw_adapter" for item in frameworks.json())
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
            assert target_content_slide_count(6, 12) == 10
            assert 44 <= target_speaker_words(6, 12) <= 56
            assert 24 <= target_speaker_words(10) <= 60
            long_speaker = expand_speaker_text("This slide introduces the method.", "Method", ["contrastive training", "backdoor robustness"], 10)
            assert len(long_speaker.split()) >= 20
            natural_speaker = expand_speaker_text("This slide introduces the method.", "Method", ["contrastive training", "backdoor robustness"], 6, 42)
            bad_key = "Key " + "evidence"
            bad_evidence = "Evidence " + "to inspect"
            bad_audience = "The slide " + "mainly asks"
            bad_main = "The main " + "point is that"
            bad_placeholder = "paper evidence " + "to inspect"
            assert bad_key not in natural_speaker
            assert bad_evidence not in natural_speaker
            assert bad_audience not in natural_speaker
            assert bad_main not in natural_speaker
            assert bad_placeholder not in natural_speaker.lower()
            method_speaker = expand_speaker_text("", "Stage 2: Delta S Filter", ["Analyzes similarity shifts in embeddings", "Detects anomalous alignments"], 6, 42)
            assert "through analyzes" not in method_speaker.lower()
            assert "focuses on analyzing similarity shifts" in method_speaker.lower()
            intro_speaker = expand_speaker_text("", "Introduction", ["CLIP aligns images and text", "uncurated noisy internet data"], 6, 42)
            assert "Start by noting that" in intro_speaker
            assert bad_main not in intro_speaker
            intro_slide = {"title": "Introduction: The Rise of Multimodal Models", "bullets": ["CLIP aligns images and text", "Trained on web-scale pairs"]}
            bad_equation = {"id": "equation_01", "kind": "equation", "caption": "sim_before", "body": "sim_before", "image": "eq.jpg"}
            bad_logo = {"id": "image_01", "kind": "image", "caption": "OCR image from page 11 university logo", "body": "", "image": "logo.jpg"}
            bad_notation = {"id": "table_00", "kind": "table", "caption": "Table 1: Notation table for symbols used in Algorithms.", "body": "Notation table", "image": "notation.jpg"}
            bad_notation_list = {"id": "table_02", "kind": "table", "caption": "Table 9. List of " + "Notations and Parameters Used in This Thesis", "body": "", "image": "notation2.jpg"}
            good_table = {"id": "table_01", "kind": "table", "caption": "Table 1 CLIP image text retrieval benchmark", "body": "CLIP image text retrieval benchmark", "image": "table.jpg"}
            framework_image = {"id": "image_02", "kind": "image", "caption": "Figure 2 three-stage defense framework for TrustCLIP", "body": "framework defense method", "image": "framework.jpg"}
            result_slide = {"title": "Experimental Results", "bullets": ["CLIP image text retrieval benchmark", "Attack success rate improves"]}
            result_method_slide = {"title": "Results: Effectiveness", "bullets": ["Outperforms existing defense methods", "Reduces ASR to 1.4 percent"]}
            assert not is_usable_ocr_asset(bad_equation, "Introduction to multimodal CLIP")
            assert not is_usable_ocr_asset(bad_logo, "attack trigger defense")
            assert not is_usable_ocr_asset(bad_notation, "Introduction to multimodal CLIP")
            assert not is_usable_ocr_asset(bad_notation_list, "Graph construction")
            assert select_visual_asset(intro_slide, [bad_equation, bad_logo, bad_notation], set()) is None
            assert select_visual_asset(intro_slide, [bad_equation, bad_logo, good_table], set()) is None
            assert select_visual_asset(result_slide, [bad_equation, bad_logo, good_table], set()) == good_table
            assert select_visual_asset(result_method_slide, [framework_image, good_table], set()) == good_table
            assert assign_visual_assets([intro_slide, result_slide], [framework_image, good_table])[0] is None
            mislabeled_table = {"id": "chart_01", "kind": "chart", "caption": "Table 5: Ablation study on each stage.", "body": "", "image": "chart.jpg"}
            assert visual_asset_display_kind(mislabeled_table) == "Table"
            assert tts_pacing_for_minutes(6, 12)["voice_speed"] == 1.0
            assert tts_pacing_for_minutes(6, 12)["sentence_pause"] == 0.0
            assert tts_pacing_for_minutes(10)["voice_speed"] == 1.0
            assert tts_pacing_for_minutes(10)["sentence_pause"] == 0.0
            assert len(split_narration_chunks("One sentence. " * 40, max_chars=80)) > 1
            subtitle_cues = split_subtitle_cues("This is a long subtitle sentence that should not cover the slide content. " * 4)
            assert len(subtitle_cues) > 2
            assert max(len(cue.replace("\n", " ")) for cue in subtitle_cues) <= 60
            wide_subtitle_cues = split_subtitle_cues("This is a wider subtitle line that should stay readable without covering most of the slide. " * 4, max_words=18, max_chars=116)
            assert max(len(cue.replace("\n", " ")) for cue in wide_subtitle_cues) <= 116
            compact_caption = compact_slide_caption("This is a long subtitle sentence that should not cover the slide content.")
            assert 3 <= len(compact_caption.split()) <= 7
            compact_captions = compact_slide_captions("This is a long subtitle sentence. A second caption should stay readable and aligned.")
            assert 1 <= len(compact_captions) <= 2
            manifest_audio = fixture_dir / "manifest_audio"
            manifest_audio.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=24000:cl=mono",
                    "-t",
                    "4",
                    str(manifest_audio / "0.wav"),
                ],
                check=True,
            )
            manifest_path = manifest_audio / "speech_manifest.json"
            manifest_path.write_text(
                json.dumps({"slides": [{"slide_index": 0, "chunks": [{"text": "First full subtitle sentence. Second full subtitle sentence.", "start": 0, "end": 4}]}]}),
                encoding="utf-8",
            )
            manifest_srt = fixture_dir / "manifest.srt"
            assert build_srt_from_speech_manifest(manifest_path, manifest_audio, manifest_srt) is True
            assert "First full subtitle sentence" in manifest_srt.read_text(encoding="utf-8")
            assert build_srt_from_audio_transcript(manifest_audio, fixture_dir / "asr.srt", fixture_dir / "asr.json") is False
            contaminated_time_phrase = "24" + "-7"
            contaminated_topic_phrase = "sports and " + "politics"
            contaminated_show_phrase = "show " + "runs"
            assert contaminated_time_phrase not in sanitize_reference_text(f"A {contaminated_show_phrase} {contaminated_time_phrase} with a host.")
            assert "sports" not in sanitize_reference_text(
                f"to experts to discuss about {contaminated_topic_phrase}. Now imagine a {contaminated_show_phrase} {contaminated_time_phrase}."
            )
            assert audit_asset_caption("%%%% $$$$ @@@@ \u03b1\u03b2\u03b3", "chart", 7) == "OCR chart from page 7"

            fallback_ref = ensure_reference_audio(str(fixture_dir / "missing_reference.wav"), fixture_dir)
            assert fallback_ref.exists()
            assert fallback_ref.suffix == ".wav"
            resolved_ref, resolved_text = resolve_reference_voice(str(fixture_dir / "missing_reference_again.wav"), f"A {contaminated_show_phrase} {contaminated_time_phrase}.", fixture_dir)
            assert resolved_ref.exists()
            assert resolved_text is not None and contaminated_time_phrase not in resolved_text
            assert contaminated_show_phrase not in resolved_text
            if "basic_ref_en.wav" in str(resolved_ref):
                assert resolved_text == "Some call me nature, others call me mother nature."
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
            avatar_lesson_dir = fixture_dir / "avatar_lesson"
            avatar_manifest = render_avatar_manifest(
                avatar_lesson_dir,
                AvatarConfig(avatar_mode="presenter_card", position="bottom_right"),
                render_media=True,
                source_video=cursor_input,
            )
            assert avatar_manifest["rendered"] is True
            assert Path(avatar_manifest["video"]).exists()


            upload = client.post(
                "/api/upload",
                files={"file": ("fixture.pdf", b"%PDF-1.4\n% fixture\n", "application/pdf")},
            )
            assert upload.status_code == 200
            upload_payload = upload.json()
            upload_id = upload_payload["id"]
            upload_2 = client.post(
                "/api/upload",
                files={"file": ("reference.pdf", b"%PDF-1.4\n% fixture 2\n", "application/pdf")},
            )
            assert upload_2.status_code == 200
            upload_id_2 = upload_2.json()["id"]

            batch = client.post(
                "/api/uploads/batch",
                files=[
                    ("files", ("batch_a.pdf", b"%PDF-1.4\n% batch a\n", "application/pdf")),
                    ("files", ("batch_b.txt", b"course notes", "text/plain")),
                ],
            )
            assert batch.status_code == 200
            assert batch.json()["count"] == 2

            aimooc = client.post(
                "/api/aimooc/projects",
                json={
                    "sources": [
                        {"source_id": upload_id, "role": "primary", "priority": 1, "title": "Primary source"},
                        {"source_id": upload_id_2, "role": "reference", "priority": 3, "title": "Reference source"},
                    ],
                    "course_title": "AIMOOC Test Course",
                    "audience": "engineering students",
                    "learning_objectives": ["Understand source material", "Build lessons"],
                    "requirements": "Create a small test course.",
                    "total_minutes": 12,
                    "module_count": 2,
                    "lessons_per_module": 1,
                    "preferred_style": "teaching_walkthrough",
                    "language": "zh-TW",
                    "difficulty": "intermediate",
                    "include_quizzes": True,
                    "include_avatar": True,
                    "avatar_mode": "presenter_card",
                    "agentic_framework": "hermes_adapter",
                },
            )
            assert aimooc.status_code == 200, aimooc.text
            aimooc_payload = aimooc.json()
            aimooc_project_id = aimooc_payload["project_id"]
            aimooc_project_ids.append(aimooc_project_id)
            assert aimooc_payload["framework"] == "hermes_adapter"
            assert aimooc_payload["source_count"] == 2
            assert len(aimooc_payload["course_plan"]["modules"]) == 2
            assert aimooc_payload["package_manifest"]["version_id"] == "v001_initial"
            assert client.get("/aimooc").status_code == 200
            artifact = client.get(f"/api/aimooc/projects/{aimooc_project_id}/artifacts/course_plan.json")
            assert artifact.status_code == 200
            assert "AIMOOC Test Course" in artifact.text
            revision = client.post(
                f"/api/aimooc/projects/{aimooc_project_id}/revise",
                json={
                    "base_version": "v001_initial",
                    "feedback": [
                        {
                            "target_type": "lesson",
                            "target_id": "module_01_lesson_01",
                            "severity": "normal",
                            "instruction": "Make the first lesson more intuitive.",
                            "preferred_action": "simplify",
                        }
                    ],
                },
            )
            assert revision.status_code == 200, revision.text
            assert revision.json()["revision"]["version_id"].endswith("_feedback")
            versions = client.get(f"/api/aimooc/projects/{aimooc_project_id}/versions")
            assert versions.status_code == 200
            assert len(versions.json()) >= 2

            payload = CreateTaskPayload(
                upload_id=upload_id,
                goal_prompt="Create a rigorous academic video presentation.",
                desired_minutes=4,
                target_slide_count=12,
                preferred_slide_style="clean beamer academic deck",
                agentic_framework="openclaw_adapter",
                avatar_mode="presenter_card",
            )
            task = {
                "id": task_id,
                "status": "queued",
                "created_at": 0.0,
                "goal_prompt": payload.goal_prompt,
                "desired_minutes": payload.desired_minutes,
                "target_slide_count": payload.target_slide_count,
                "preferred_slide_style": payload.preferred_slide_style,
                "agentic_framework": payload.agentic_framework,
                "avatar_mode": payload.avatar_mode,
                "avatar_position": payload.avatar_position,
                "upload_id": upload_id,
                "upload_name": upload_payload["file_name"],
                "upload_path": upload_payload["saved_path"],
                "result_dir": str(fixture_dir.resolve()),
                "settings": load_settings(),
                "steps": build_steps(payload, load_settings()),
            }
            initialize_task_runtime(task)
            command = build_pipeline_command(task)
            assert "--avatar_mode" in command
            assert "presenter_card" in command
            task["status"] = "running"
            task["job"]["state"] = "running"
            apply_pipeline_event(task, {"kind": "start", "step": "mineru_ocr", "message": "MinerU OCR started.", "data": {}})
            apply_pipeline_event(task, {"kind": "done", "step": "mineru_ocr", "message": "MinerU OCR completed.", "data": metadata["steps"]["mineru_ocr"]})
            finalize_task_artifacts(task, metadata)
            task["status"] = "completed"
            task["job"]["state"] = "completed"
            write_task(task)

            rendered_aimooc = client.post(
                "/api/aimooc/projects",
                json={
                    "sources": [
                        {"source_id": upload_id, "role": "primary", "priority": 1, "title": "Primary source"},
                        {"source_id": upload_id_2, "role": "reference", "priority": 3, "title": "Reference source"},
                    ],
                    "course_title": "Rendered AIMOOC Test Course",
                    "audience": "engineering students",
                    "learning_objectives": ["Understand source material", "Build lessons"],
                    "requirements": "Create a rendered test course.",
                    "total_minutes": 12,
                    "module_count": 1,
                    "lessons_per_module": 1,
                    "preferred_style": "teaching_walkthrough",
                    "language": "zh-TW",
                    "difficulty": "intermediate",
                    "include_quizzes": True,
                    "include_avatar": True,
                    "avatar_mode": "presenter_card",
                    "agentic_framework": "openclaw_adapter",
                    "render_videos": True,
                    "lesson_video_task_id": task_id,
                },
            )
            assert rendered_aimooc.status_code == 200, rendered_aimooc.text
            rendered_payload = rendered_aimooc.json()
            aimooc_project_ids.append(rendered_payload["project_id"])
            assert rendered_payload["framework"] == "openclaw_adapter"
            assert any("avatar_video.mp4" in lesson.get("artifacts", []) for lesson in rendered_payload["package_manifest"]["lessons"])

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

            for artifact_name in [
                "ocr_markdown",
                "ocr_assets",
                "slides_pdf",
                "slides_tex",
                "script",
                "speech_manifest",
                "audio_transcript",
                "agentic_pacing",
                "cursor",
                "subtitles",
                "video",
                "sat",
                "token",
            ]:
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
        if upload_id_2:
            for upload_file in (ROOT / "web" / "data" / "uploads").glob(f"{upload_id_2}.*"):
                upload_file.unlink(missing_ok=True)
        for meta_path in (ROOT / "web" / "data" / "uploads").glob("*.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("file_name", "")).startswith("batch_"):
                for upload_file in (ROOT / "web" / "data" / "uploads").glob(f"{payload.get('id')}.*"):
                    upload_file.unlink(missing_ok=True)
        for project_id in aimooc_project_ids:
            project_dir = ROOT / "result" / "aimooc_projects" / project_id
            if project_dir.exists():
                shutil.rmtree(project_dir)
            db_path = ROOT / "web" / "data" / "aimooc.sqlite3"
            if db_path.exists():
                import sqlite3

                with sqlite3.connect(db_path) as conn:
                    conn.execute("DELETE FROM aimooc_feedback WHERE project_id=?", (project_id,))
                    conn.execute("DELETE FROM aimooc_versions WHERE project_id=?", (project_id,))
                    conn.execute("DELETE FROM aimooc_projects WHERE project_id=?", (project_id,))
                    conn.commit()
        speech_skills_path.write_bytes(speech_skills_bytes)
        if fixture_dir.exists():
            shutil.rmtree(fixture_dir)


if __name__ == "__main__":
    main()









