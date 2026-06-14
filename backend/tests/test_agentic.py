import base64
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
import pymupdf

from backend.agentic import (
    _action_arguments,
    _agent_list_text,
    _fallback_success_score,
    _is_repeated_delegation,
    _looks_like_clarification_request,
    _repair_director_action,
    _runtime_action_guidance,
    _upsert_generated_document,
    build_director_prompt,
    build_sub_agent_prompt,
    parse_director_action,
)
from backend.documents import create_document, normalize_upload_to_markdown, search_documents
from backend.multi_agent import validate_participant_payloads
from backend.scrape import _extract_readable_structure, scrape_website_content


class AgenticHelpersTest(unittest.TestCase):
    def test_parse_director_action_from_json_fence(self):
        raw = '```json\n{"action": "complete", "rationale": "done", "arguments": {"final_answer": "ok"}}\n```'
        action = parse_director_action(raw)
        self.assertEqual(action["action"], "complete")
        self.assertEqual(action["rationale"], "done")

    def test_build_sub_agent_prompt_includes_task_only(self):
        prompt = build_sub_agent_prompt(
            {"name": "Analyst", "personality": "Data specialist"},
            "Summarize risks",
            "Bullet list",
            "Evidence A",
        )
        self.assertIn("Summarize risks", prompt)
        self.assertIn("Data specialist", prompt)
        self.assertNotIn("roundtable", prompt.lower())

    def test_build_director_prompt_renders_placeholders(self):
        prompt = build_director_prompt(
            template="Goal: {goal}\nCriteria: {success_criteria}\nAgents: {agent_list}",
            goal="Ship report",
            success_criteria="Complete memo",
            report_format="RCA",
            agent_list="- id=1 name=Analyst",
            document_list="(none)",
            scrape_url="https://example.com",
        )
        self.assertIn("Ship report", prompt)
        self.assertIn("Complete memo", prompt)
        self.assertIn("Analyst", prompt)

    def test_runtime_guidance_disables_document_tools_when_empty(self):
        guidance = _runtime_action_guidance(
            [{"id": 7, "name": "Analyst"}],
            [],
            None,
        )
        self.assertIn("No documents are currently available", guidance)
        self.assertIn("Do not use search_documents or read_document", guidance)
        self.assertIn("Valid participant_id values for call_agent: 7", guidance)

    def test_agent_list_includes_roles_and_ids(self):
        roster = _agent_list_text(
            [
                {
                    "id": 7,
                    "name": "Analyst",
                    "personality": "Financial risk reviewer",
                    "llm_model": "qwen",
                }
            ]
        )
        self.assertIn("id=7", roster)
        self.assertIn("Analyst", roster)
        self.assertIn("Financial risk reviewer", roster)

    def test_action_arguments_normalizes_non_dict(self):
        self.assertEqual(_action_arguments({"arguments": "participant_id=1"}), {})

    def test_detects_agent_clarification_request(self):
        self.assertTrue(_looks_like_clarification_request("Could you clarify which audience this report targets?"))
        self.assertFalse(_looks_like_clarification_request("I assumed the audience is executives and completed the task."))

    def test_repeated_delegation_detects_same_agent_and_task(self):
        prior_steps = [
            {
                "message_kind": "director_delegation",
                "content": "Delegate to Jack:\nWrite LinkedIn copy",
                "metadata_json": json.dumps(
                    {"participant_id": 19, "arguments": {"task": "Write LinkedIn copy"}}
                ),
            }
        ]
        self.assertTrue(_is_repeated_delegation(19, "Write LinkedIn copy", prior_steps))
        self.assertFalse(_is_repeated_delegation(20, "Write LinkedIn copy", prior_steps))
        self.assertFalse(_is_repeated_delegation(19, "Review the LinkedIn copy", prior_steps))

    def test_fallback_success_score_gives_partial_credit_for_agent_content(self):
        score = _fallback_success_score(
            "Here is a draft LinkedIn post with a hook, call to action, and suggested assumptions for educators."
            " It includes concrete messaging and a meeting-booking ask."
        )
        self.assertGreater(score, 0)
        self.assertEqual(_fallback_success_score("Could you clarify the target audience?"), 5)

    def test_repair_director_action_defaults_only_participant(self):
        repaired, repairs = _repair_director_action(
            {
                "action": "call_agent",
                "rationale": "Ask the specialist to analyze the scraped content.",
                "arguments": {},
            },
            [{"id": 19, "name": "Researcher"}],
            "Assess the website.",
        )
        self.assertEqual(repaired["arguments"]["participant_id"], 19)
        self.assertIn("Ask the specialist", repaired["arguments"]["task"])
        self.assertTrue(repairs)

    def test_repair_director_action_matches_role_with_many_participants(self):
        repaired, repairs = _repair_director_action(
            {
                "action": "call_agent",
                "rationale": "Ask the legal compliance reviewer to assess risks.",
                "arguments": {"task": "Review legal compliance risks", "expected_output": "Risk list"},
            },
            [
                {"id": 19, "name": "Market Analyst", "personality": "Market research and demand forecasting"},
                {"id": 20, "name": "Legal Reviewer", "personality": "Legal compliance and policy risk review"},
            ],
            "Assess the website.",
        )
        self.assertEqual(repaired["arguments"]["participant_id"], 20)
        self.assertTrue(any("Resolved participant_id" in repair for repair in repairs))


class DocumentsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "test.db"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_markdown TEXT NOT NULL,
                source_filename TEXT,
                source_media_type TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT
            );
            INSERT INTO conversations (id, title) VALUES (1, 'Test');
            """
        )

    def tearDown(self):
        self.connection.close()
        self.tempdir.cleanup()

    def test_normalize_upload_markdown(self):
        markdown, media, metadata = normalize_upload_to_markdown("notes.md", "text/markdown", b"# Title\n\nBody")
        self.assertIn("Title", markdown)
        self.assertEqual(media, "text/markdown")
        self.assertIsNone(metadata)

    def test_normalize_upload_pdf_stores_image_metadata_only(self):
        image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PDF Title\nBody text")
        page.insert_image(pymupdf.Rect(72, 92, 82, 102), stream=image_bytes)
        pdf_bytes = pdf.tobytes()
        pdf.close()

        markdown, media, metadata = normalize_upload_to_markdown("brief.pdf", "application/pdf", pdf_bytes)

        self.assertEqual(media, "application/pdf")
        self.assertIn("PDF Title", markdown)
        self.assertNotIn("![", markdown)
        self.assertEqual(metadata["converter"]["name"], "pymupdf4llm")
        self.assertEqual(metadata["images"]["count"], 1)
        self.assertEqual(metadata["images"]["handling"], "metadata_only_no_interpretation")
        self.assertNotIn("image", metadata["images"]["items"][0])

    def test_create_and_search_documents(self):
        created = create_document(
            self.connection,
            1,
            title="Brief",
            content_markdown="# Brief\n\nFindings",
            kind="uploaded",
        )
        docs = search_documents([created], "findings")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["title"], "Brief")

    def test_upsert_generated_report_keeps_one_active_report(self):
        first = _upsert_generated_document(
            self.connection,
            1,
            kind="generated_report",
            title="Draft answer",
            content_markdown="First draft",
            metadata={"draft": 1},
        )
        second = _upsert_generated_document(
            self.connection,
            1,
            kind="generated_report",
            title="Final answer",
            content_markdown="Final draft",
            metadata={"draft": 2},
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["title"], "Final answer")
        self.assertEqual(second["content_markdown"], "Final draft")
        active_reports = [
            document
            for document in self.connection.execute(
                "SELECT * FROM documents WHERE kind = 'generated_report' AND archived_at IS NULL"
            ).fetchall()
        ]
        self.assertEqual(len(active_reports), 1)

    def test_upsert_generated_process_is_separate_from_output_report(self):
        output = _upsert_generated_document(
            self.connection,
            1,
            kind="generated_report",
            title="Final answer",
            content_markdown="Answer",
            metadata={"document_role": "output"},
        )
        process = _upsert_generated_document(
            self.connection,
            1,
            kind="generated_process",
            title="Process and rationale",
            content_markdown="Rationale",
            metadata={"document_role": "process"},
        )
        self.assertNotEqual(output["id"], process["id"])
        active_generated = self.connection.execute(
            "SELECT kind, COUNT(*) AS count FROM documents WHERE archived_at IS NULL GROUP BY kind"
        ).fetchall()
        counts = {row["kind"]: row["count"] for row in active_generated}
        self.assertEqual(counts["generated_report"], 1)
        self.assertEqual(counts["generated_process"], 1)


class ParticipantValidationTest(unittest.TestCase):
    def test_discussion_mode_caps_participants(self):
        participants = [SimpleNamespace(llm_model_id=1) for _ in range(4)]
        with self.assertRaises(HTTPException):
            validate_participant_payloads(participants, None, lambda *_args, **_kwargs: {}, mode="discussion")

    def test_agentic_mode_allows_many_participants(self):
        participants = [SimpleNamespace(llm_model_id=1) for _ in range(6)]
        validate_participant_payloads(participants, None, lambda *_args, **_kwargs: {}, mode="agentic")


class ScrapeStructureTest(unittest.TestCase):
    def test_beautiful_soup_extracts_readable_structure(self):
        html = """
        <html>
          <head><title>Course Catalog</title><script>ignore()</script></head>
          <body>
            <main>
              <h1>Programs</h1>
              <h2>Engineering</h2>
              <p>Mechanical engineering program details.</p>
              <ul><li>Admissions</li><li>Tuition</li></ul>
            </main>
          </body>
        </html>
        """
        readable, structure = _extract_readable_structure(html, "https://example.com")
        self.assertIn("# Course Catalog", readable)
        self.assertIn("## Engineering", readable)
        self.assertIn("- Admissions", readable)
        self.assertEqual(structure["title"], "Course Catalog")
        self.assertIn("Programs", structure["headings"])

    def test_focus_extract_keeps_full_content_available_to_memory(self):
        class FakeResponse:
            content = b"""
            <html>
              <head><title>Admissions</title></head>
              <body>
                <main>
                  <p>Alpha section.</p>
                  <p>Beta section with admissions.</p>
                  <p>Gamma section retained for later consultation.</p>
                </main>
              </body>
            </html>
            """
            encoding = "utf-8"

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, *_args, **_kwargs):
                return FakeResponse()

        with patch("backend.scrape.httpx.Client", FakeClient):
            result = scrape_website_content("https://example.com", query="admissions", depth=1)

        self.assertIn("Beta section", result["summary"])
        self.assertIn("Alpha section", result["content_markdown"])
        self.assertIn("Gamma section retained", result["content_markdown"])

    def test_spa_shell_uses_rendered_dom(self):
        class FakeResponse:
            content = b"""
            <html>
              <head><title>SPA</title></head>
              <body>
                <div id="root"></div>
                <script type="module" src="/assets/index-abc123.js"></script>
              </body>
            </html>
            """
            encoding = "utf-8"

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, *_args, **_kwargs):
                return FakeResponse()

        class FakePage:
            def goto(self, *_args, **_kwargs):
                return None

            def wait_for_load_state(self, *_args, **_kwargs):
                return None

            def wait_for_function(self, *_args, **_kwargs):
                return None

            def content(self):
                return """
                <html>
                  <head><title>Rendered App</title></head>
                  <body>
                    <main>
                      <h1>Rendered Course Catalog</h1>
                      <p>Real admissions content loaded by JavaScript.</p>
                    </main>
                  </body>
                </html>
                """

        class FakeContext:
            def new_page(self):
                return FakePage()

            def close(self):
                return None

        class FakeBrowser:
            def new_context(self, **_kwargs):
                return FakeContext()

            def close(self):
                return None

        class FakeChromium:
            def launch(self, **_kwargs):
                return FakeBrowser()

        class FakePlaywright:
            chromium = FakeChromium()

        class FakeSyncPlaywright:
            def __enter__(self):
                return FakePlaywright()

            def __exit__(self, *_args):
                return False

        with patch("backend.scrape.httpx.Client", FakeClient), patch(
            "backend.scrape.sync_playwright", lambda: FakeSyncPlaywright()
        ):
            result = scrape_website_content("https://example.com", depth=1)

        self.assertEqual(result["rendered_pages"], 1)
        self.assertEqual(result["render_errors"], [])
        self.assertIn("Rendered Course Catalog", result["content_markdown"])
        self.assertIn("Real admissions content loaded by JavaScript", result["content_markdown"])


if __name__ == "__main__":
    unittest.main()
