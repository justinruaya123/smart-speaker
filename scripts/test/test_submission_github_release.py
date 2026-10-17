"""Release integrity checks use small real ZIPs, without network credentials."""
import hashlib
from contextlib import redirect_stderr, redirect_stdout
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from urllib.error import HTTPError
import zipfile


ROOT = Path(__file__).resolve().parents[2]


class GitHubReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        model = b"selected finite model"
        self.model_hash = hashlib.sha256(model).hexdigest()
        assets = []
        for kind in ("inference", "benchmark"):
            archive = self.path / (kind + ".zip")
            manifest = {"files": {"submission/models/full/model.onnx": self.model_hash},
                        "parameters": 2066831, "actions_enabled": False,
                        "qualified": False, "purpose": "inference-only"}
            if kind == "benchmark":
                manifest.pop("parameters")
                manifest.pop("qualified")
                manifest.update(purpose="silent-benchmark-2m", model_choice="full")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("submission/models/full/model.onnx", model)
                handle.writestr("SUBMISSION-MANIFEST.json", json.dumps(manifest))
            assets.append({"filename": archive.name, "kind": kind,
                           "bytes": archive.stat().st_size,
                           "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                           "url": "https://example.invalid/" + archive.name})
        self.receipt = {"assets": assets, "model_sha256": self.model_hash}
        self.receipt_path = self.path / "downloads.json"
        self.receipt_path.write_text(json.dumps(self.receipt))

    def run_check(self):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/submission/github_release.py"),
             "--verify-only", "--download-dir", str(self.path),
             "--downloads-file", str(self.receipt_path)],
            capture_output=True, text=True, cwd=ROOT)

    def test_valid_archives_are_verified_without_network_or_token(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["verified_assets"], 2)

    def rewrite_first(self, change):
        archive = self.path / self.receipt["assets"][0]["filename"]
        with zipfile.ZipFile(archive) as handle:
            entries = {name: handle.read(name) for name in handle.namelist()}
        change(entries)
        with zipfile.ZipFile(archive, "w") as handle:
            for name, data in entries.items():
                handle.writestr(name, data)
        self.receipt["assets"][0].update(
            bytes=archive.stat().st_size,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        self.receipt_path.write_text(json.dumps(self.receipt))

    def test_outer_corruption_is_rejected(self):
        archive = self.path / self.receipt["assets"][0]["filename"]
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("size", result.stderr.lower())

    def test_changed_model_is_rejected_even_with_valid_outer_hash(self):
        self.rewrite_first(lambda entries: entries.update(
            {"submission/models/full/model.onnx": b"wrong model"}))
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("member hash", result.stderr.lower())

    def test_unlisted_member_is_rejected(self):
        self.rewrite_first(lambda entries: entries.update({"private.wav": b"capture"}))
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inventory", result.stderr.lower())

    def release_fixture(self, draft):
        return {"id": 7, "draft": draft, "target_commitish": "a" * 40,
                "tag_name": "2m-submission-v1", "html_url": "https://github.com/example/repo/releases/tag/2m-submission-v1",
                "assets": [{"name": asset["filename"], "size": asset["bytes"],
                            "digest": "sha256:" + asset["sha256"],
                            "browser_download_url": "https://example.invalid/" + asset["filename"]}
                           for asset in self.receipt["assets"]]}

    def release_args(self):
        return SimpleNamespace(repository="example/repo", target="a" * 40,
                               tag="2m-submission-v1", download_dir=self.path)

    def test_wrong_existing_tag_blocks_publication(self):
        from scripts.submission import github_release as release
        fixture = self.release_fixture(True)

        def request(route, token, method="GET", payload=None, raw=None):
            if "/git/ref/tags/" in route:
                return {"object": {"type": "commit", "sha": "b" * 40}}
            if method == "PATCH":
                return {**fixture, "draft": False}
            return fixture

        status = {"release_id": 7, "assets": self.receipt["assets"]}
        with patch.object(release, "github_request", side_effect=request):
            with self.assertRaisesRegex(ValueError, "tag"):
                release.publish_verified_draft(self.release_args(), status, "test-token")

    def test_identical_published_release_is_verified_without_mutation(self):
        from scripts.submission import github_release as release
        fixture = self.release_fixture(False)

        def request(route, token, method="GET", payload=None, raw=None):
            if method != "GET":
                self.fail("An identical published release must not be modified")
            if "/git/ref/tags/" in route:
                return {"object": {"type": "commit", "sha": "a" * 40}}
            return [fixture]

        with patch.object(release, "github_request", side_effect=request):
            try:
                result = release.prepare_draft(self.release_args(), self.receipt["assets"], "test-token")
            except ValueError as error:
                self.fail(str(error))
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["release_id"], 7)

    def test_transient_receipt_error_retains_confirmed_publication(self):
        from scripts.submission import github_release as release
        fixture = self.release_fixture(False)
        writes = []
        attempts = 0

        def request(route, token, method="GET", payload=None, raw=None):
            nonlocal attempts
            if "/contents/" in route:
                if method == "GET":
                    raise HTTPError("https://api.github.com/receipt", 404, "Not Found", {}, None)
                attempts += 1
                if attempts == 1:
                    raise TimeoutError("temporary status receipt timeout")
                writes.append(json.loads(base64.b64decode(payload["content"])))
                return {"content": {"sha": "c" * 40}}
            if method != "GET":
                self.fail("An identical published release must not be modified")
            if "/git/ref/tags/" in route:
                return {"object": {"type": "commit", "sha": "a" * 40}}
            return [fixture]

        argv = ["--draft", "--publish", "--target", "a" * 40,
                "--repository", "example/repo", "--status-branch", "codex/final-submission",
                "--download-dir", str(self.path), "--downloads-file", str(self.receipt_path)]
        with patch.object(release, "github_request", side_effect=request), \
                patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = release.main(argv)
        self.assertEqual(result, 0)
        self.assertEqual(writes[-1]["status"], "published")
        self.assertEqual(writes[-1]["release_id"], 7)

    def test_api_errors_keep_their_http_code_and_explain_the_barrier(self):
        from scripts.submission import github_release as release
        error = HTTPError("https://api.github.com/example", 403, "Forbidden", {},
                          io.BytesIO(b'{"message":"Resource not accessible by integration"}'))
        with patch.object(release, "urlopen", side_effect=error):
            with self.assertRaises(HTTPError) as caught:
                release.github_request("/example", "test-token")
        self.assertEqual(caught.exception.code, 403)
        self.assertIn("Resource not accessible by integration", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
