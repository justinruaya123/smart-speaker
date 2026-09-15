"""Check model integrity and pinned acquisition without downloading a dataset."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from model.scripts import acquire
from scripts.check_setup import verify_bundle


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data/manifests").mkdir(parents=True)
        self.raw = self.root / "data/raw"
        self.raw.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(acquire, "ROOT", self.root).start()
        patch.object(acquire, "RAW", self.raw).start()

    def ledger(self, name, data):
        (self.root / "data/manifests" / f"{name}.source.json").write_text(json.dumps(data))

    def source(self, path, content=b"pinned public bytes"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"path": path, "sha256": hashlib.sha256(content).hexdigest()}

    def test_model_verification_detects_same_size_corruption(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        files = {}
        for model in ("vcm", "vcm_stream", "asr"):
            for variant in ("fp32", "int8"):
                name = f"{model}.{variant}.onnx"
                content = name.encode()
                (bundle / name).write_bytes(content)
                files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        (bundle / "bundle.json").write_text(json.dumps({"files": files}))
        self.assertEqual(verify_bundle(bundle), sorted(files))
        target = bundle / "vcm.fp32.onnx"
        target.write_bytes(b"x" * target.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_bundle(bundle)

    def test_missing_source_pin_fails_without_network_fallback(self):
        with patch.object(acquire, "get_json") as network:
            with self.assertRaisesRegex(RuntimeError, "Missing pinned source ledger"):
                acquire.acquire_slurp(audio=False)
        network.assert_not_called()

    def prepare_metadata(self):
        names = ("train.jsonl", "devel.jsonl", "test.jsonl", "metadata.json")
        files = [self.source(f"data/raw/slurp/{name}") for name in names]
        self.ledger("slurp", {"revision": "a" * 40, "files": files})

    def test_metadata_uses_pinned_revision_and_checks_local_bytes(self):
        self.prepare_metadata()
        with patch.object(acquire, "get_json", side_effect=AssertionError("Latest revision requested")), \
             patch.object(acquire, "download") as download:
            acquire.acquire_slurp(audio=False)
        self.assertEqual(download.call_count, 4)
        for call in download.call_args_list:
            self.assertIn("/" + "a" * 40 + "/", call.args[0])
        pinned = acquire.source_ledger("slurp")
        self.assertEqual(pinned["revision"], "a" * 40)

    def test_changed_source_bytes_do_not_replace_the_pinned_ledger(self):
        self.prepare_metadata()
        (self.raw / "slurp/train.jsonl").write_bytes(b"changed")
        with patch.object(acquire, "download"), patch.object(acquire, "write_ledger") as write:
            with self.assertRaisesRegex(RuntimeError, "Pinned source checksum mismatch"):
                acquire.acquire_slurp(audio=False)
        write.assert_not_called()

    def test_librispeech_uses_recorded_url_and_checksums(self):
        entry = self.source("data/raw/librispeech/train-clean-100.tar.gz")
        entry.update(url="https://example.invalid/pinned.tar.gz", publisher_md5="pinned-publisher-md5")
        self.ledger("librispeech", {"files": [entry]})
        with patch.object(acquire, "request", side_effect=AssertionError("Current checksums requested")), \
             patch.object(acquire, "download") as download, patch.object(acquire, "extract"):
            acquire.acquire_libri()
        download.assert_called_once_with(entry["url"], self.root / entry["path"], entry["publisher_md5"])

    def test_mirror_uses_recorded_files_without_discovering_latest_revision(self):
        entry = self.source("data/raw/slurp/mirror/train.parquet")
        entry.update(url="https://example.invalid/resolve/pinned/train.parquet", lfs_sha256=entry["sha256"])
        self.ledger("slurp_audio_mirror", {"revision": "pinned", "files": [entry]})
        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = lambda path: types.SimpleNamespace(schema_arrow="fixture", iter_batches=lambda **kw: [])
        pyarrow.parquet = parquet
        with patch.dict("sys.modules", {"pyarrow": pyarrow, "pyarrow.parquet": parquet}), \
             patch.object(acquire, "acquire_slurp"), \
             patch.object(acquire, "get_json", side_effect=AssertionError("Latest mirror requested")), \
             patch.object(acquire, "download") as download, redirect_stdout(io.StringIO()):
            acquire.acquire_slurp_mirror()
        download.assert_called_once_with(entry["url"], self.root / entry["path"])
        self.assertEqual(acquire.source_ledger("slurp_audio_mirror")["revision"], "pinned")


if __name__ == "__main__":
    unittest.main()
