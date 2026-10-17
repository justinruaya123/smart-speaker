"""Verify pinned archives and prepare a release without changing repo visibility.

The GitHub Actions workflow supplies its short-lived token. Verification uses only
the standard library and needs neither a token nor project dependencies.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path, asset, model_hash):
    if path.stat().st_size != asset["bytes"]:
        raise ValueError("Archive size mismatch: " + path.name)
    if sha256(path) != asset["sha256"]:
        raise ValueError("Archive SHA-256 mismatch: " + path.name)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members if not member.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive inventory entries")
        for member in members:
            name = member.filename.rstrip("/")
            parsed = PurePosixPath(name)
            if (not name or parsed.is_absolute() or ".." in parsed.parts
                    or "\\" in name or ":" in name
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive inventory entry: " + name)
        manifest = json.loads(archive.read("SUBMISSION-MANIFEST.json"))
        expected = manifest["files"]
        if set(names) != set(expected) | {"SUBMISSION-MANIFEST.json"}:
            raise ValueError("Archive inventory differs from its manifest")
        for name, expected_hash in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise ValueError("Archive member hash mismatch: " + name)
        if expected.get("submission/models/full/model.onnx") != model_hash:
            raise ValueError("Archive does not contain the selected 2M model")
        if manifest.get("actions_enabled") is not False:
            raise ValueError("Archive device actions must stay disabled")
        if manifest.get("purpose") == "inference-only":
            if manifest.get("parameters") != 2066831:
                raise ValueError("Archive model parameter count mismatch")
            if manifest.get("qualified") is not False:
                raise ValueError("Archive must retain its unqualified research status")
        elif (manifest.get("purpose") != "silent-benchmark-2m"
              or manifest.get("model_choice") != "full"):
            raise ValueError("Unexpected release archive purpose or model choice")
    return len(names)


def verified_assets(receipt, directory):
    directory.mkdir(parents=True, exist_ok=True)
    assets = receipt["assets"]
    if {asset["kind"] for asset in assets} != {"inference", "benchmark"} or len(assets) != 2:
        raise ValueError("Exactly the inference and benchmark archives are required")
    for asset in assets:
        name = asset["filename"]
        if name != Path(name).name or not name.endswith(".zip"):
            raise ValueError("Unsafe release filename")
        path = directory / name
        if not path.exists():
            revision = receipt["revision"]
            prefix = ("https://huggingface.co/datasets/" + receipt["repository"]
                      + "/resolve/" + revision + "/releases/")
            if not re.fullmatch(r"[0-9a-f]{40}", revision) or not asset["url"].startswith(prefix):
                raise ValueError("Release download is not pinned to the recorded source")
            temporary = path.with_suffix(".zip.partial")
            with urlopen(asset["url"], timeout=180) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            verify_archive(temporary, asset, receipt["model_sha256"])
            temporary.replace(path)
        verify_archive(path, asset, receipt["model_sha256"])
    return assets


def github_request(route, token, method="GET", payload=None, raw=None):
    url = route if route.startswith("https://") else "https://api.github.com" + route
    if not url.startswith(("https://api.github.com/", "https://uploads.github.com/")):
        raise ValueError("Unexpected GitHub API origin")
    headers = {"Accept": "application/vnd.github+json", "Authorization": "Bearer " + token,
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "smart-speaker-draft-release"}
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if raw is not None:
        headers["Content-Type"] = "application/zip"
    try:
        with urlopen(Request(url, data=data, headers=headers, method=method), timeout=180) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            message = json.loads(error.read().decode("utf-8"))["message"]
            error.msg = str(error.reason) + ": " + str(message)
        except (ValueError, KeyError, UnicodeError):
            pass
        raise


def verify_tag_target(args, token, allow_missing):
    route = "/repos/" + args.repository
    try:
        reference = github_request(route + "/git/ref/tags/" + quote(args.tag, safe=""), token)
    except HTTPError as error:
        if error.code == 404 and allow_missing:
            return None
        raise
    obj = reference["object"]
    seen = set()
    while obj.get("type") == "tag":
        if obj["sha"] in seen or len(seen) >= 16:
            raise ValueError("Cannot resolve release tag")
        seen.add(obj["sha"])
        obj = github_request(route + "/git/tags/" + obj["sha"], token)["object"]
    if obj.get("type") != "commit" or obj.get("sha") != args.target:
        raise ValueError("Release tag points at a different source commit")
    return obj["sha"]


def verified_release_assets(release, assets):
    expected = {asset["filename"]: asset for asset in assets}
    actual = {asset["name"]: asset for asset in release["assets"]}
    if set(actual) != set(expected):
        raise ValueError("Release asset inventory differs from the selected archives")
    rows = []
    for name, asset in expected.items():
        if (actual[name].get("size") != asset["bytes"]
                or actual[name].get("digest") != "sha256:" + asset["sha256"]):
            raise ValueError("Release asset hash or size mismatch: " + name)
        rows.append({"filename": name, "bytes": asset["bytes"], "sha256": asset["sha256"],
                     "url": actual[name]["browser_download_url"]})
    return rows


def prepare_draft(args, assets, token):
    route = "/repos/" + args.repository
    if not re.fullmatch(r"[0-9a-f]{40}", args.target or ""):
        raise ValueError("A full, immutable source target commit is required")
    tag_target = verify_tag_target(args, token, allow_missing=True)
    releases = github_request(route + "/releases?per_page=100", token)
    existing = next((release for release in releases if release["tag_name"] == args.tag), None)
    if existing is not None:
        if existing.get("draft") is not True:
            if tag_target is None:
                raise ValueError("Published release tag is missing")
            uploaded = verified_release_assets(existing, assets)
            return {"status": "published", "draft": False, "release_id": existing["id"],
                    "url": existing["html_url"], "target_commit": tag_target,
                    "tag_commit": tag_target, "assets": uploaded}
        if existing.get("target_commitish") != args.target:
            raise ValueError("Existing draft points at a different source commit")
        release = existing
    else:
        release = github_request(route + "/releases", token, "POST", {
            "tag_name": args.tag, "target_commitish": args.target,
            "name": "Smart Speaker — selected 2M submission",
            "body": (ROOT / "docs/RELEASE-final-submission-v1.md").read_text(encoding="utf-8"),
            "draft": True, "prerelease": False, "make_latest": "false"})
    uploaded = []
    for asset in assets:
        found = next((item for item in release["assets"] if item["name"] == asset["filename"]), None)
        if found is None:
            url = release["upload_url"].split("{", 1)[0] + "?name=" + quote(asset["filename"])
            found = github_request(url, token, "POST", raw=(args.download_dir / asset["filename"]).read_bytes())
        if found.get("size") != asset["bytes"] or found.get("digest") != "sha256:" + asset["sha256"]:
            raise ValueError("Uploaded GitHub asset hash or size mismatch: " + asset["filename"])
        uploaded.append({"filename": asset["filename"], "bytes": asset["bytes"],
                         "sha256": asset["sha256"], "url": found["browser_download_url"]})
    return {"status": "draft_ready", "draft": True, "release_id": release["id"],
            "url": release["html_url"], "target_commit": args.target, "assets": uploaded}


def publish_verified_draft(args, status, token):
    route = "/repos/" + args.repository + "/releases/" + str(status["release_id"])
    draft = github_request(route, token)
    if draft.get("draft") is not True or draft.get("target_commitish") != args.target:
        raise ValueError("Draft target or publication state changed before publication")
    verified_release_assets(draft, status["assets"])
    verify_tag_target(args, token, allow_missing=True)
    published = github_request(route, token, "PATCH", {"draft": False, "make_latest": "false"})
    if published.get("draft") is not False:
        raise ValueError("GitHub did not confirm release publication")
    status.update(status="published", draft=False, url=published["html_url"])
    status["tag_commit"] = verify_tag_target(args, token, allow_missing=False)
    return status


def publish_status(args, status, token):
    if args.status_branch != "codex/final-submission":
        raise ValueError("Status receipts are limited to codex/final-submission")
    path = "/repos/" + args.repository + "/contents/submission/release/github-status.json"
    payload = {"message": "Record private draft release result", "branch": args.status_branch,
               "content": base64.b64encode((json.dumps(status, indent=2) + "\n").encode()).decode()}
    try:
        payload["sha"] = github_request(path + "?ref=" + quote(args.status_branch, safe=""), token)["sha"]
    except HTTPError as error:
        if error.code != 404:
            raise
    github_request(path, token, "PUT", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-file", type=Path, default=ROOT / "submission/release/downloads.json")
    parser.add_argument("--download-dir", type=Path, default=ROOT / ".cache/submission/release-downloads")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--publish", action="store_true",
                        help="Publish only after verifying this newly prepared draft and its assets")
    parser.add_argument("--repository", default="justinruaya123/smart-speaker")
    parser.add_argument("--target")
    parser.add_argument("--tag", default="2m-submission-v1")
    parser.add_argument("--status-branch")
    args = parser.parse_args(argv)
    status = None
    token = os.environ.get("GITHUB_TOKEN", "")
    try:
        receipt = json.loads(args.downloads_file.read_text(encoding="utf-8"))
        assets = verified_assets(receipt, args.download_dir)
        if args.verify_only:
            print(json.dumps({"verified_assets": len(assets), "assets": assets}))
            return 0
        if not args.draft or not token:
            raise ValueError("Draft creation requires --draft and the workflow GITHUB_TOKEN")
        status = prepare_draft(args, assets, token)
        if args.publish and status["draft"]:
            status = publish_verified_draft(args, status, token)
        status["workflow_url"] = (os.environ.get("GITHUB_SERVER_URL", "https://github.com")
                                  + "/" + args.repository + "/actions/runs/"
                                  + os.environ.get("GITHUB_RUN_ID", ""))
    except Exception as error:
        print(str(error), file=sys.stderr)
        if args.status_branch and token:
            failure = {"status": "failed", "target_commit": args.target, "error": str(error)}
            if status is not None:
                failure = {**status, "error": str(error),
                           "status": "published_verification_failed" if status.get("draft") is False
                           else "draft_preparation_failed"}
            try:
                publish_status(args, failure, token)
            except Exception as status_error:
                print("Status receipt also failed: " + str(status_error), file=sys.stderr)
        return 1
    if args.status_branch:
        for attempt in range(2):
            try:
                publish_status(args, status, token)
                break
            except Exception as error:
                status["receipt_error"] = str(error)
                print("Release state confirmed; status receipt failed: " + str(error), file=sys.stderr)
                if attempt:
                    print(json.dumps(status, indent=2))
                    return 1
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
