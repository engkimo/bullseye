#!/usr/bin/env python3
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PM_DIR = os.path.join(ROOT, "tools", "pm")


def run(cmd: List[str], input_str: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input_str else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(input_str)
    return p.returncode, out.strip(), err.strip()


def ensure_pm_files() -> Dict[str, str]:
    files = {
        "labels": os.path.join(PM_DIR, "labels.json"),
        "milestones": os.path.join(PM_DIR, "milestones.json"),
        "issues": os.path.join(PM_DIR, "issues.json"),
    }
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"Missing config files: {missing}. Expected under {PM_DIR}", file=sys.stderr)
        sys.exit(2)
    return files


def get_owner_repo() -> Tuple[str, str]:
    # Try gh first
    code, out, _ = run(["gh", "repo", "view", "--json", "name,owner"])
    if code == 0:
        data = json.loads(out)
        owner = data["owner"]["login"] if isinstance(data.get("owner"), dict) else data.get("owner")
        name = data["name"]
        if owner and name:
            return owner, name
    # Fallback to parsing git remote
    code, out, _ = run(["git", "config", "--get", "remote.origin.url"])
    if code != 0:
        print("Unable to determine GitHub repo. Run inside a cloned GitHub repo.", file=sys.stderr)
        sys.exit(2)
    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)", out)
    if not m:
        print(f"Remote origin not recognized as GitHub: {out}", file=sys.stderr)
        sys.exit(2)
    return m.group("owner"), m.group("repo")


def gh_api(method: str, path: str, fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = ["gh", "api", "-X", method.upper(), path]
    if fields:
        # Use -f to avoid needing JSON encoding everywhere
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                cmd += ["-f", f"{k}={v}"]
            else:
                cmd += ["-f", f"{k}={v}"]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"gh api failed: {' '.join(shlex.quote(x) for x in cmd)}\n{err}")
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        return {}


def fetch_labels(owner: str, repo: str) -> Dict[str, Dict[str, Any]]:
    items: List[Dict[str, Any]] = gh_api("GET", f"/repos/{owner}/{repo}/labels?per_page=100")
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            result[it["name"]] = it
    return result


def sync_labels(owner: str, repo: str, cfg: Dict[str, Any]) -> None:
    existing = fetch_labels(owner, repo)
    labels = cfg.get("labels", [])
    print(f"Syncing {len(labels)} labels...")
    for lab in labels:
        name = lab["name"]
        color = lab.get("color", "ededed").lstrip("#")
        desc = lab.get("description", "")
        if name in existing:
            gh_api("PATCH", f"/repos/{owner}/{repo}/labels/{name}", {"new_name": name, "color": color, "description": desc})
            print(f"  updated: {name}")
        else:
            gh_api("POST", f"/repos/{owner}/{repo}/labels", {"name": name, "color": color, "description": desc})
            print(f"  created: {name}")


def fetch_milestones(owner: str, repo: str) -> Dict[str, Dict[str, Any]]:
    items: List[Dict[str, Any]] = gh_api("GET", f"/repos/{owner}/{repo}/milestones?state=all&per_page=100")
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            result[it["title"]] = it
    return result


def sync_milestones(owner: str, repo: str, cfg: Dict[str, Any]) -> None:
    existing = fetch_milestones(owner, repo)
    mls = cfg.get("milestones", [])
    print(f"Syncing {len(mls)} milestones...")
    for m in mls:
        title = m["title"]
        fields = {
            "title": title,
            "state": m.get("state", "open"),
            "description": m.get("description", ""),
        }
        if m.get("due_on"):
            fields["due_on"] = m["due_on"]
        if title in existing:
            gh_api("PATCH", f"/repos/{owner}/{repo}/milestones/{existing[title]['number']}", fields)
            print(f"  updated: {title}")
        else:
            gh_api("POST", f"/repos/{owner}/{repo}/milestones", fields)
            print(f"  created: {title}")


def list_issue_titles() -> List[str]:
    code, out, _ = run(["gh", "issue", "list", "--limit", "500", "--state", "all", "--json", "title"])
    if code != 0:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    return [d.get("title", "") for d in data]


def sync_issues(owner: str, repo: str, cfg: Dict[str, Any]) -> None:
    existing_titles = set(list_issue_titles())
    mls = fetch_milestones(owner, repo)
    want = cfg.get("issues", [])
    print(f"Syncing {len(want)} issues...")
    created = 0
    for it in want:
        title = it["title"]
        if title in existing_titles:
            print(f"  exists: {title}")
            continue
        body = it.get("body", "")
        labels = it.get("labels", [])
        milestone_title = it.get("milestone")
        cmd = ["gh", "issue", "create", "-t", title, "-b", body]
        for lab in labels:
            cmd += ["-l", lab]
        if milestone_title:
            cmd += ["-m", milestone_title]
        code, _, err = run(cmd)
        if code != 0:
            print(f"  failed: {title}: {err}")
        else:
            created += 1
            print(f"  created: {title}")
    print(f"Done. Created {created} issues.")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: ga_sync.py [plan|sync] [...]", file=sys.stderr)
        sys.exit(2)
    action = sys.argv[1]
    files = ensure_pm_files()
    owner, repo = get_owner_repo()
    labels_cfg = load_json(files["labels"])
    ms_cfg = load_json(files["milestones"])
    issues_cfg = load_json(files["issues"])

    if action == "plan":
        print("Plan summary:")
        print(f"  Repo: {owner}/{repo}")
        print(f"  Labels: {len(labels_cfg.get('labels', []))}")
        print(f"  Milestones: {len(ms_cfg.get('milestones', []))}")
        print(f"  Issues: {len(issues_cfg.get('issues', []))}")
        print("  Use: ga sync all  # to create/update")
        return

    if action == "sync":
        what = sys.argv[2] if len(sys.argv) > 2 else "all"
        if what in ("labels", "all"):
            sync_labels(owner, repo, labels_cfg)
        if what in ("milestones", "all"):
            sync_milestones(owner, repo, ms_cfg)
        if what in ("issues", "all"):
            sync_issues(owner, repo, issues_cfg)
        return

    print(f"Unknown action: {action}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

