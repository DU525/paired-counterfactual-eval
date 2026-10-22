#!/usr/bin/env python3
"""Repository hygiene scanner: secrets, large files, and forbidden paths.

Run before every `git add -A` and before publishing a repository.

    python scan_repo.py <repo_root> [--max-file-mb 20] [--json out.json]

Exit codes
    0  clean
    1  findings that must be fixed
    2  usage / IO error

Why this file exists
--------------------
Naive secret regexes produce an unusable amount of false positives: any
`api_key=some_variable` in source, any `${SECRET_KEY:-dev-default}` in a compose
file, and any documented key *format* in a README all look like leaks. A scanner
that cries wolf gets ignored, which is worse than no scanner.

So every candidate match is classified by the *shape of the value*:
  * identifier-like  -> variable reference, not a secret
  * expression-like  -> function call / subscript / interpolation, not a secret
  * placeholder-like -> template text, not a secret
  * high-entropy     -> a real secret
Only the last category is reported. Report output never contains the secret
itself, only a stable sha256 fingerprint, so a report can be pasted into a PR.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --- high-confidence token formats (a hit is a secret unless templated) ----

TOKEN_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("github_oauth", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("openai_key", re.compile(r"\bsk-(?!your\b|proj-|ant-)[A-Za-z0-9_\-]{24,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{24,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "aws_secret_key",
        re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?"),
    ),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}")),
    (
        "bearer_token",
        re.compile(r"(?i)\bauthorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._\-]{20,}"),
    ),
    (
        "connection_uri_creds",
        re.compile(
            r"(?i)\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s:@/]+@"
        ),
    ),
]

# Assignment-style rules. The captured group is the VALUE; it is then classified.
ASSIGN_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|refresh[_\-]?token|"
            r"app[_\-]?secret|app[_\-]?id|webhook[_\-]?url|passwd|password|passphrase|"
            r"private[_\-]?key|client[_\-]?secret)\b\s*[:=]\s*[\"']?([^\s\"'#,}]{8,})[\"']?"
        ),
    ),
    (
        "feishu_secret",
        re.compile(r"(?i)\bFEISHU_APP_SECRET\s*[:=]\s*[\"']?([A-Za-z0-9]{16,})[\"']?"),
    ),
]

# Keys whose *name* implies the value is a secret even without a known format.
SECRET_NAME_RE = re.compile(
    r"(?i)\b(api[_\-]?key|secret|token|password|passwd|passphrase|credential|private[_\-]?key)\b"
)

FORBIDDEN_NAMES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    "credentials.json",
    "credentials.yaml",
    "secrets.json",
    "secrets.yaml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_train",
    "id_train.pub",
    "zcode-network-ca.key",
    ".netrc",
    ".npmrc",
    "service-account.json",
    "known_hosts",
    "auth.json",
    "token.json",
]
FORBIDDEN_GLOBS = [
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.ckpt",
    "*.safetensors",
    "*.pth",
    "*.pt",
    "*.onnx",
    "*.engine",
    "*.npz",
    "*.npy",
    "*.h5",
    "*.hdf5",
    "*.parquet",
    "*.arrow",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
    "*.7z",
    "*.rar",
    ".env.*",
    "*.log",
    "*.pid",
    "*.secret",
]
FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "wandb",
    "outputs",
    "artifacts",
    "results",
    "runs",
    "logs",
    "checkpoints",
    "data",
    "datasets",
    "weights",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".vue",
    ".html",
    ".css",
    ".scss",
    ".xml",
    ".sql",
    ".rst",
    ".ipynb",
    ".lock",
    ".example",
    ".env",
    ".in",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".c",
    ".h",
    ".cpp",
    ".rs",
    ".go",
    ".java",
    ".m",
    ".mm",
}
BINARY_SKIP = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".mp4",
    ".mp3",
    ".wav",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".onnx",
    ".safetensors",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".h5",
    ".parquet",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
}
TEXT_FILE_SCAN_MAX_BYTES = 4_000_000

PLACEHOLDER_WORDS = (
    "your",
    "example",
    "placeholder",
    "changeme",
    "change-me",
    "change_me",
    "dummy",
    "fake",
    "sample",
    "demo",
    "test",
    "todo",
    "none",
    "null",
    "xxx",
    "abc123",
    "secret",
    "redacted",
    "masked",
    "<",
    ">",
    "...",
)
PLACEHOLDER_RE = re.compile(
    r"(?i)(your|example|placeholder|changeme|dummy|fake|sample|demo|xxxx|redacted|\.\.\.)"
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ENV_REF_RE = re.compile(r"^(\$\{|\$[A-Za-z_]|<[A-Za-z_]|\{\{)")
EXPRESSION_RE = re.compile(
    r"(\(|\)|\[|\]|\{|\}|=>|::|\.get\(|\.environ|os\.getenv|process\.env|config\.)"
)


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:12]


def classify_value(value: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict in {secret, placeholder, identifier,
    expression, env_ref, low_entropy}."""
    v = value.strip().strip("\"'").rstrip(",;")
    if not v:
        return "placeholder", "empty value"
    low = v.lower()
    for word in PLACEHOLDER_WORDS:
        if word in low:
            return "placeholder", f"contains placeholder word {word!r}"
    if ENV_REF_RE.match(v):
        return "env_ref", "environment-variable interpolation"
    if IDENTIFIER_RE.match(v):
        return "identifier", "bare identifier (variable reference)"
    if EXPRESSION_RE.search(v):
        return "expression", "expression / function call, not a literal"
    if low.startswith(("http://", "https://")) and "@" not in v:
        return "placeholder", "plain URL"
    if v.startswith("<") or v.startswith("["):
        return "placeholder", "bracketed template"
    letters = sum(c.isalpha() for c in v)
    digits = sum(c.isdigit() for c in v)
    if len(v) < 12:
        return "low_entropy", "shorter than 12 characters"
    if letters and digits and len(set(v)) >= 10:
        return "secret", "high-entropy mixed alphanumeric literal"
    if len(set(v)) <= 4:
        return "low_entropy", "very low character diversity"
    return "low_entropy", "no high-entropy characteristics"


def _is_example_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".example") or name.endswith(".example.txt") or ".example." in name


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in FORBIDDEN_DIRS and not d.startswith(".git")]
        for name in filenames:
            yield Path(dirpath) / name


def _scan_private_key_block(text: str, rel: str, findings: list[dict]) -> None:
    """A private-key header alone in prose is documentation; a header followed
    by real base64 payload is a leak."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "PRIVATE KEY-----" not in line:
            continue
        payload = 0
        for nxt in lines[i + 1 : i + 8]:
            s = nxt.strip()
            if not s or s.startswith("...") or "-----" in s:
                continue
            if re.fullmatch(r"[A-Za-z0-9+/=]{32,}", s):
                payload += 1
        if payload:
            findings.append(
                {
                    "kind": "secret",
                    "path": rel,
                    "line": i + 1,
                    "rule": "private_key_block",
                    "fingerprint": _fingerprint(lines[i].strip()),
                    "detail": "PRIVATE KEY header followed by base64 payload",
                }
            )


def scan(root: Path, max_file_mb: float) -> dict:
    findings: list[dict] = []
    scanned = 0
    max_bytes = int(max_file_mb * 1024 * 1024)

    for path in iter_files(root):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        name = path.name
        example_file = _is_example_file(path)

        # 1) forbidden names
        is_forbidden = name in FORBIDDEN_NAMES or any(
            fnmatch.fnmatch(name, g) for g in FORBIDDEN_GLOBS
        )
        if is_forbidden and not example_file and name != ".env.example":
            findings.append(
                {
                    "kind": "forbidden_path",
                    "path": rel,
                    "detail": f"file name matches a forbidden pattern ({name})",
                }
            )

        # 2) size
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            findings.append(
                {
                    "kind": "oversized_file",
                    "path": rel,
                    "detail": f"{size / 1024 / 1024:.1f} MB exceeds limit {max_file_mb} MB",
                }
            )

        # 3) secrets
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            or name.startswith(".")
            or name in {"Dockerfile", "Makefile", "LICENSE", "README"}
        ):
            try:
                if size > TEXT_FILE_SCAN_MAX_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            _scan_private_key_block(text, rel, findings)

            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") and not TOKEN_RULES[0][1].search(stripped):
                    # comment: only unambiguous token formats still count
                    pass
                for rule_name, rx in TOKEN_RULES:
                    for m in rx.finditer(line):
                        value = m.group(1) if m.groups() else m.group(0)
                        if PLACEHOLDER_RE.search(value):
                            continue
                        findings.append(
                            {
                                "kind": "secret",
                                "path": rel,
                                "line": lineno,
                                "rule": rule_name,
                                "fingerprint": _fingerprint(value),
                                "detail": f"{rule_name}: value matches a known credential format",
                            }
                        )
                        break

                for rule_name, rx in ASSIGN_RULES:
                    for m in rx.finditer(line):
                        value = m.group(1)
                        verdict, reason = classify_value(value)
                        if verdict != "secret":
                            continue
                        if example_file and not TOKEN_RULES:
                            continue
                        findings.append(
                            {
                                "kind": "secret",
                                "path": rel,
                                "line": lineno,
                                "rule": rule_name,
                                "fingerprint": _fingerprint(value),
                                "detail": f"{rule_name}: {reason}",
                            }
                        )
                        break

    # 4) tracked files that match forbidden patterns
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "ls-files"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if _is_example_file(p):
                    continue
                if p.name in FORBIDDEN_NAMES or any(
                    fnmatch.fnmatch(p.name, g) for g in FORBIDDEN_GLOBS
                ):
                    findings.append(
                        {
                            "kind": "tracked_forbidden",
                            "path": line,
                            "detail": "file is tracked by git AND matches a forbidden pattern",
                        }
                    )
        except (OSError, subprocess.SubprocessError):
            pass

    counts: dict[str, int] = {}
    for f in findings:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    return {
        "root": str(root),
        "text_files_scanned": scanned,
        "finding_count": len(findings),
        "counts_by_kind": counts,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository hygiene scanner")
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-file-mb", type=float, default=20.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    report = scan(root, args.max_file_mb)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not report["findings"]:
        print(f"CLEAN  {root}  (scanned {report['text_files_scanned']} text files)")
        return 0

    print(f"FOUND {len(report['findings'])} issue(s) in {root}")
    for f in report["findings"]:
        loc = f["path"] + (f":{f['line']}" if "line" in f else "")
        extra = (
            f"  [{f.get('rule', '')} {f.get('fingerprint', '')}]" if f["kind"] == "secret" else ""
        )
        print(f"  {f['kind']:<18} {loc}\n      {f['detail']}{extra}")
    print("\nSecret values are never printed; only a sha256 fingerprint is shown.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
