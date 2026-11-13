"""
integrations/claude_code/install.py - turn AMP memory on (or off) for a project
===============================================================================
    python -m integrations.claude_code.install DIR               add the hooks and the recall tool
    python -m integrations.claude_code.install DIR --uninstall   remove them again
    python -m integrations.claude_code.install DIR --mcp-url URL  memory server elsewhere

Writes two files in the project folder, merging with what is already there
(other hooks, servers and settings are kept):

    .claude/settings.local.json   the five AMP hooks, and "amp-memory" in
                                  enabledMcpjsonServers (so Claude Code does not
                                  ask before starting the recall tool). This file
                                  is personal: it holds paths of this computer.
    .mcp.json                     the "amp-memory" MCP server (recall tool)

AMP's entries are recognised by the path of hooks.py / recall_server.py, so
installing twice changes nothing and uninstalling removes only them. A file
left empty by uninstalling is deleted.

Claude Code reads hooks when a session starts: start a new session (or open
/hooks once) after installing.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import sys
import tempfile
from typing import Dict, List, Optional
from urllib.parse import urlparse

if __package__ in (None, ""):              # run as a file: make the repository importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.claude_code import project as proj

HOOKS_SCRIPT = os.path.join(proj.REPO, "integrations", "claude_code", "hooks.py")
RECALL_SCRIPT = os.path.join(proj.REPO, "integrations", "claude_code", "recall_server.py")
SERVER_NAME = "amp-memory"
#             Claude Code event   our event     timeout (s)
HOOKS = [("UserPromptSubmit", "prompt", 15),
         ("SessionStart", "start", 15),
         ("Stop", "stop", 10),
         ("PreCompact", "precompact", 10),
         ("SessionEnd", "sessionend", 10)]


def python_path() -> str:
    venv = os.path.join(proj.REPO, ".venv", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def hook_command(event: str, project: str, mcp_url: Optional[str] = None) -> str:
    command = " ".join(shlex.quote(part) for part in (python_path(), HOOKS_SCRIPT, event, "--project", project))
    if mcp_url and mcp_url != proj.DEFAULT_MCP_URL:
        command = f"{proj.MCP_URL_ENV}={shlex.quote(mcp_url)} {command}"
    return command


def _read(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()
    data = json.loads(text) if text else {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object; fix it by hand first.")
    return data


def _write(path: str, data: Dict) -> None:
    if not data:
        if os.path.exists(path):
            os.unlink(path)
        return
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".amp.", dir=folder)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _is_ours(entry: Dict) -> bool:
    return any(HOOKS_SCRIPT in str(hook.get("command", "")) for hook in entry.get("hooks", []) if isinstance(hook, dict))


def _without_ours(settings: Dict) -> Dict:
    hooks = settings.get("hooks")
    if isinstance(hooks, dict):
        for name in list(hooks):
            entries = [e for e in hooks[name] if not (isinstance(e, dict) and _is_ours(e))] \
                if isinstance(hooks[name], list) else hooks[name]
            if entries:
                hooks[name] = entries
            else:
                del hooks[name]
        if not hooks:
            del settings["hooks"]
    enabled = settings.get("enabledMcpjsonServers")
    if isinstance(enabled, list):
        enabled[:] = [name for name in enabled if name != SERVER_NAME]
        if not enabled:
            del settings["enabledMcpjsonServers"]
    return settings


def paths(project: str) -> Dict[str, str]:
    return {"settings": os.path.join(project, ".claude", "settings.local.json"),
            "mcp": os.path.join(project, ".mcp.json")}


def install(project: str, mcp_url: Optional[str] = None) -> Dict[str, str]:
    project = proj.project_dir(project)
    if not os.path.isdir(project):
        raise SystemExit(f"{project} is not a folder.")
    where = paths(project)
    settings = _without_ours(_read(where["settings"]))
    hooks = settings.setdefault("hooks", {})
    for name, event, timeout in HOOKS:
        hooks.setdefault(name, []).append(
            {"hooks": [{"type": "command", "command": hook_command(event, project, mcp_url), "timeout": timeout}]})
    settings.setdefault("enabledMcpjsonServers", []).append(SERVER_NAME)
    _write(where["settings"], settings)

    servers = _read(where["mcp"])
    server = {"type": "stdio", "command": python_path(), "args": [RECALL_SCRIPT, "--project", project]}
    if mcp_url and mcp_url != proj.DEFAULT_MCP_URL:
        server["env"] = {proj.MCP_URL_ENV: mcp_url}
    servers.setdefault("mcpServers", {})[SERVER_NAME] = server
    _write(where["mcp"], servers)
    return where


def uninstall(project: str) -> Dict[str, str]:
    project = proj.project_dir(project)
    where = paths(project)
    _write(where["settings"], _without_ours(_read(where["settings"])))
    servers = _read(where["mcp"])
    if isinstance(servers.get("mcpServers"), dict):
        servers["mcpServers"].pop(SERVER_NAME, None)
        if not servers["mcpServers"]:
            del servers["mcpServers"]
    _write(where["mcp"], servers)
    return where


def reachable(url: str, timeout: float = 1.0) -> bool:
    parts = urlparse(url)
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Turn AMP long-term memory on or off for a Claude Code project.")
    parser.add_argument("project", help="the project folder you open Claude Code in")
    parser.add_argument("--uninstall", action="store_true", help="remove AMP's hooks and recall tool")
    parser.add_argument("--mcp-url", default=None, help=f"memory server URL (default {proj.DEFAULT_MCP_URL})")
    args = parser.parse_args(argv)
    if args.uninstall:
        where = uninstall(args.project)
        print(f"AMP removed from {proj.project_dir(args.project)} ({where['settings']}, {where['mcp']}).")
        return 0
    where = install(args.project, args.mcp_url)
    url = args.mcp_url or proj.mcp_url()
    print(f"AMP memory is on for {proj.project_dir(args.project)}")
    print(f"  hooks:        {where['settings']}")
    print(f"  recall tool:  {where['mcp']}")
    print(f"  AMP session:  {proj.session_id(args.project)}")
    print(f"  memory server {url}: {'running' if reachable(url) else 'NOT running (python server/amp_memory_server.py)'}")
    print(f"  Ollama (RLM): {'running' if reachable('http://localhost:11434') else 'not running (text is summarized by rules)'}")
    print("Start a new Claude Code session in that folder (or open /hooks once) to load the hooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
