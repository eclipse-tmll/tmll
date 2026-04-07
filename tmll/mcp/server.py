#!/usr/bin/env python3
"""MCP server for TMLL CLI - exposes all CLI commands as MCP tools."""

import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tmll-cli-mcp-server")

CLI_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "cli.py")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080


def _server_is_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if the trace server is reachable."""
    try:
        urllib.request.urlopen(f"http://{host}:{port}/tsp/api/health", timeout=3)
        return True
    except Exception:
        return False


@mcp.tool()
def ensure_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Ensure the Trace Compass server is running. Downloads and installs it if not found, then starts it."""
    if _server_is_running(host, port):
        return f"Trace server already running at {host}:{port}"

    from tmll.services.tsp_installer import TSPInstaller
    installer = TSPInstaller()
    installer.install()

    import time
    for _ in range(15):
        time.sleep(2)
        if _server_is_running(host, port):
            return f"Trace server started at {host}:{port}"

    return "Trace server was launched but is not yet responding. It may need more time to start."


def run_cli(*args: str) -> str:
    """Run a tmll_cli.py command and return output."""
    result = subprocess.run(
        [sys.executable, CLI_PATH, "--log-stderr", *args],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or f"CLI exited with code {result.returncode}")
    return result.stdout.strip()


def build_args(flag_map: dict[str, tuple[str, any]]) -> list[str]:
    """Convert (flag, value) pairs to CLI flags."""
    args = []
    for flag, val in flag_map.values():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                args.append(flag)
        elif isinstance(val, list):
            args.extend([flag] + [str(v) for v in val])
        else:
            args.extend([flag, str(val)])
    return args


def _global_args(host: Optional[str], port: Optional[int]) -> list[str]:
    args = []
    if host:
        args.extend(["--host", host])
    if port:
        args.extend(["--port", str(port)])
    return args


@mcp.tool()
def create_experiment(traces: list[str], experiment_name: str, host: Optional[str] = None, port: Optional[int] = None) -> str:
    """Create a trace experiment from LTTng trace files or directories."""
    return run_cli(*_global_args(host, port), "create", *traces, "-n", experiment_name)


@mcp.tool()
def list_experiments() -> str:
    """List all open experiments."""
    return run_cli("list")


@mcp.tool()
def list_outputs(experiment_id: str, keywords: Optional[list[str]] = None) -> str:
    """List available outputs for an experiment."""
    args = build_args({"keywords": ("-k", keywords)})
    return run_cli("list-outputs", experiment_id, *args)


@mcp.tool()
def fetch_data(experiment_id: str, keywords: Optional[list[str]] = None, output_file: Optional[str] = None) -> str:
    """Fetch data from experiment outputs."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "output_file": ("-o", output_file)})
    return run_cli("fetch-data", experiment_id, *args)


@mcp.tool()
def delete_experiment(experiment_id: str) -> str:
    """Delete an experiment."""
    return run_cli("delete", experiment_id)


@mcp.tool()
def detect_anomalies(experiment_id: str, keywords: Optional[list[str]] = None, method: Optional[str] = None, resample_freq: Optional[str] = None) -> str:
    """Detect anomalies in trace data using ML methods (iforest, zscore, iqr, moving_average, seasonality, frequency_domain, combined)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "method": ("-m", method or "iforest"), "resample_freq": ("-H", resample_freq)})
    return run_cli("anomaly", experiment_id, *args)


@mcp.tool()
def detect_memory_leak(experiment_id: str, keywords: Optional[list[str]] = None) -> str:
    """Detect memory leaks in trace data."""
    args = build_args({"keywords": ("-k", keywords or ["memory"])})
    return run_cli("memory-leak", experiment_id, *args)


@mcp.tool()
def detect_changepoints(experiment_id: str, keywords: Optional[list[str]] = None, methods: Optional[list[str]] = None) -> str:
    """Detect change points in performance trends (single, zscore, voting, pca)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "methods": ("-m", methods or ["single", "zscore", "voting", "pca"])})
    return run_cli("changepoint", experiment_id, *args)


@mcp.tool()
def analyze_correlation(experiment_id: str, keywords: Optional[list[str]] = None, method: Optional[str] = None) -> str:
    """Analyze correlation between outputs for root cause analysis (pearson, kendall, spearman)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu", "memory"]), "method": ("-m", method or "pearson")})
    return run_cli("correlation", experiment_id, *args)


@mcp.tool()
def detect_idle_resources(experiment_id: str, keywords: Optional[list[str]] = None,
                          cpu_idle_threshold: Optional[float] = None,
                          memory_idle_threshold: Optional[float] = None,
                          disk_idle_threshold: Optional[float] = None) -> str:
    """Detect idle/underutilized resources."""
    args = build_args({
        "keywords": ("-k", keywords or ["cpu usage"]),
        "cpu": ("--cpu-idle-threshold", cpu_idle_threshold),
        "memory": ("--memory-idle-threshold", memory_idle_threshold),
        "disk": ("--disk-idle-threshold", disk_idle_threshold),
    })
    return run_cli("idle-resources", experiment_id, *args)


@mcp.tool()
def plan_capacity(experiment_id: str, keywords: Optional[list[str]] = None, horizon: Optional[int] = None) -> str:
    """Perform capacity planning with predictive models."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "horizon": ("-H", horizon or 100)})
    return run_cli("capacity", experiment_id, *args)


if __name__ == "__main__":
    mcp.run()
