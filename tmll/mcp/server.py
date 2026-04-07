#!/usr/bin/env python3
"""MCP server for TMLL CLI - exposes all CLI commands as MCP tools."""

import base64
import contextlib
import functools
import io
import json
import subprocess
import sys
import traceback as _tb
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

mcp = FastMCP("tmll-cli-mcp-server")

CLI_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "cli.py")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Write debug message to stderr (safe for MCP stdio transport)."""
    print(f"[tmll-mcp-debug] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _protect_stdout():
    """Temporarily redirect stdout→stderr so stray print() cannot corrupt the MCP stdio transport."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _safe_tool(fn):
    """Decorator applied to every tool: protects stdout, logs entry/exit/errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        name = fn.__name__
        _log(f">>> TOOL CALL  {name}  args={args!r}  kwargs={kwargs!r}")
        with _protect_stdout():
            try:
                result = fn(*args, **kwargs)
                preview = repr(result)[:300]
                _log(f"<<< TOOL OK    {name}  result_preview={preview}")
                return result
            except Exception as exc:
                tb = _tb.format_exc()
                _log(f"!!! TOOL ERROR {name}  {type(exc).__name__}: {exc}\n{tb}")
                raise
    return wrapper


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------

def _server_is_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if the trace server is reachable."""
    url = f"http://{host}:{port}/tsp/api/health"
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception as exc:
        _log(f"Server health check failed ({url}): {exc}")
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


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run_cli(*args: str) -> str:
    """Run a tmll_cli.py command and return output."""
    cmd = [sys.executable, CLI_PATH, "--log-stderr", *args]
    _log(f"run_cli: executing {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        msg = (
            f"CLI timed out after 120s\n"
            f"  command: {' '.join(cmd)}\n"
            f"  partial stdout: {exc.stdout!r}\n"
            f"  partial stderr: {exc.stderr!r}"
        )
        _log(f"run_cli TIMEOUT: {msg}")
        raise RuntimeError(msg)
    except Exception as exc:
        msg = (
            f"Failed to launch CLI: {type(exc).__name__}: {exc}\n"
            f"  command: {' '.join(cmd)}"
        )
        _log(f"run_cli LAUNCH ERROR: {msg}")
        raise RuntimeError(msg)

    _log(f"run_cli: exit_code={result.returncode} stdout_len={len(result.stdout)} stderr_len={len(result.stderr)}")
    if result.stderr.strip():
        _log(f"run_cli stderr:\n{result.stderr.strip()}")

    if result.returncode != 0:
        msg = (
            f"CLI exited with code {result.returncode}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
        _log(f"run_cli FAILED: {msg}")
        raise RuntimeError(msg)
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe_tool
def create_experiment(traces: list[str], experiment_name: str, host: Optional[str] = None, port: Optional[int] = None) -> str:
    """Create a trace experiment from LTTng trace files or directories."""
    return run_cli(*_global_args(host, port), "create", *traces, "-n", experiment_name)


@mcp.tool()
@_safe_tool
def list_experiments() -> str:
    """List all open experiments."""
    return run_cli("list")


@mcp.tool()
@_safe_tool
def list_outputs(experiment_id: str, keywords: Optional[list[str]] = None) -> str:
    """List available outputs for an experiment."""
    args = build_args({"keywords": ("-k", keywords)})
    return run_cli("list-outputs", experiment_id, *args)


@mcp.tool()
@_safe_tool
def fetch_data(experiment_id: str, keywords: Optional[list[str]] = None, output_file: Optional[str] = None) -> str:
    """Fetch data from experiment outputs."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "output_file": ("-o", output_file)})
    return run_cli("fetch-data", experiment_id, *args)


@mcp.tool()
@_safe_tool
def delete_experiment(experiment_id: str) -> str:
    """Delete an experiment."""
    return run_cli("delete", experiment_id)


@mcp.tool()
def detect_anomalies(experiment_id: str, keywords: Optional[list[str]] = None, method: Optional[str] = None, resample_freq: Optional[str] = None) -> str:
    """Detect anomalies in trace data using ML methods (iforest, zscore, iqr, moving_average, seasonality, frequency_domain, combined)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "method": ("-m", method or "iforest"), "resample_freq": ("-H", resample_freq)})
    return run_cli("anomaly", experiment_id, *args)


@mcp.tool()
@_safe_tool
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
@_safe_tool
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
@_safe_tool
def plan_capacity(experiment_id: str, keywords: Optional[list[str]] = None, horizon: Optional[int] = None) -> str:
    """Perform capacity planning with predictive models."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "horizon": ("-H", horizon or 100)})
    return run_cli("capacity", experiment_id, *args)


@mcp.tool()
@_safe_tool
def plot_xy_with_anomalies(
    experiment_id: str,
    keywords: Optional[list[str]] = None,
    method: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    resample_freq: Optional[str] = None,
) -> list[TextContent | ImageContent]:
    """Fetch XY data from an experiment, run anomaly detection, and return an annotated plot image with a text summary."""
    from tmll.tmll_client import TMLLClient
    from tmll.common.models.experiment import Experiment
    from tmll.ml.modules.anomaly_detection.anomaly_detection_module import AnomalyDetection

    h = host or DEFAULT_HOST
    p = port or DEFAULT_PORT
    keywords = keywords or ["cpu usage"]
    method = method or "iforest"

    client = TMLLClient(h, p)

    resp = client.tsp_client.fetch_experiment(experiment_id)
    if resp.status_code != 200:
        return [TextContent(type="text", text=f"Experiment {experiment_id} not found (status={resp.status_code}).")]
    experiment = Experiment.from_tsp_experiment(resp.model)
    experiment.assign_outputs(client._fetch_outputs(experiment))

    outputs = experiment.find_outputs(keyword=keywords, type=["xy"])
    if not outputs:
        return [TextContent(type="text", text="No XY outputs found matching keywords.")]

    ad_kwargs = {}
    if resample_freq:
        ad_kwargs["resample_freq"] = resample_freq
    ad = AnomalyDetection(client, experiment, outputs, **ad_kwargs)
    result = ad.find_anomalies(method=method)
    if not result or not result.anomalies:
        return [TextContent(type="text", text="Anomaly detection returned no results.")]

    colors = plt.colormaps.get_cmap("tab10")
    contents: list[TextContent | ImageContent] = []
    total_anomalies = 0

    for idx, (name, dataframe) in enumerate(ad.dataframes.items()):
        anomaly_df = result.anomalies.get(name, pd.DataFrame())
        periods = result.anomaly_periods.get(name, [])

        fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
        ax.plot(dataframe.index, dataframe.iloc[:, 0], color=colors(idx), linewidth=1.2, label=name)

        for i, (start, end) in enumerate(periods):
            ax.axvspan(start, end, color="red", alpha=0.2, label="Anomaly Period" if i == 0 else None)

        if not anomaly_df.empty:
            is_anomaly_cols = anomaly_df.filter(regex="_is_anomaly$")
            if not is_anomaly_cols.empty:
                is_anomaly = is_anomaly_cols.any(axis=1)
            else:
                is_anomaly = anomaly_df.any(axis=1)
            n_anomaly_points = int(is_anomaly.sum())
            total_anomalies += n_anomaly_points

            # Scatter points not already inside a shaded period
            for point in anomaly_df[is_anomaly].index:
                if any(s <= point <= e for s, e in periods):
                    continue
                if point in dataframe.index:
                    ax.scatter(point, dataframe.loc[point].values[0], color="red", s=40, zorder=5)

        ax.set_title(f"Anomaly Detection: {name} ({method})")
        ax.set_xlabel("Time")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        contents.append(ImageContent(type="image", data=base64.b64encode(buf.read()).decode(), mimeType="image/png"))

    period_summary = []
    for name, periods in result.anomaly_periods.items():
        for start, end in periods:
            period_summary.append(f"  {name}: {start} → {end}")

    summary = f"Found {total_anomalies} anomalies across {len(ad.dataframes)} outputs using '{method}'."
    if period_summary:
        summary += "\n\nAnomaly periods:\n" + "\n".join(period_summary)

    contents.insert(0, TextContent(type="text", text=summary))
    return contents



if __name__ == "__main__":
    _log(f"MCP server starting — CLI_PATH={CLI_PATH} python={sys.executable}")
    mcp.run()
