#!/usr/bin/env python3
"""MCP server for TMLL CLI - exposes all CLI commands as MCP tools."""

import contextlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import matplotlib.patches as mpatches

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations
from tmll.mcp.ui.gantt import build_gantt_figure


# Prevent stray print() calls from corrupting the MCP stdio JSON transport.
# MCP's stdio_server reads sys.stdout.buffer, so we must preserve it.
# We replace sys.stdout with a wrapper that redirects text writes to stderr
# but keeps the original .buffer attribute for MCP's binary I/O.


class _StderrWithBuffer:
    """Text writes go to stderr; .buffer stays as real stdout for MCP."""

    def __init__(self):
        self.buffer = sys.__stdout__.buffer

    def write(self, s):
        return sys.__stderr__.write(s)

    def flush(self):
        sys.__stderr__.flush()

    def __getattr__(self, name):
        return getattr(sys.__stderr__, name)


sys.stdout = _StderrWithBuffer()

mcp = FastMCP("tmll-cli-mcp-server")

_last_xy_payload: dict = {}

CLI_PATH = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent / "cli.py")
UI_DIR = Path(__file__).resolve().parent / "ui"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080

# MCP Apps spec: HTML UI resource that hosts render in a sandboxed iframe.
XY_UI_URI = "ui://tmll/xy-anomalies.html"
MCP_APP_MIME = "text/html;profile=mcp-app"


@contextlib.contextmanager
def _protect_stdout():
    """Redirect stdout→stderr so stray prints can't corrupt the MCP stdio transport."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _server_is_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if the trace server is reachable."""
    try:
        urllib.request.urlopen(
            f"http://{host}:{port}/tsp/api/health", timeout=3)
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
        raise RuntimeError(
            result.stderr or f"CLI exited with code {result.returncode}")
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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def list_experiments() -> str:
    """List all open experiments. Returns name-UUID pairs.

    If a user refers to an experiment by name, call this tool first to resolve the name to its UUID.
    All other tools require the experiment UUID, not the name.
    """
    return run_cli("list")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def list_outputs(experiment_id: str, keywords: Optional[list[str]] = None) -> str:
    """List available outputs for an experiment."""
    args = build_args({"keywords": ("-k", keywords)})
    return run_cli("list-outputs", experiment_id, *args)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def fetch_data(experiment_id: str, keywords: Optional[list[str]] = None, output_file: Optional[str] = None) -> str:
    """Fetch data from experiment outputs."""
    args = build_args({"keywords": (
        "-k", keywords or ["cpu usage"]), "output_file": ("-o", output_file)})
    return run_cli("fetch-data", experiment_id, *args)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def fetch_timeline(experiment_id: str, keywords: Optional[list[str]] = None, start: Optional[int] = None, end: Optional[int] = None, entries: Optional[list[int]] = None, output: Optional[str] = None) -> str:
    """Fetch timeline (states and arrows) from time graph experiment outputs."""
    args = build_args({
        "keywords": ("-k", keywords),
        "start": ("--start", start),
        "end": ("--end", end),
        "entries": ("--entries", entries),
        "output": ("--output", output)
    })
    return run_cli("timeline", experiment_id, *args)


@mcp.tool()
def delete_experiment(experiment_id: str) -> str:
    """Delete an experiment."""
    return run_cli("delete", experiment_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def detect_anomalies(experiment_id: str, keywords: Optional[list[str]] = None, method: Optional[str] = None, resample_freq: Optional[str] = None) -> str:
    """Detect anomalies in trace data using ML methods (iforest, zscore, iqr, moving_average, seasonality, frequency_domain, combined)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "method": (
        "-m", method or "iforest"), "resample_freq": ("-H", resample_freq)})
    return run_cli("anomaly", experiment_id, *args)


@mcp.tool()
def detect_memory_leak(experiment_id: str, keywords: Optional[list[str]] = None) -> str:
    """Detect memory leaks in trace data."""
    args = build_args({"keywords": ("-k", keywords or ["memory"])})
    return run_cli("memory-leak", experiment_id, *args)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def detect_changepoints(experiment_id: str, keywords: Optional[list[str]] = None, methods: Optional[list[str]] = None) -> str:
    """Detect change points in performance trends (single, zscore, voting, pca)."""
    args = build_args({"keywords": ("-k", keywords or ["cpu usage"]), "methods": (
        "-m", methods or ["single", "zscore", "voting", "pca"])})
    return run_cli("changepoint", experiment_id, *args)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def analyze_correlation(experiment_id: str, keywords: Optional[list[str]] = None, method: Optional[str] = None) -> str:
    """Analyze correlation between outputs for root cause analysis (pearson, kendall, spearman)."""
    args = build_args({"keywords": (
        "-k", keywords or ["cpu", "memory"]), "method": ("-m", method or "pearson")})
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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def plan_capacity(experiment_id: str, keywords: Optional[list[str]] = None, horizon: Optional[int] = None) -> str:
    """Perform capacity planning with predictive models."""
    args = build_args({"keywords": (
        "-k", keywords or ["cpu usage"]), "horizon": ("-H", horizon or 100)})
    return run_cli("capacity", experiment_id, *args)


def _render_gantt_png(states) -> "Image":
    """Render Gantt-style states into a PNG image."""
    import io
    fig = build_gantt_figure(states)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    fig.clf()
    return Image(data=buf.getvalue(), format="png")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def plot_timeline(experiment_id: str, keywords: Optional[list[str]] = None,
                  start: Optional[int] = None, end: Optional[int] = None,
                  entries: Optional[list[int]] = None) -> "Image":
    """Render a Gantt-style PNG chart from time graph states. Best used with a narrow time range."""
    args = build_args({
        "keywords": ("-k", keywords),
        "start": ("--start", start),
        "end": ("--end", end),
        "entries": ("--entries", entries),
    })
    result = run_cli("timeline", experiment_id, *args)
    data = json.loads(result)
    output_data = list(data.values())[0]
    states = output_data["states"]
    return _render_gantt_png(states)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False), meta={"ui": {"resourceUri": XY_UI_URI}})
def plot_xy_with_anomalies(
    experiment_id: str,
    keywords: Optional[list[str]] = None,
    method: Optional[str] = None,
    resample_freq: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    as_image: Optional[bool] = None,
):
    """Detect anomalies on XY outputs and return interactive plot data.

    This tool is READ-ONLY and does not modify any experiment data.

    Args:
        experiment_id: Required. The UUID of an existing experiment. If the user refers to an
            experiment by name, call list_experiments first to get the name-UUID mapping.
        keywords: Optional list of keywords to filter outputs, e.g. ["cpu usage"]. Defaults to ["cpu usage"].
        method: Optional anomaly detection method. Must be one of: "iforest", "zscore", "iqr",
            "moving_average", "seasonality", "frequency_domain", "combined". Defaults to "iforest".
        resample_freq: Optional pandas resample frequency string, e.g. "1s", "100ms". Leave None for auto.
        host: Optional trace server host. Defaults to "localhost".
        port: Optional trace server port. Defaults to 8080.
        as_image: If True (default), returns a rendered PNG image. If False, returns a JSON object
            with { "type": "html", "html": "<full HTML content>" } for inline rendering.

    Returns:
        PNG image (when as_image is True/None) or JSON with type "html" and inline HTML content.

    Important:
        - experiment_id must be a valid UUID from an already-opened experiment.
        - Do NOT invent or guess experiment IDs. Use list_experiments to get valid IDs.
        - Do NOT pass output IDs as experiment_id.
        - keywords filter which XY outputs to analyze; they are NOT output IDs.
    """
    global _last_xy_payload
    with _protect_stdout():
        from tmll.tmll_client import TMLLClient
        from tmll.common.models.experiment import Experiment
        from tmll.ml.modules.anomaly_detection.anomaly_detection_module import AnomalyDetection

        client = TMLLClient(host or DEFAULT_HOST, port or DEFAULT_PORT)
        resp = client.tsp_client.fetch_experiment(experiment_id)
        if resp.status_code != 200:
            return json.dumps({"error": f"Experiment {experiment_id} not found", "series": {}})

        experiment = Experiment.from_tsp_experiment(resp.model)
        experiment.assign_outputs(client._fetch_outputs(experiment))
        outputs = experiment.find_outputs(
            keyword=keywords or ["cpu usage"], type=["xy"])
        if not outputs:
            return json.dumps({"error": "No XY outputs match keywords", "series": {}})

        ad_kwargs = {"resample_freq": resample_freq} if resample_freq else {}
        ad = AnomalyDetection(client, experiment, outputs, **ad_kwargs)
        result = ad.find_anomalies(method=method or "iforest")
        if not result:
            return json.dumps({"error": "Anomaly detection returned no results", "series": {}})

        series = {}
        total = 0
        for name, df in ad.dataframes.items():
            anomaly_df = result.anomalies.get(name)
            periods = [[str(s), str(e)]
                       for s, e in result.anomaly_periods.get(name, [])]
            ax, ay = [], []
            if anomaly_df is not None and not anomaly_df.empty:
                mask_cols = anomaly_df.filter(regex="_is_anomaly$")
                mask = mask_cols.any(
                    axis=1) if not mask_cols.empty else anomaly_df.any(axis=1)
                for ts in anomaly_df.index[mask]:
                    if ts in df.index:
                        ax.append(str(ts))
                        ay.append(float(df.loc[ts].iloc[0]))
                total += int(mask.sum())
            series[name] = {
                "x": [str(i) for i in df.index],
                "y": [float(v) for v in df.iloc[:, 0].tolist()],
                "anomaly_x": ax,
                "anomaly_y": ay,
                "periods": periods,
            }

        payload = {
            "summary": f"Found {total} anomalies across {len(series)} outputs using '{method or 'iforest'}'.",
            "method": method or "iforest",
            "series": series,
        }

        if as_image is not False:
            _last_xy_payload = payload
            return _render_png(payload)

        # Return self-contained HTML directly so MCP clients can render it.
        _last_xy_payload = payload
        html = _build_html(payload)
        return json.dumps({"type": "html", "html": html})


_PLOTLY_JS = (UI_DIR / "plotly-basic.min.js").read_text(encoding="utf-8")


def _build_html(payload: dict | None = None) -> str:
    """Return the XY anomalies HTML with Plotly inlined and optional payload baked in."""
    template = (UI_DIR / "xy_anomalies.html").read_text(encoding="utf-8")
    html = template.replace("/* __PLOTLY_INLINE__ */", _PLOTLY_JS)
    if payload:
        bootstrap = f"<script>window.__MCP_PAYLOAD__ = {json.dumps(payload)};</script>"
        html = html.replace("</head>", bootstrap + "</head>", 1)
    return html


def _render_png(payload: dict) -> "Image":
    """Render the anomaly payload to a PNG using matplotlib."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    series = payload.get("series", {})
    n = max(len(series), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for ax, (name, s) in zip(axes[:, 0], series.items()):
        x = pd.to_datetime(s["x"])
        ax.plot(x, s["y"], linewidth=1.2, label=name)
        if s.get("anomaly_x"):
            ax.scatter(pd.to_datetime(s["anomaly_x"]), s["anomaly_y"],
                       color="red", marker="x", s=40, label="Anomalies", zorder=5)
        for a, b in s.get("periods", []):
            ax.axvspan(pd.to_datetime(a), pd.to_datetime(
                b), color="red", alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(payload.get("summary", ""), fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.resource(XY_UI_URI, mime_type=MCP_APP_MIME)
def xy_anomalies_ui() -> str:
    """Interactive iframe plot rendered by MCP Apps hosts for plot_xy_with_anomalies."""
    return _build_html(_last_xy_payload or None)


@mcp.resource("experiment://{experiment_id}", name="experiment", description="An open trace experiment")
def get_experiment_resource(experiment_id: str) -> str:
    """Return details for a specific experiment."""
    for name, uid in _fetch_experiments_fast():
        if uid == experiment_id:
            return json.dumps({"name": name, "uuid": uid})
    return json.dumps({"error": f"Experiment {experiment_id} not found"})


# Override list_resources to dynamically expose each open experiment as a resource.
_original_list_resources = mcp.list_resources


def _fetch_experiments_fast(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> list[tuple[str, str]]:
    """Fetch experiments directly via HTTP (avoids slow subprocess spawn)."""
    try:
        url = f"http://{host}:{port}/tsp/api/experiments"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        return [(exp["name"], exp["UUID"]) for exp in data if "name" in exp and "UUID" in exp]
    except Exception:
        return []


async def _dynamic_list_resources():
    from mcp.types import Resource as MCPResource

    resources = await _original_list_resources()
    for name, exp_id in _fetch_experiments_fast():
        resources.append(MCPResource(
            uri=f"experiment://{exp_id}",
            name=name,
            description=f"Trace experiment: {name}",
            mimeType="application/json",
        ))
    return resources

# Monkey-patching the attribute alone doesn't work because FastMCP registers
# the handler reference at construction time via self._mcp_server.list_resources().
# We must re-register on the underlying server to actually override the handler.
mcp.list_resources = _dynamic_list_resources
mcp._mcp_server.list_resources()(_dynamic_list_resources)


@mcp.tool()
def create_field_plots(analysis_name: str, series: dict[str, list[list[str]]], host: Optional[str] = None, port: Optional[int] = None) -> str:
    """Generate an XML analysis to plot event fields and post it to the trace server.

    Args:
        analysis_name: Unique name for the analysis
        series: Dict mapping series names to lists of [event_name, field_name] pairs.
                Example: {"cpu_prio": [["sched_switch", "prev_prio"]], "mem": [["kmem_alloc", "bytes_alloc"]]}
    """
    import json as _json
    series_json = _json.dumps(series)
    return run_cli(*_global_args(host, port), "create-field-plots", analysis_name, series_json)


if __name__ == "__main__":
    mcp.run()
