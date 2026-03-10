#!/usr/bin/env python3
"""MCP server for TMLL CLI - exposes all CLI commands as MCP tools."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("tmll-cli-mcp-server")

CLI_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "cli.py")


def run_cli(*args: str) -> str:
    """Run a tmll_cli.py command and return output."""
    result = subprocess.run(
        [sys.executable, CLI_PATH, "--log-stderr", *args],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or f"CLI exited with code {result.returncode}")
    return result.stdout.strip()


def build_args(arguments: dict, flag_map: dict[str, str]) -> list[str]:
    """Convert tool arguments to CLI flags."""
    args = []
    for key, flag in flag_map.items():
        val = arguments.get(key)
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


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_experiment",
            description="Create a trace experiment from LTTng trace files or directories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "traces": {"type": "array", "items": {"type": "string"}, "description": "Trace file/directory paths"},
                    "experiment_name": {"type": "string"},
                    "host": {"type": "string", "default": "localhost"},
                    "port": {"type": "integer", "default": 8080},
                },
                "required": ["traces", "experiment_name"],
            },
        ),
        Tool(
            name="list_experiments",
            description="List all open experiments",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_outputs",
            description="List available outputs for an experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="fetch_data",
            description="Fetch data from experiment outputs",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu usage"]},
                    "output_file": {"type": "string", "description": "CSV output file prefix"},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="delete_experiment",
            description="Delete an experiment",
            inputSchema={
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="detect_anomalies",
            description="Detect anomalies in trace data using ML methods (iforest, zscore, iqr, moving_average, seasonality, frequency_domain, combined)",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu usage"]},
                    "method": {"type": "string", "default": "iforest", "enum": ["iforest", "zscore", "iqr", "moving_average", "seasonality", "frequency_domain", "combined"]},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="detect_memory_leak",
            description="Detect memory leaks in trace data",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["memory"]},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="detect_changepoints",
            description="Detect change points in performance trends (single, zscore, voting, pca)",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu usage"]},
                    "methods": {"type": "array", "items": {"type": "string"}, "default": ["single", "zscore", "voting", "pca"],
                               "description": "Analysis methods (single, zscore, voting, pca)"},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="analyze_correlation",
            description="Analyze correlation between outputs for root cause analysis (pearson, kendall, spearman)",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu", "memory"]},
                    "method": {"type": "string", "default": "pearson", "enum": ["pearson", "spearman", "kendall"]},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="detect_idle_resources",
            description="Detect idle/underutilized resources",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu usage"]},
                    "cpu_idle_threshold": {"type": "number", "default": 5.0, "description": "CPU idle threshold percentage"},
                    "memory_idle_threshold": {"type": "number", "default": 5.0, "description": "Memory idle threshold percentage"},
                    "disk_idle_threshold": {"type": "number", "default": 5.0, "description": "Disk idle threshold percentage"},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="plan_capacity",
            description="Perform capacity planning with predictive models",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "default": ["cpu usage"]},
                    "horizon": {"type": "integer", "default": 100, "description": "Number of forecast steps"},
                },
                "required": ["experiment_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Optional[dict] = None) -> list[TextContent]:
    arguments = arguments if isinstance(arguments, dict) else {}
    global_args = build_args(arguments, {"host": "--host", "port": "--port"})

    if name == "create_experiment":
        out = run_cli(*global_args, "create", *arguments["traces"], "-n", arguments["experiment_name"])

    elif name == "list_experiments":
        out = run_cli(*global_args, "list")

    elif name == "list_outputs":
        args = build_args(arguments, {"keywords": "-k"})
        out = run_cli(*global_args, "list-outputs", arguments["experiment_id"], *args)

    elif name == "fetch_data":
        args = build_args(arguments, {"keywords": "-k", "output_file": "-o"})
        out = run_cli(*global_args, "fetch-data", arguments["experiment_id"], *args)

    elif name == "delete_experiment":
        out = run_cli(*global_args, "delete", arguments["experiment_id"])

    elif name == "detect_anomalies":
        args = build_args(arguments, {"keywords": "-k", "method": "-m"})
        out = run_cli(*global_args, "anomaly", arguments["experiment_id"], *args)

    elif name == "detect_memory_leak":
        args = build_args(arguments, {"keywords": "-k"})
        out = run_cli(*global_args, "memory-leak", arguments["experiment_id"], *args)

    elif name == "detect_changepoints":
        args = build_args(arguments, {"keywords": "-k", "methods": "-m"})
        out = run_cli(*global_args, "changepoint", arguments["experiment_id"], *args)

    elif name == "analyze_correlation":
        args = build_args(arguments, {"keywords": "-k", "method": "-m"})
        out = run_cli(*global_args, "correlation", arguments["experiment_id"], *args)

    elif name == "detect_idle_resources":
        args = build_args(arguments, {"keywords": "-k",
                                       "cpu_idle_threshold": "--cpu-idle-threshold",
                                       "memory_idle_threshold": "--memory-idle-threshold",
                                       "disk_idle_threshold": "--disk-idle-threshold"})
        out = run_cli(*global_args, "idle-resources", arguments["experiment_id"], *args)

    elif name == "plan_capacity":
        args = build_args(arguments, {"keywords": "-k", "horizon": "-H"})
        out = run_cli(*global_args, "capacity", arguments["experiment_id"], *args)

    else:
        raise ValueError(f"Unknown tool: {name}")

    return [TextContent(type="text", text=out)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
