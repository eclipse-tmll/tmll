#!/usr/bin/env python3
"""TMLL CLI - Command-line interface for Trace-Server Machine Learning Library"""
import argparse
import sys
import json
import pandas as pd
from tmll.tmll_client import TMLLClient
from tmll.common.models.experiment import Experiment
from tmll.ml.modules.anomaly_detection.anomaly_detection_module import AnomalyDetection
from tmll.ml.modules.anomaly_detection.memory_leak_detection_module import MemoryLeakDetection
from tmll.ml.modules.performance_trend.change_point_module import ChangePointAnalysis
from tmll.ml.modules.root_cause.correlation_module import CorrelationAnalysis
from tmll.ml.modules.resource_optimization.idle_resource_detection_module import IdleResourceDetection
from tmll.ml.modules.predictive_maintenance.capacity_planning_module import CapacityPlanning
from tmll.common.models.timegraph.timegraph import TimeGraph


def get_experiment(client, exp_uuid):
    """Helper to fetch experiment by UUID"""
    resp = client.tsp_client.fetch_experiment(exp_uuid)
    if resp.status_code != 200:
        return None
    exp = Experiment.from_tsp_experiment(resp.model)
    exp.assign_outputs(client._fetch_outputs(exp))
    return exp


def create_experiment(args):
    """Create an experiment from trace files"""
    import os
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    traces = [{"path": os.path.expanduser(path)} for path in args.traces]
    experiment = client.create_experiment(
        traces=traces, experiment_name=args.name)
    if not experiment:
        print("Failed to create experiment")
        return
    print(f"Created experiment: {experiment.name} (UUID: {experiment.uuid})")


def list_outputs(args):
    """List outputs for an experiment"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(
        keyword=args.keywords if args.keywords else None)
    for output in outputs:
        print(f"{output.id}: {output.name} ({output.type})")


def fetch_data_cmd(args):
    """Fetch and export data from outputs"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])
    if not outputs:
        print("No outputs found")
        return

    outputs_with_tree = client.fetch_outputs_with_tree(
        experiment, [o.id for o in outputs])
    data = client.fetch_data(experiment, outputs_with_tree)

    if args.output:
        for output_id, content in data.items():
            if isinstance(content, dict):
                for series_name, df in content.items():
                    safe_name = series_name.replace(
                        "->", "_").replace(" ", "_")
                    df.to_csv(
                        f"{args.output}_{output_id}_{safe_name}.csv", index=False)
            elif isinstance(content, pd.DataFrame):
                content.to_csv(f"{args.output}_{output_id}.csv", index=False)
        print(f"Data exported to {args.output}_*.csv")
    else:
        serializable_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                serializable_data[k] = {
                    sk: sv.to_dict() for sk, sv in v.items() if isinstance(sv, pd.DataFrame)}
            elif isinstance(v, pd.DataFrame):
                serializable_data[k] = v.to_dict()
        print(json.dumps(serializable_data, indent=2, default=str))


COLOR_PALETTE = ["green", "red", "blue", "orange", "purple", "gray"]
BAR_HEIGHT = 9


def states_to_bars(states):
    return [(state["start_time"], state["end_time"] - state["start_time"]) for state in states]


def group_states_by_entry(states):
    grouped = {}
    for state in states:
        entry = state["entry_name"]
        if entry not in grouped:
            grouped[entry] = []
        grouped[entry].append(state)
    return grouped


def get_label_colors(states):
    label_colors = {}
    for state in states:
        label = state["label"]
        if label is None or (isinstance(label, float) and label != label):
            label = "(no label)"
        if label not in label_colors:
            label_colors[label] = COLOR_PALETTE[len(
                label_colors) % len(COLOR_PALETTE)]
    return label_colors


def draw_gantt_to_file(states, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    label_colors = get_label_colors(states)
    grouped = group_states_by_entry(states)

    fig, ax = plt.subplots(figsize=(12, max(4, len(grouped) * 0.5)))
    ax.set_title("TMLL Timeline — Gantt Chart")
    ax.set_xlabel("Time (ns)")

    yticks = []
    ylabels = []
    for i, (entry_name, entry_states) in enumerate(grouped.items()):
        bars = states_to_bars(entry_states)
        colors = []
        for state in entry_states:
            label = state["label"]
            if label is None or (isinstance(label, float) and label != label):
                label = "(no label)"
            colors.append(label_colors[label])
        y_pos = i * 10
        ax.broken_barh(bars, (y_pos, BAR_HEIGHT), facecolors=colors)
        yticks.append(y_pos + BAR_HEIGHT / 2)
        ylabels.append(entry_name)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-1, len(grouped) * 10)

    handles = [mpatches.Patch(color=color, label=label)
               for label, color in label_colors.items()]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fetch_timeline(args):
    """Fetch the timeline of an experiment"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(
        keyword=args.keywords, type=['time_graph'])
    if not outputs:
        print("No timeline outputs found")
        return

    outputs_with_tree = client.fetch_outputs_with_tree(
        experiment, [o.id for o in outputs])
    extra_kwargs = {}
    if args.start is not None:
        extra_kwargs["start"] = args.start
    if args.end is not None:
        extra_kwargs["end"] = args.end

    data = client.fetch_data(experiment, outputs_with_tree, **extra_kwargs)
    arrows_by_output = {}
    for output in outputs:
        arrows_params = {
            "parameters": {
                "requested_timerange": {
                    "start": args.start if args.start is not None else experiment.start,
                    "end": args.end if args.end is not None else experiment.end,
                    "nbTimes": 2
                }
            }
        }
        arrows_response = client.tsp_client.fetch_timegraph_arrows(
            exp_uuid=experiment.uuid, output_id=output.id, parameters=arrows_params)

        if arrows_response.status_code == 200 and arrows_response.model.model:
            arrows_by_output[output.id] = TimeGraph.parse_tsp_arrows(
                arrows_response.model.model)
        else:
            arrows_by_output[output.id] = []

    if args.entries:
        for output_id, df in data.items():
            data[output_id] = df[df["entry_id"].isin(args.entries)]

    serializable_data = {}
    for output_id, df in data.items():
        serializable_data[output_id] = {
            "states": df.to_dict(orient='records'),
            "arrows": [
                {
                    "source_id": arrow.source_id,
                    "target_id": arrow.target_id,
                    "start": arrow.start,
                    "end": arrow.end,
                    "duration": arrow.duration,
                    "style": arrow.style
                }
                for arrow in arrows_by_output[output_id]
            ]
        }

    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(serializable_data, indent=2, default=str))
        print(f"Timeline data exported to {args.output}")
    else:
        print(json.dumps(serializable_data, indent=2, default=str))

    if args.plot:
        for output_id, entry in serializable_data.items():
            draw_gantt_to_file(entry["states"], args.plot)
            break 
        print(f"Gantt chart saved to {args.plot}")


def detect_anomalies(args):
    """Run anomaly detection on an experiment"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])

    if not outputs:
        print("No outputs found matching criteria")
        return

    ad_kwargs = {}
    if args.resample_freq:
        ad_kwargs["resample_freq"] = args.resample_freq
    if args.min_size is not None:
        ad_kwargs["min_size"] = args.min_size
    ad = AnomalyDetection(client, experiment, outputs, **ad_kwargs)
    result = ad.find_anomalies(method=args.method)

    if args.plot:
        ad.plot_anomalies(result)
    else:
        total = sum(len(df[df.filter(regex="_is_anomaly$").any(axis=1)])
                    for df in result.anomalies.values())
        print(f"Found {total} anomalies across {len(result.anomalies)} outputs")

        if total > 0:
            print("\nTop 3 Anomalies:")
            all_anomalies = []
            for name, df in result.anomalies.items():
                is_anomaly = df.filter(regex="_is_anomaly$").any(axis=1)
                anomaly_df = df[is_anomaly].copy()
                anomaly_df["source"] = name
                all_anomalies.append(anomaly_df)

            if all_anomalies:
                combined_anomalies = pd.concat(all_anomalies)
                if "anomaly_score" in combined_anomalies.columns:
                    top_3 = combined_anomalies.sort_values(
                        "anomaly_score", ascending=False).head(3)
                    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
                        print(
                            f"{i}. Time: {idx}, Source: {row['source']}, Score: {row['anomaly_score']:.4f}")


def detect_memory_leak(args):
    """Detect memory leaks"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    mld = MemoryLeakDetection(client, experiment)
    result = mld.analyze_memory_leaks()
    print(f"Memory leak analysis: {result}")


def detect_changepoints(args):
    """Detect change points in performance trends"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])

    if not outputs:
        print("No outputs found")
        return

    cpa = ChangePointAnalysis(client, experiment, outputs)
    changepoints = cpa.get_change_points(methods=args.methods)

    if args.plot:
        cpa.plot_change_points(changepoints)
    else:
        print(
            f"Found {len(changepoints.metrics) if changepoints else 0} change point metrics")


def analyze_correlation(args):
    """Analyze correlation between outputs"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])

    if not outputs:
        print("No outputs found")
        return

    ca = CorrelationAnalysis(client, experiment, outputs)
    correlations = ca.analyze_correlations(method=args.method)

    if args.plot:
        ca.plot_correlation_matrix(correlations)
    else:
        print(f"Correlation results: {correlations}")


def detect_idle_resources(args):
    """Detect idle resources"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])

    if not outputs:
        print("No outputs found")
        return

    ird = IdleResourceDetection(client, experiment, outputs)
    idle = ird.analyze_idle_resources(
        cpu_idle_threshold=args.cpu_idle_threshold,
        memory_idle_threshold=args.memory_idle_threshold,
        disk_idle_threshold=args.disk_idle_threshold,
    )
    print(f"Idle resources: {idle}")


def plan_capacity(args):
    """Perform capacity planning"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    experiment = get_experiment(client, args.experiment)

    if not experiment:
        print("Experiment not found")
        return

    outputs = experiment.find_outputs(keyword=args.keywords, type=['xy'])

    if not outputs:
        print("No outputs found")
        return

    cp = CapacityPlanning(client, experiment, outputs)
    plan = cp.forecast_capacity(forecast_steps=args.horizon)
    print(f"Capacity forecast: {plan}")


def list_experiments(args):
    """List all experiments"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    resp = client.tsp_client.fetch_experiments()

    if resp.status_code != 200:
        print("Failed to fetch experiments")
        return

    for exp in resp.model.experiments:
        print(f"{exp.name} - {exp.UUID}")


def create_field_plots(args):
    """Generate XML analysis for field plots and post it to the trace server."""
    import tempfile
    import os

    client = TMLLClient(args.host, args.port, verbose=args.verbose)

    # Parse the series spec: JSON dict of {series_name: [[event, field], ...]}
    series_spec = json.loads(args.series)

    # Build XML: state provider stores fields under series_name/*,
    # xyView uses entry path="series_name/*" to plot all fields in that series.
    analysis_id = f"org.eclipse.tracecompass.tmll.field.{args.analysis_name}"
    handlers = []
    xy_views = []

    for series_name, event_fields in series_spec.items():
        for event_name, field_name in event_fields:
            handlers.append(
                f'    <eventHandler eventName="{event_name}">\n'
                f'      <stateChange>\n'
                f'        <stateAttribute type="constant" value="{series_name}"/>\n'
                f'        <stateAttribute type="constant" value="{event_name}.{field_name}"/>\n'
                f'        <stateValue type="eventField" value="{field_name}"/>\n'
                f'      </stateChange>\n'
                f'    </eventHandler>'
            )
        xy_views.append(
            f'  <xyView id="{analysis_id}.{series_name}.xy">\n'
            f'    <head>\n'
            f'      <analysis id="{analysis_id}"/>\n'
            f'      <label value="{series_name}"/>\n'
            f'    </head>\n'
            f'    <entry path="{series_name}/*">\n'
            f'      <display type="self"/>\n'
            f'    </entry>\n'
            f'  </xyView>'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<tmfxml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '    xsi:noNamespaceSchemaLocation="xmlDefinition.xsd">\n'
        f'  <stateProvider id="{analysis_id}" version="1">\n'
        f'    <head>\n'
        f'      <label value="{args.analysis_name}"/>\n'
        f'    </head>\n'
        + '\n'.join(handlers) + '\n'
        f'  </stateProvider>\n'
        + '\n'.join(xy_views) + '\n'
        '</tmfxml>\n'
    )

    # Write to file and post (delete first for idempotency)
    xml_path = os.path.join(tempfile.gettempdir(), f"{args.analysis_name}.xml")
    with open(xml_path, 'w') as f:
        f.write(xml)

    config_type = 'org.eclipse.tracecompass.tmf.core.config.xmlsourcetype'
    config_id = f"{args.analysis_name}.xml"
    client.tsp_client.delete_configuration(config_type, config_id)

    response = client.tsp_client.post_configuration(
        config_type, {'path': xml_path})

    if response.status_code == 200:
        print(f"Posted analysis '{args.analysis_name}' (file: {xml_path})")
        print(f"Analysis ID: {analysis_id}")
        print(f"XY View ID: {analysis_id}.xy")
    else:
        print(
            f"Failed to post analysis: {response.status_code} {response.status_text}")


def delete_experiment(args):
    """Delete an experiment"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    client.tsp_client.delete_experiment(args.experiment)
    print(f"Deleted experiment: {args.experiment}")


def main():
    parser = argparse.ArgumentParser(
        description="TMLL CLI - Trace-Server Machine Learning Library")
    parser.add_argument("--host", default="localhost",
                        help="Trace server host")
    parser.add_argument("--port", type=int, default=8080,
                        help="Trace server port")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--log-stderr", action="store_true",
                        help="Send logs to stderr instead of stdout")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # create command
    create_parser = subparsers.add_parser(
        "create", help="Create an experiment")
    create_parser.add_argument("traces", nargs="+", help="Trace file paths")
    create_parser.add_argument(
        "-n", "--name", required=True, help="Experiment name")
    create_parser.set_defaults(func=create_experiment)

    # list command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.set_defaults(func=list_experiments)

    # list-outputs command
    outputs_parser = subparsers.add_parser(
        "list-outputs", help="List outputs for an experiment")
    outputs_parser.add_argument("experiment", help="Experiment UUID")
    outputs_parser.add_argument(
        "-k", "--keywords", nargs="+", help="Filter by keywords")
    outputs_parser.set_defaults(func=list_outputs)

    # fetch-data command
    fetch_parser = subparsers.add_parser(
        "fetch-data", help="Fetch and export data")
    fetch_parser.add_argument("experiment", help="Experiment UUID")
    fetch_parser.add_argument(
        "-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    fetch_parser.add_argument("-o", "--output", help="Output file prefix")
    fetch_parser.set_defaults(func=fetch_data_cmd)

    # timeline command
    timeline_parser = subparsers.add_parser(
        "timeline", help="Fetch timeline data for an experiment")
    timeline_parser.add_argument("experiment", help="Experiment UUID")
    timeline_parser.add_argument(
        "-k", "--keywords", nargs="+", help="Output keywords")
    timeline_parser.add_argument(
        "-o", "--output", help="Output file path (JSON)")
    timeline_parser.add_argument(
        "--start", type=int, help="Start time (nanoseconds)")
    timeline_parser.add_argument(
        "--end", type=int, help="End time (nanoseconds)")
    timeline_parser.add_argument(
        "--entries", nargs="+", type=int, help="Specific entry IDs to query")
    timeline_parser.add_argument(
        "-p", "--plot", help="Render a Gantt-style PNG to this path")
    timeline_parser.set_defaults(func=fetch_timeline)

    # create-field-plots command
    field_plots_parser = subparsers.add_parser(
        "create-field-plots", help="Generate and post XML analysis for field plots")
    field_plots_parser.add_argument(
        "analysis_name", help="Unique name for the analysis")
    field_plots_parser.add_argument(
        "series", help='JSON: {"series_name": [["event", "field"], ...], ...}')
    field_plots_parser.set_defaults(func=create_field_plots)

    # delete command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete an experiment")
    delete_parser.add_argument("experiment", help="Experiment UUID")
    delete_parser.set_defaults(func=delete_experiment)

    # anomaly command
    anomaly_parser = subparsers.add_parser("anomaly", help="Detect anomalies")
    anomaly_parser.add_argument("experiment", help="Experiment UUID")
    anomaly_parser.add_argument(
        "-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    anomaly_parser.add_argument(
        "-m", "--method", default="iforest", help="Detection method")
    anomaly_parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot anomalies")
    anomaly_parser.add_argument(
        "-H", "--resample-freq", help="Resampling frequency")
    anomaly_parser.add_argument(
        "-s", "--min-size", type=int, help="Minimum data points")
    anomaly_parser.set_defaults(func=detect_anomalies)

    # memory-leak command
    memleak_parser = subparsers.add_parser(
        "memory-leak", help="Detect memory leaks")
    memleak_parser.add_argument("experiment", help="Experiment UUID")
    memleak_parser.add_argument(
        "-k", "--keywords", nargs="+", default=["memory"], help="Output keywords")
    memleak_parser.set_defaults(func=detect_memory_leak)

    # changepoint command
    cp_parser = subparsers.add_parser(
        "changepoint", help="Detect change points")
    cp_parser.add_argument("experiment", help="Experiment UUID")
    cp_parser.add_argument("-k", "--keywords", nargs="+",
                           default=["cpu usage"], help="Output keywords")
    cp_parser.add_argument("-m", "--methods", nargs="+", default=["single", "zscore", "voting", "pca"],
                           help="Analysis methods (single, zscore, voting, pca)")
    cp_parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot change points")
    cp_parser.set_defaults(func=detect_changepoints)

    # correlation command
    corr_parser = subparsers.add_parser(
        "correlation", help="Analyze correlation")
    corr_parser.add_argument("experiment", help="Experiment UUID")
    corr_parser.add_argument("-k", "--keywords", nargs="+",
                             default=["cpu", "memory"], help="Output keywords")
    corr_parser.add_argument(
        "-m", "--method", default="pearson", help="Correlation method")
    corr_parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot correlation")
    corr_parser.set_defaults(func=analyze_correlation)

    # idle-resources command
    idle_parser = subparsers.add_parser(
        "idle-resources", help="Detect idle resources")
    idle_parser.add_argument("experiment", help="Experiment UUID")
    idle_parser.add_argument(
        "-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    idle_parser.add_argument("--cpu-idle-threshold", type=float,
                             default=5.0, help="CPU idle threshold percentage")
    idle_parser.add_argument("--memory-idle-threshold", type=float,
                             default=5.0, help="Memory idle threshold percentage")
    idle_parser.add_argument("--disk-idle-threshold", type=float,
                             default=5.0, help="Disk idle threshold percentage")
    idle_parser.set_defaults(func=detect_idle_resources)

    # capacity command
    capacity_parser = subparsers.add_parser(
        "capacity", help="Perform capacity planning")
    capacity_parser.add_argument("experiment", help="Experiment UUID")
    capacity_parser.add_argument(
        "-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    capacity_parser.add_argument(
        "-H", "--horizon", type=int, default=100, help="Forecast steps")
    capacity_parser.set_defaults(func=plan_capacity)

    args = parser.parse_args()

    if args.log_stderr:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, colorize=True, format="{message}")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
