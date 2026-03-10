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
    experiment = client.create_experiment(traces=traces, experiment_name=args.name)
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
    
    outputs = experiment.find_outputs(keyword=args.keywords if args.keywords else None)
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
    
    outputs_with_tree = client.fetch_outputs_with_tree(experiment, [o.id for o in outputs])
    data = client.fetch_data(experiment, outputs_with_tree)
    
    if args.output:
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(f"{args.output}_{key}.csv", index=False)
            elif isinstance(value, dict):
                for sub_key, df in value.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(f"{args.output}_{key}_{sub_key}.csv", index=False)
        print(f"Data exported to {args.output}_*.csv")
    else:
        result = {}
        for k, v in data.items():
            if isinstance(v, pd.DataFrame):
                result[k] = v.to_dict()
            elif isinstance(v, dict):
                result[k] = {sk: sv.to_dict() for sk, sv in v.items() if isinstance(sv, pd.DataFrame)}
        print(json.dumps(result, indent=2, default=str))


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
    
    ad = AnomalyDetection(client, experiment, outputs)
    result = ad.find_anomalies(method=args.method)
    
    if args.plot:
        ad.plot_anomalies(result)
    else:
        total = sum(len(df) for df in result.anomalies.values())
        print(f"Found {total} anomalies across {len(result.anomalies)} outputs")


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
        print(f"Found {len(changepoints.metrics) if changepoints else 0} change point metrics")


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


def delete_experiment(args):
    """Delete an experiment"""
    client = TMLLClient(args.host, args.port, verbose=args.verbose)
    client.tsp_client.delete_experiment(args.experiment)
    print(f"Deleted experiment: {args.experiment}")


def main():
    parser = argparse.ArgumentParser(description="TMLL CLI - Trace-Server Machine Learning Library")
    parser.add_argument("--host", default="localhost", help="Trace server host")
    parser.add_argument("--port", type=int, default=8080, help="Trace server port")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-stderr", action="store_true", help="Send logs to stderr instead of stdout")

    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # create command
    create_parser = subparsers.add_parser("create", help="Create an experiment")
    create_parser.add_argument("traces", nargs="+", help="Trace file paths")
    create_parser.add_argument("-n", "--name", required=True, help="Experiment name")
    create_parser.set_defaults(func=create_experiment)
    
    # list command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.set_defaults(func=list_experiments)
    
    # list-outputs command
    outputs_parser = subparsers.add_parser("list-outputs", help="List outputs for an experiment")
    outputs_parser.add_argument("experiment", help="Experiment UUID")
    outputs_parser.add_argument("-k", "--keywords", nargs="+", help="Filter by keywords")
    outputs_parser.set_defaults(func=list_outputs)
    
    # fetch-data command
    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch and export data")
    fetch_parser.add_argument("experiment", help="Experiment UUID")
    fetch_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    fetch_parser.add_argument("-o", "--output", help="Output file prefix")
    fetch_parser.set_defaults(func=fetch_data_cmd)
    
    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an experiment")
    delete_parser.add_argument("experiment", help="Experiment UUID")
    delete_parser.set_defaults(func=delete_experiment)
    
    # anomaly command
    anomaly_parser = subparsers.add_parser("anomaly", help="Detect anomalies")
    anomaly_parser.add_argument("experiment", help="Experiment UUID")
    anomaly_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    anomaly_parser.add_argument("-m", "--method", default="iforest", help="Detection method")
    anomaly_parser.add_argument("-p", "--plot", action="store_true", help="Plot anomalies")
    anomaly_parser.set_defaults(func=detect_anomalies)
    
    # memory-leak command
    memleak_parser = subparsers.add_parser("memory-leak", help="Detect memory leaks")
    memleak_parser.add_argument("experiment", help="Experiment UUID")
    memleak_parser.add_argument("-k", "--keywords", nargs="+", default=["memory"], help="Output keywords")
    memleak_parser.set_defaults(func=detect_memory_leak)
    
    # changepoint command
    cp_parser = subparsers.add_parser("changepoint", help="Detect change points")
    cp_parser.add_argument("experiment", help="Experiment UUID")
    cp_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    cp_parser.add_argument("-m", "--methods", nargs="+", default=["single", "zscore", "voting", "pca"],
                           help="Analysis methods (single, zscore, voting, pca)")
    cp_parser.add_argument("-p", "--plot", action="store_true", help="Plot change points")
    cp_parser.set_defaults(func=detect_changepoints)
    
    # correlation command
    corr_parser = subparsers.add_parser("correlation", help="Analyze correlation")
    corr_parser.add_argument("experiment", help="Experiment UUID")
    corr_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu", "memory"], help="Output keywords")
    corr_parser.add_argument("-m", "--method", default="pearson", help="Correlation method")
    corr_parser.add_argument("-p", "--plot", action="store_true", help="Plot correlation")
    corr_parser.set_defaults(func=analyze_correlation)
    
    # idle-resources command
    idle_parser = subparsers.add_parser("idle-resources", help="Detect idle resources")
    idle_parser.add_argument("experiment", help="Experiment UUID")
    idle_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    idle_parser.add_argument("--cpu-idle-threshold", type=float, default=5.0, help="CPU idle threshold percentage")
    idle_parser.add_argument("--memory-idle-threshold", type=float, default=5.0, help="Memory idle threshold percentage")
    idle_parser.add_argument("--disk-idle-threshold", type=float, default=5.0, help="Disk idle threshold percentage")
    idle_parser.set_defaults(func=detect_idle_resources)
    
    # capacity command
    capacity_parser = subparsers.add_parser("capacity", help="Perform capacity planning")
    capacity_parser.add_argument("experiment", help="Experiment UUID")
    capacity_parser.add_argument("-k", "--keywords", nargs="+", default=["cpu usage"], help="Output keywords")
    capacity_parser.add_argument("-H", "--horizon", type=int, default=100, help="Forecast steps")
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
