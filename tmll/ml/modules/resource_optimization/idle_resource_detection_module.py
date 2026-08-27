from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.ml.modules.common.statistics import Statistics
from tmll.ml.utils.document_generator import DocumentGenerator
from tmll.ml.utils.formatter import Formatter
from tmll.tmll_client import TMLLClient


class ResourceType(Enum):
    CPU = auto()
    MEMORY = auto()
    DISK = auto()


@dataclass
class ResourceThresholds:
    cpu_idle_threshold: float = 10.0  # percentage
    memory_idle_threshold: float = 10 * 1024 * 1024  # bytes
    disk_idle_threshold: float = 5.0 * 1024 * 1024  # bytes/s
    idle_time: str = '1m'  # time-based


@dataclass
class ResourceMetrics:
    average_usage: float
    peak_usage: float
    idle_percentage: float
    total_duration: float
    idle_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    usage_pattern: str


@dataclass
class IdleResourceAnalysisResult:
    idle_resources: Dict[str, ResourceMetrics]
    resource_type: ResourceType
    analysis_period: Dict[str, Optional[pd.Timestamp]]
    thresholds_used: ResourceThresholds


@dataclass
class SchedulingMetrics:
    total_duration: float
    active_samples: int
    idle_samples: int
    utilization: float
    unique_tasks: int
    most_common_tasks: Dict[str, int]
    number_of_idle_periods: int
    longest_idle_period: int
    average_idle_period: float
    context_switches: int
    context_switches_per_second: float
    task_distribution: Dict[str, float]


class IdleResourceDetection(BaseModule):
    """
    Idle Resource Identification Module

    This module analyzes system resource utilization to identify underutilized (idle) resources
    such as CPU cores, memory, and disk I/O. It uses time series data to detect patterns of
    low utilization and provides insights for potential resource optimization.
    """

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the idle resource identification module.

        :param client: The TMLL client for data communication
        :type client: TMLLClient
        :param experiment: The experiment to analyze
        :type experiment: Experiment
        :param outputs: Optional list of outputs to analyze
        :type outputs: Optional[List[Output]]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        """
        super().__init__(client, experiment)

        self.thresholds: ResourceThresholds = ResourceThresholds()

        self.logger.info("Initializing Idle Resource Identification module.")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        super()._process(outputs=outputs,
                         normalize=False,
                         resample=True,
                         align_timestamps=False,
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        resources_output = self.experiment.get_output_by_name("Resources Status")
        if resources_output is None:
            self.logger.warning("No Resources Status output found")
            return

        resources_df = self.dataframes.get(resources_output.id, pd.DataFrame())
        if resources_df.empty:
            self.logger.warning("No CPU resources found in data")
            return

        # Create a duration column based on end_time - timestamp (index)
        resources_df["duration"] = resources_df.index.to_series().diff().dt.total_seconds().fillna(0).cumsum()

        # Keep only "CPU x Threads" in the "entry_name" column
        resources_df = resources_df[resources_df["entry_name"].str.match(r"^CPU\s+\d+\s+Threads$")].copy()

        # Extract CPU ID
        resources_df["cpu_id"] = resources_df["entry_name"].str.extract(r"CPU (\d+) Threads").astype(int)

        # Create label encoder dictionary and store it
        unique_labels = resources_df["label"].dropna().unique()
        self.resources_task_mapping = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        # Handle empty string, None, and NaN values
        self.resources_task_mapping.update({
            "": 0,
            None: 0,
            pd.NA: 0,
            np.nan: 0
        })

        # Create encoded label column with fillna(0) to handle any remaining NaN values
        resources_df["task"] = resources_df["label"].map(self.resources_task_mapping).fillna(0).astype(int)

        # Process each CPU separately
        all_resampled = []
        for cpu_id in resources_df["cpu_id"].unique():
            # Get data for this CPU
            cpu_data = resources_df[resources_df["cpu_id"] == cpu_id].copy()

            # Resample using forward fill to maintain task values between samples
            resampled = cpu_data.resample("100us")["task"].ffill().fillna(0).astype(int)

            # Convert to DataFrame and add CPU ID
            resampled = pd.DataFrame(resampled)
            resampled["cpu_id"] = cpu_id
            resampled["activity"] = (resampled["task"] > 0).astype(int)

            all_resampled.append(resampled)

        # Combine all resampled data
        resources_df = pd.concat(all_resampled)

        # Reset index to make timestamp a regular column and sort
        resources_df = resources_df.reset_index()
        resources_df = resources_df.rename(columns={"index": "timestamp"})
        resources_df = resources_df.sort_values(["timestamp", "cpu_id"])

        # Set multi-index with timestamp and cpu_id
        resources_df = resources_df.set_index(["timestamp", "cpu_id"])

        self.dataframes[resources_output.id] = resources_df

    def _get_formatted_resource_property(self, resource_property: float, resource_type: ResourceType) -> str:
        """
        Format resource property based on its type.

        :param resource_property: Resource property value
        :type resource_property: float
        :param resource_type: Type of resource (CPU, Memory, Disk)
        :type resource_type: ResourceType
        :return: Formatted threshold string
        :rtype: str
        """
        if resource_type.name == ResourceType.CPU.name:
            return f"{resource_property:.2f}%"
        else:
            val, unit = Formatter.format_bytes(resource_property)
            if resource_type.name == ResourceType.DISK.name:
                return f"{val:.2f}{unit}/s"
            return f"{val:.2f}{unit}"

    def _detect_idle_periods(self, series: pd.Series, resource_type: ResourceType, idle_time: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Detect periods where the resource usage is below the threshold.

        :param series: Time series data of resource usage
        :type series: pd.Series
        :param resource_type: Type of resource being analyzed
        :type resource_type: ResourceType
        :param idle_time: Time threshold for idle periods
        :type idle_time: str
        :return: List of (start_time, end_time) tuples for idle periods
        :rtype: List[Tuple[pd.Timestamp, pd.Timestamp]]
        """
        if resource_type == ResourceType.CPU:
            threshold = self.thresholds.cpu_idle_threshold
        elif resource_type == ResourceType.MEMORY:
            threshold = self.thresholds.memory_idle_threshold
        else:
            threshold = self.thresholds.disk_idle_threshold

        idle_mask = series < threshold

        idle_periods = []
        current_start = None
        idle_time_seconds = Formatter.parse_time_to_seconds(idle_time)

        for time, is_idle in idle_mask.items():
            if is_idle and current_start is None:
                current_start = time
            elif not is_idle and current_start is not None:
                idle_periods.append((current_start, time))
                current_start = None

        # Handle case where series ends during an idle period
        if current_start is not None:
            idle_periods.append((current_start, series.index[-1]))

        # Filter out idle periods shorter than the threshold
        idle_periods = [(start, end) for start, end in idle_periods if (end - start).total_seconds() >= idle_time_seconds]

        return idle_periods

    def _analyze_utilization_pattern(self, series: pd.Series) -> str:
        """
        Analyze the usage pattern of a resource.

        :param series: Time series data of resource usage
        :type series: pd.Series
        :return: Description of the usage pattern
        :rtype: str
        """
        cv = Statistics.get_coefficient_of_variation(series)

        return (
            "No variation" if cv < 0 else
            "Very stable" if cv < 0.1 else
            "Stable" if cv < 0.3 else
            "Moderate variation" if cv < 0.6 else
            "Highly variable"
        )

    def analyze_cpu_scheduling(self) -> Dict[int, SchedulingMetrics]:
        """
        Analyze CPU activity based on thread scheduling data.

        :return: Analysis results for each CPU core
        :rtype: Dict[int, SchedulingMetrics]
        """
        resources_output = self.experiment.get_output_by_name("Resources Status")
        if resources_output is None:
            self.logger.warning("No Resources Status output found")
            return {}

        resources_df = self.dataframes.get(resources_output.id, pd.DataFrame()) 
        if resources_df.empty:
            self.logger.warning("No Resources Status data found")
            return {}

        cpu_ids = resources_df.index.get_level_values("cpu_id").unique()
        results = {}

        for cpu_id in cpu_ids:
            cpu_data = resources_df.xs(cpu_id, level="cpu_id").copy()

            # Revert-back to original task labels
            cpu_data["task"] = cpu_data["task"].map({v: k for k, v in self.resources_task_mapping.items()})

            # Calculate time metrics
            total_duration = (cpu_data.index.get_level_values("timestamp")[-1] - cpu_data.index.get_level_values("timestamp")[0]).total_seconds()
            active_periods = cpu_data[cpu_data["activity"] == 1]
            idle_periods = cpu_data[cpu_data["activity"] == 0]

            # Task analysis
            active_tasks = cpu_data[cpu_data["activity"] == 1]["task"]
            task_counts = active_tasks.value_counts()

            # Calculate consecutive idle periods
            idle_periods = idle_periods.copy()
            idle_periods["idle_group"] = ((idle_periods.index.to_series().diff() > pd.Timedelta(microseconds=100)).cumsum())
            idle_period_lengths = idle_periods.groupby("idle_group").size()

            # Count context switches (changes in task when active)
            task_changes = (active_tasks != active_tasks.shift()).sum()

            results[cpu_id] = SchedulingMetrics(
                total_duration=total_duration,
                active_samples=len(active_periods),
                idle_samples=len(idle_periods),
                utilization=(len(active_periods) / len(cpu_data) * 100),
                unique_tasks=len(task_counts),
                most_common_tasks=task_counts.head(5).to_dict(),
                number_of_idle_periods=len(idle_period_lengths),
                longest_idle_period=idle_period_lengths.max() if not idle_period_lengths.empty else 0,  # type: ignore
                average_idle_period=idle_period_lengths.mean() if not idle_period_lengths.empty else 0,  # type: ignore
                context_switches=task_changes,
                context_switches_per_second=task_changes / total_duration if total_duration > 0 else 0,
                task_distribution={
                    str(task): (count / len(active_periods) * 100) for task, count in task_counts.items()
                }
            )

        return results

    def analyze_idle_resources(self, resource_types: Optional[List[ResourceType]] = None,
                               start_time: Optional[pd.Timestamp] = None,
                               end_time: Optional[pd.Timestamp] = None,
                               **kwargs) -> Dict[ResourceType, IdleResourceAnalysisResult]:
        """
        Analyze resource utilization to identify idle resources.

        :param resource_types: List of resource types to analyze (None for all)
        :type resource_types: Optional[List[ResourceType]]
        :param start_time: Start time for analysis period
        :type start_time: Optional[pd.Timestamp]
        :param end_time: End time for analysis period
        :type end_time: Optional[pd.Timestamp]
        :return: Analysis results for each resource type
        :rtype: Dict[ResourceType, IdleResourceAnalysisResult]
        """
        if not self.dataframes:
            self.logger.warning("No data available for analysis")
            return {}

        if resource_types is None:
            resource_types = list(ResourceType)

        self.thresholds = ResourceThresholds(
            cpu_idle_threshold=kwargs.get("cpu_idle_threshold", ResourceThresholds.cpu_idle_threshold),
            memory_idle_threshold=kwargs.get("memory_idle_threshold", ResourceThresholds.memory_idle_threshold),
            disk_idle_threshold=kwargs.get("disk_idle_threshold", ResourceThresholds.disk_idle_threshold),
            idle_time=kwargs.get("idle_time", ResourceThresholds.idle_time)
        )

        results = {}

        for resource_type in resource_types:
            resource_dfs = {
                name: df for name, df in self.dataframes.items()
                if (
                    (resource_type == ResourceType.CPU and "cpu" in name.lower()) or
                    (resource_type == ResourceType.MEMORY and "memory" in name.lower()) or
                    (resource_type == ResourceType.DISK and "disk" in name.lower())
                )
            }

            if not resource_dfs:
                self.logger.warning(f"No {resource_type.name} resources found in data")
                continue

            # Filter by time period if specified
            if start_time is not None and end_time is not None:
                resource_dfs = {
                    name: df[(df.index >= start_time) & (df.index <= end_time)]
                    for name, df in resource_dfs.items()
                }

            idle_resources = {}

            for name, df in resource_dfs.items():
                if df.empty:
                    continue

                series = df[df.columns[0]]

                avg_usage = series.mean()
                peak_usage = series.max()

                idle_periods = self._detect_idle_periods(series, resource_type, self.thresholds.idle_time)

                total_idle_time = sum((end - start).total_seconds() for start, end in idle_periods)
                total_time = (series.index[-1] - series.index[0]).total_seconds()  # type: ignore
                idle_percentage = (total_idle_time / total_time * 100) if total_time > 0 else 0

                idle_resources[name] = ResourceMetrics(
                    average_usage=avg_usage,
                    peak_usage=peak_usage,
                    idle_percentage=idle_percentage,
                    total_duration=total_time,
                    idle_periods=idle_periods,
                    usage_pattern=self._analyze_utilization_pattern(series),
                )

            df = next(iter(resource_dfs.values()))
            results[resource_type] = IdleResourceAnalysisResult(
                idle_resources=idle_resources,
                resource_type=resource_type,
                analysis_period={
                    "start": start_time if start_time else df.index[0],
                    "end": end_time if end_time else df.index[-1]
                },
                thresholds_used=self.thresholds
            )

        return results

    def interpret(self, idle_results: Optional[Dict[ResourceType, IdleResourceAnalysisResult]] = None,
                  scheduling_results: Optional[Dict[int, SchedulingMetrics]] = None) -> None:
        """
        Interpret and display idle resource analysis and CPU scheduling analysis results using the DocumentGenerator.
        This method provides comprehensive insights into:
        - Overall resource utilization
        - Detailed analysis of idle resources by resource type
        - CPU scheduling analysis results
        - Resource optimization recommendations

        :param idle_results: Results from idle resource analysis by resource type
        :type idle_results: Optional[Dict[ResourceType, IdleResourceAnalysisResult]]
        :param scheduling_results: Results from CPU scheduling analysis by CPU ID
        :type scheduling_results: Optional[Dict[int, SchedulingMetrics]]
        """
        DocumentGenerator.section("Idle Resource Detection Analysis Results")

        # Overall Resource Utilization - Show for all resource types
        overall_metrics = {}
        for resource_type in ResourceType:
            if not any(resource_type.name.lower() in id.lower() for id in self.dataframes):
                overall_metrics[f"{resource_type.name} Average Usage"] = "N/A (No data available)"
                overall_metrics[f"{resource_type.name} Monitoring Duration"] = "N/A (No data available)"
                continue

            if idle_results and resource_type in idle_results and idle_results[resource_type].idle_resources:
                result = idle_results[resource_type]
                avg_idle_percent = sum(m.idle_percentage for m in result.idle_resources.values()) / len(result.idle_resources)
                total_duration = sum(m.total_duration for m in result.idle_resources.values())

                if resource_type == ResourceType.CPU:
                    usage_str = f"{self._get_formatted_resource_property(100 - avg_idle_percent, resource_type)} (Idle periods detected)"
                else:
                    usage_str = f"{self._get_formatted_resource_property(sum(m.average_usage for m in result.idle_resources.values()), resource_type)} (Idle periods detected)"
            else:
                # Calculate metrics for resources without idle periods
                df = next((df for id, df in self.dataframes.items() if resource_type.name.lower() in id.lower()), None)
                if df is not None:
                    usage_str = self._get_formatted_resource_property(df.mean().mean(), resource_type)
                    total_duration = (df.index[-1] - df.index[0]).total_seconds()
                else:
                    usage_str = "N/A (No data available)"
                    total_duration = 0

            overall_metrics[f"{resource_type.name} Average Usage"] = usage_str
            overall_metrics[f"{resource_type.name} Monitoring Duration"] = f"{total_duration:.2f}s" if total_duration > 0 else "N/A (No data available)"

        DocumentGenerator.metrics_group("Overall Resources Utilization", overall_metrics)

        if idle_results:
            for resource_type, result in idle_results.items():
                if result.idle_resources:
                    DocumentGenerator.section(f"{resource_type.name} Resource Analysis")

                    threshold_metrics = {
                        "Analysis Period Start": result.analysis_period["start"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.analysis_period["start"] else "N/A",
                        "Analysis Period End": result.analysis_period["end"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.analysis_period["end"] else "N/A"
                    }

                    threshold_metrics[f"{resource_type.name} Idle Threshold"] = self._get_formatted_resource_property(
                        result.thresholds_used.cpu_idle_threshold if resource_type.name == ResourceType.CPU.name else
                        result.thresholds_used.memory_idle_threshold if resource_type.name == ResourceType.MEMORY.name else
                        result.thresholds_used.disk_idle_threshold,
                        resource_type
                    )

                    DocumentGenerator.metrics_group("Analysis Parameters", threshold_metrics)

                    for resource_id, metrics in result.idle_resources.items():
                        resource_metrics = {}
                        resource_metrics["Average Usage"] = self._get_formatted_resource_property(metrics.average_usage, resource_type)
                        resource_metrics["Peak Usage"] = self._get_formatted_resource_property(metrics.peak_usage, resource_type)

                        resource_metrics.update({
                            "Idle Time Percentage": f"{metrics.idle_percentage:.1f}%",
                            "Total Duration": f"{metrics.total_duration:.2f}s",
                            "Usage Pattern": metrics.usage_pattern,
                            "Number of Idle Periods": len(metrics.idle_periods)
                        })

                        if metrics.idle_periods:
                            longest_idle = max((end - start).total_seconds() for start, end in metrics.idle_periods)
                            resource_metrics["Longest Idle Period"] = f"{longest_idle:.2f}s"

                        output = self.experiment.get_output_by_id(resource_id)
                        resource_name = output.name if output else resource_id
                        DocumentGenerator.metrics_group(f"Resource: {resource_name}", resource_metrics)

                        if metrics.idle_periods:
                            idle_period_headers = ["Start Time", "End Time", "Duration"]
                            top_periods = [(start, end) for start, end in sorted(metrics.idle_periods,
                                                                                 key=lambda x: (x[1] - x[0]).total_seconds(),
                                                                                 reverse=True)[:5]]

                            idle_period_rows = []
                            for start, end in top_periods:
                                duration = (end - start).total_seconds()
                                time_val, time_unit = Formatter.format_seconds(duration)
                                idle_period_rows.append([start.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                                         end.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                                         f"{time_val:.2f} {time_unit}"])

                            DocumentGenerator.table(
                                idle_period_headers,
                                idle_period_rows,
                                f"Top Idle Periods for {resource_name}"
                            )

        if scheduling_results:
            DocumentGenerator.section("CPU Scheduling Analysis")

            overall_scheduling_metrics = {
                "Total CPUs": len(scheduling_results),
                "Total Context Switches": f"{sum(m.context_switches for m in scheduling_results.values()):,}",
                "Total Unique Tasks": sum(m.unique_tasks for m in scheduling_results.values())
            }

            DocumentGenerator.metrics_group("Overall Scheduling Metrics", overall_scheduling_metrics)

            for cpu_id, metrics in scheduling_results.items():
                avg_idle_time = (metrics.average_idle_period * 100) / 1e6
                longest_idle_time = (metrics.longest_idle_period * 100) / 1e6

                cpu_metrics = {
                    "CPU Utilization": f"{metrics.utilization:.1f}%",
                    "Context Switches/s": f"{metrics.context_switches_per_second:.1f}",
                    "Unique Tasks": metrics.unique_tasks,
                    "Active/Idle Samples": f"{metrics.active_samples:,}/{metrics.idle_samples:,}",
                    "Number of Idle Periods": f"{metrics.number_of_idle_periods:,}",
                    "Average Idle Period": f"{Formatter.format_seconds(avg_idle_time)[0]:.2f} {Formatter.format_seconds(avg_idle_time)[1]}",
                    "Longest Idle Period": f"{Formatter.format_seconds(longest_idle_time)[0]:.2f} {Formatter.format_seconds(longest_idle_time)[1]}"
                }

                DocumentGenerator.metrics_group(f"CPU {cpu_id + 1} Scheduling Metrics", cpu_metrics)

                # Show top tasks distribution
                if metrics.task_distribution:
                    task_headers = ["Task", "Active Time %"]
                    task_rows = [
                        [task, f"{percentage:.1f}%"]
                        for task, percentage in sorted(
                            metrics.task_distribution.items(),
                            key=lambda x: float(x[1]),
                            reverse=True
                        )[:min(5, len(metrics.task_distribution))]
                    ]
                    DocumentGenerator.table(
                        task_headers,
                        task_rows,
                        f"Top Tasks for CPU {cpu_id + 1}"
                    )

            # Generate optimization recommendations
            DocumentGenerator.section("Resource Optimization Recommendations")

            DEFAULT_RECOMMENDATION = "Everything looks good! No specific recommendations at this time."
            recommendations = {
                "Workload Distribution": [],
                "Resource Utilization": [],
                "Performance Optimization": [],
                "System Configuration": []
            }

            # CPU-specific recommendations
            if idle_results and any(r.name == ResourceType.CPU.name for r in idle_results):
                cpu_result = next(r for r in idle_results.values() if r.resource_type.name == ResourceType.CPU.name)
                idle_cpus = []
                variable_cpus = []
                high_usage_cpus = []

                for resource_id, metrics in cpu_result.idle_resources.items():
                    output = self.experiment.get_output_by_id(resource_id)
                    resource_name = output.name if output else resource_id
                    idle_cpus.append(f"{resource_name} ({metrics.idle_percentage:.1f}% idle)")

                    if metrics.usage_pattern == "Highly variable":
                        variable_cpus.append(resource_name)
                    elif metrics.peak_usage > 90:
                        high_usage_cpus.append(f"{resource_name} ({metrics.peak_usage:.1f}%)")

                if idle_cpus:
                    recommendations["Workload Distribution"].append(
                        "Consider consolidating workloads on the following CPUs:\n" +
                        "  • " + "\n  • ".join(idle_cpus))

                if variable_cpus:
                    recommendations["Performance Optimization"].append(
                        "High CPU usage variability detected on:\n" +
                        "  • " + "\n  • ".join(variable_cpus) +
                        "\nConsider implementing workload smoothing or batch processing")

                if high_usage_cpus:
                    recommendations["Resource Utilization"].append(
                        "Peak CPU usage exceeds 90% on:\n" +
                        "  • " + "\n  • ".join(high_usage_cpus) +
                        "\nConsider implementing load balancing or scaling resources")

            # Memory-specific recommendations
            if idle_results and any(r.name == ResourceType.MEMORY.name for r in idle_results):
                memory_result = next(r for r in idle_results.values() if r.resource_type.name == ResourceType.MEMORY.name)
                idle_memory = []
                inefficient_memory = []

                for resource_id, metrics in memory_result.idle_resources.items():
                    output = self.experiment.get_output_by_id(resource_id)
                    resource_name = output.name if output else resource_id
                    val, unit = Formatter.format_bytes(metrics.average_usage)
                    peak_val, peak_unit = Formatter.format_bytes(metrics.peak_usage)

                    idle_memory.append(f"{resource_name} (Using {val:.2f} {unit}, {metrics.idle_percentage:.1f}% idle)")

                    memory_efficiency = (metrics.average_usage / metrics.peak_usage) * 100
                    if memory_efficiency < 50:
                        inefficient_memory.append(f"{resource_name} (Avg: {val:.2f} {unit}, Peak: {peak_val:.2f} {peak_unit})")

                if idle_memory:
                    recommendations["Resource Utilization"].append(
                        "Memory allocation could be optimized for:\n" +
                        "  • " + "\n  • ".join(idle_memory))

                if inefficient_memory:
                    recommendations["System Configuration"].append(
                        "Low memory efficiency detected on:\n" +
                        "  • " + "\n  • ".join(inefficient_memory) +
                        "\nConsider adjusting memory limits or implementing memory pooling")

            # Scheduling-specific recommendations
            if scheduling_results:
                high_cs_cpus = []
                high_util_cpus = []
                fragmented_cpus = []

                for cpu_id, metrics in scheduling_results.items():
                    if metrics.context_switches_per_second > 10:
                        high_cs_cpus.append(f"CPU {cpu_id + 1} ({metrics.context_switches_per_second:.1f}/s)")

                    if metrics.utilization > 80:
                        high_util_cpus.append(f"CPU {cpu_id + 1} ({metrics.utilization:.1f}%)")

                    if metrics.number_of_idle_periods > 1000 and metrics.average_idle_period < 0.001:
                        fragmented_cpus.append(f"CPU {cpu_id + 1}")

                if high_cs_cpus:
                    recommendations["Performance Optimization"].append(
                        "High context switch rates detected:\n" +
                        "  • " + "\n  • ".join(high_cs_cpus) +
                        "\nRecommended actions:\n" +
                        "  - Review task scheduling policy\n" +
                        "  - Implement task affinity\n" +
                        "  - Adjust time slice duration")

                if high_util_cpus:
                    recommendations["Resource Utilization"].append(
                        "High CPU utilization detected:\n" +
                        "  • " + "\n  • ".join(high_util_cpus) +
                        "\nRecommended actions:\n" +
                        "  - Implement dynamic load balancing\n" +
                        "  - Configure task prioritization\n" +
                        "  - Consider resource scaling")

                if fragmented_cpus:
                    recommendations["Performance Optimization"].append(
                        "Frequent short idle periods detected on:\n" +
                        "  • " + "\n  • ".join(fragmented_cpus) +
                        "\nConsider coalescing tasks or adjusting scheduler parameters")

            # General recommendations based on overall patterns
            if scheduling_results and idle_results:
                avg_cpu_util = sum(m.utilization for m in scheduling_results.values()) / len(scheduling_results)
                if avg_cpu_util < 30:
                    recommendations["System Configuration"].append(
                        f"Overall low CPU utilization ({avg_cpu_util:.1f}%). Consider:\n"
                        "  - Implementing power-saving strategies\n"
                        "  - Consolidating workloads\n"
                        "  - Adjusting resource allocation"
                    )

            for category, items in recommendations.items():
                if not items:
                    items = [DEFAULT_RECOMMENDATION]
                items = {f"{idx + 1}": item for idx, item in enumerate(items)}
                DocumentGenerator.metrics_group(category, items)

    def plot_resource_utilization(self, analysis_results: Dict[ResourceType, IdleResourceAnalysisResult], **kwargs) -> None:
        """
        Plot resource utilization patterns and highlight idle periods.

        :param analysis_results: Results from idle resource analysis
        :type analysis_results: Dict[ResourceType, IdleResourceAnalysisResult]
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if not analysis_results:
            self.logger.warning("No analysis results to plot")
            return

        fig_size = kwargs.get("fig_size", (15, 4))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab10")

        for idx, (resource_type, result) in enumerate(analysis_results.items()):
            if not result.idle_resources:
                continue

            for resource_id, metrics in result.idle_resources.items():
                df = self.dataframes[resource_id]
                output = self.experiment.get_output_by_id(resource_id)
                resource_name = output.name if output else resource_id

                plots = []
                # Resource usage plot
                plots.append({
                    "plot_type": "time_series",
                    "data": df,
                    "label": f"{resource_name} Usage",
                    "color": colors(idx % 10),
                    "alpha": 0.8,
                    "linewidth": 2
                })

                # Highlight idle periods
                for start, end in metrics.idle_periods:
                    plots.append({
                        "plot_type": "span",
                        "start": start,
                        "end": end,
                        "color": "green",
                        "alpha": 0.2,
                        "label": "Idle Period"
                    })

                # Threshold line
                if resource_type.name == ResourceType.CPU.name:
                    threshold_line = self.thresholds.cpu_idle_threshold
                    ylabel = "CPU Usage (%)"
                elif resource_type.name == ResourceType.MEMORY.name:
                    threshold_line = self.thresholds.memory_idle_threshold
                    ylabel = "Memory Usage"
                else:
                    threshold_line = self.thresholds.disk_idle_threshold
                    ylabel = "Disk Throughput"

                threshold = self._get_formatted_resource_property(threshold_line, resource_type)

                plots.append({
                    "plot_type": "hline",
                    "y": threshold_line,
                    "color": "red",
                    "linestyle": "--",
                    "alpha": 0.5,
                    "label": f"Idle Threshold ({threshold})"
                })

                usage_str = self._get_formatted_resource_property(metrics.average_usage, resource_type)
                peak_str = self._get_formatted_resource_property(metrics.peak_usage, resource_type)

                self._plot(
                    plots,
                    plot_size=fig_size,
                    dpi=fig_dpi,
                    fig_title=f"{resource_type.name} Utilization: {resource_name}\n"
                    f"(Idle {metrics.idle_percentage:.1f}% of time, "
                    f"Avg: {usage_str}, Peak: {peak_str}, "
                    f"Pattern: {metrics.usage_pattern})",
                    fig_xlabel="Time",
                    fig_ylabel=ylabel,
                    fig_num_yticks=8
                )

    def plot_cpu_scheduling(self, scheduling_results: Dict[int, SchedulingMetrics], **kwargs) -> None:
        """
        Plot CPU scheduling analysis results including task distribution and activity patterns.

        :param scheduling_results: Results from analyze_cpu_scheduling method
        :type scheduling_results: Dict[str, Dict[str, Any]]
        :param kwargs: Additional plotting parameters
        :type kwargs: dict
        """
        if not scheduling_results:
            self.logger.warning("No CPU scheduling results to plot")
            return

        resources_output = self.experiment.get_output_by_name("Resources Status")
        if resources_output is None:
            self.logger.warning("No Resources Status output found")
            return

        resources_df = self.dataframes.get(resources_output.id, pd.DataFrame())
        if resources_df.empty:
            self.logger.warning("No Resources Status data found")
            return

        fig_size = kwargs.get("fig_size", (15, 4))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab10")

        # 1: CPU utilization heatmap
        num_slices = 50
        start_time = resources_df.index.get_level_values("timestamp").min()
        end_time = resources_df.index.get_level_values("timestamp").max()
        time_edges = pd.date_range(start=start_time, end=end_time, periods=num_slices+1)
        cpu_ids = sorted(scheduling_results.keys())

        heatmap_data = np.zeros((len(cpu_ids), num_slices))
        for i, cpu_id in enumerate(cpu_ids):
            cpu_data = resources_df.xs(cpu_id, level="cpu_id")
            timestamps = cpu_data.index.get_level_values("timestamp")
            binned_indices = np.digitize(timestamps.astype(np.int64), time_edges.astype(np.int64)[:-1]) - 1

            for slice_idx in range(num_slices):
                mask = binned_indices == slice_idx
                if mask.any():
                    heatmap_data[i, slice_idx] = cpu_data.loc[mask, "activity"].mean() * 100  # type: ignore

        num_xticks = 10
        step = num_slices // num_xticks
        xtick_positions = np.arange(0, num_slices, step)
        xtick_labels = [time_edges[i].strftime("%H:%M:%S") for i in xtick_positions]

        self._plot([{
            "plot_type": "heatmap",
            "data": pd.DataFrame(heatmap_data, index=cpu_ids, columns=time_edges[:-1]),
            "mask": heatmap_data < 0,
            "cmap": "YlOrRd",
            "annot": False,
            "vmin": 0,
            "vmax": 100
        }], plot_size=(fig_size[0], len(cpu_ids)), dpi=fig_dpi, grid=False,
            fig_title="CPU Utilization Heatmap", fig_xlabel="Time", fig_ylabel="CPU Cores",
            fig_xticks=xtick_positions, fig_xticklabels=xtick_labels,
            fig_yticks=range(len(cpu_ids)), fig_yticklabels=[f"CPU {id + 1}" for id in cpu_ids])

        # 2: Task distribution and statistics
        for cpu_id, metrics in scheduling_results.items():
            cpu_data = resources_df.xs(cpu_id, level="cpu_id").copy()
            cpu_data["task"] = cpu_data["task"].map({v: k for k, v in self.resources_task_mapping.items()})
            active_mask = cpu_data["activity"] == 1

            unique_tasks = list(metrics.most_common_tasks.keys())
            plots = []
            for idx, task in enumerate(unique_tasks):
                task_mask = (cpu_data["task"] == task) & active_mask
                if task_mask.any():
                    plots.append({
                        "plot_type": "fill_between",
                        "data": cpu_data,
                        "y1": idx,
                        "y2": idx + 0.8,
                        "where": task_mask,
                        "color": colors(idx % 10),
                        "alpha": 0.8,
                        "label": f"Task {idx + 1}: {task}"
                    })

            self._plot(plots, plot_size=fig_size, dpi=fig_dpi,
                       fig_title=f"Task Distribution Over Time - CPU {cpu_id + 1}",
                       fig_ylabel="Tasks", fig_xlabel="Time",
                       fig_yticks=range(len(unique_tasks)), fig_yticklabels=[f"Task {idx + 1}" for idx in range(len(unique_tasks))],
                       grid=True, grid_alpha=0.3)

            task_distribution = pd.Series(metrics.task_distribution)
            task_distribution = task_distribution.nlargest(25)
            num_tasks = len(task_distribution)

            self._plot([{
                "plot_type": "bar",
                "data": task_distribution,
                "color": [colors(i % 10) for i in range(num_tasks)]
            }], plot_size=fig_size, dpi=fig_dpi,
                fig_title=f"Top {num_tasks} Tasks Distribution Statistics - CPU {cpu_id + 1}",
                fig_xlabel="Tasks",
                fig_ylabel="Percentage of Active Time (%)",
                grid=True,
                grid_alpha=0.3,
                legend=False,
                fig_xticklabels_rotation=90,
                fig_num_yticks=5)
