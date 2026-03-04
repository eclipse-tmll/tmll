import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Tuple, Dict, cast
from scipy import stats
from dataclasses import dataclass
from enum import Enum, auto

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.ml.utils.formatter import Formatter
from tmll.tmll_client import TMLLClient
from tmll.ml.utils.document_generator import DocumentGenerator


class MemoryLeakSeverity(Enum):
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class MemoryThresholds:
    window_size: str = "1s"  # Window size for trend analysis
    fragmentation: float = 0.7  # Memory fragmentation threshold
    growth_slope: float = 0.5  # Growth slope threshold


@dataclass
class MemoryMetrics:
    unreleased_allocations: int
    total_allocations: int
    leak_rate: float
    avg_allocation_size: float
    max_continuous_growth_duration: float
    memory_fragmentation_score: float
    regression_slope: float
    regression_intercept: float


@dataclass
class LeakAnalysisResult:
    severity: MemoryLeakSeverity
    confidence_score: float
    metrics: MemoryMetrics
    detected_patterns: List[str]
    suspicious_locations: pd.DataFrame


class MemoryLeakDetection(BaseModule):
    """
    Memory Leak Detection Module

    In this module, we analyze memory usage patterns to detect memory leaks in the data.
    Memory leaks occur when a program allocates memory but fails to release it, leading to memory exhaustion.
    The analysis consists of the following steps:
        1. Track pointer lifecycles to identify memory leaks
        2. Analyze memory usage trends to detect systematic growth
        3. Analyze allocation patterns to identify potential issues
        4. Calculate comprehensive memory metrics
        5. Evaluate the severity of memory issues and confidence in the assessment
        6. Identify suspicious memory allocation locations
        7. Collect and describe detected memory leak patterns

    Based on the analysis results, we can determine the severity of memory issues and provide insights into potential causes.
    """

    def __init__(self, client: TMLLClient, experiment: Experiment, **kwargs) -> None:
        """
        Initialize the memory leak detection module with the given TMLL client and experiment.

        :param client: The TMLL client to use
        :type client: TMLLClient
        :param experiment: The experiment to analyze
        :type experiment: Experiment
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        """
        super().__init__(client, experiment)

        self.thresholds: MemoryThresholds = MemoryThresholds()
        self.ptr_lifecycle: pd.DataFrame = pd.DataFrame()

        self.logger.info("Initializing Memory Leak Detection module.")

        self.required_outputs = experiment.find_outputs(keyword=["memory usage"], type=["xy"], match_any=True)
        self.required_outputs.extend(experiment.find_outputs(keyword=["events table"], type=["table"], match_any=True))
        self._process(self.required_outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        super()._process(outputs=outputs,
                         fetch_params={"table_line_column_names": ["event type", "contents", "timestamp ns"]},
                         resample=kwargs.get("resample", False),
                         align_timestamps=False,
                         normalize=False,
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        events_df = self._get_events_dataframe()
        if events_df is not None:
            events_output = self.experiment.get_output_by_name("Events Table")
            id = events_output.id if events_output is not None else None
            if id is None:
                return

            if not all(col in events_df.columns for col in ["size", "ptr"]):
                self.logger.warning("Events table does not contain necessary columns for memory leak analysis")
                return

            events_df["event_category"] = "other"
            events_df.loc[events_df["Event type"].str.contains("malloc", na=False), "event_category"] = "allocation"
            events_df.loc[events_df["Event type"].str.contains("free", na=False), "event_category"] = "deallocation"
            events_df = events_df[events_df["event_category"] != "other"]

            events_df = events_df.rename({"size": "allocation_size"}, axis=1)
            events_df["allocation_size"] = events_df["allocation_size"].astype(float)
            events_df["ptr"] = events_df["ptr"].astype(str)

            events_df = self._separate_events(events_df)

            self.dataframes[id] = events_df

    def _separate_events(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Separate memory allocation and deallocation events from other events.
        Also, we only keep the events related to memory allocation and deallocation.

        :param dataframe: The original events table
        :type dataframe: pd.DataFrame
        :return: The processed events table
        :rtype: pd.DataFrame
        """
        dataframe["event_category"] = "other"
        dataframe.loc[dataframe["Event type"].str.contains("malloc", na=False), "event_category"] = "allocation"
        dataframe.loc[dataframe["Event type"].str.contains("free", na=False), "event_category"] = "deallocation"
        dataframe = dataframe[dataframe["event_category"] != "other"]

        dataframe = dataframe.rename({"size": "allocation_size"}, axis=1)
        dataframe["allocation_size"] = dataframe["allocation_size"].astype(float)

        dataframe["ptr"] = dataframe["ptr"].astype(str)

        return dataframe

    def _get_memory_usage_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the memory usage dataframe from the experiment.

        :return: The memory usage dataframe
        :rtype: Optional[pd.DataFrame]
        """
        mu_outputs = self.experiment.get_outputs_by_name("Memory Usage", True)
        ust_mu_output = next((o for o in mu_outputs if "ust" in o.id.lower()), None)
        if ust_mu_output is not None:
            return self.dataframes.get(ust_mu_output.id, None)
        return None

    def _get_events_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the events table dataframe from the experiment.

        :return: The events dataframe
        :rtype: Optional[pd.DataFrame]
        """
        et_output = self.experiment.get_output_by_name("Events Table")
        if et_output is not None:
            return self.dataframes.get(et_output.id, None)
        return None

    def analyze_memory_leaks(self, window_size: str = MemoryThresholds.window_size,
                             fragmentation_threshold: float = MemoryThresholds.fragmentation,
                             slope_threshold: float = MemoryThresholds.growth_slope) -> LeakAnalysisResult:
        """
        Analyze memory usage patterns and detect memory leaks in the data.
        Here, the analysis consists of the following steps:
            1. Track pointer lifecycles to identify memory leaks
            2. Analyze memory usage trends to detect systematic growth
            3. Analyze allocation patterns to identify potential issues
            4. Calculate comprehensive memory metrics
            5. Evaluate the severity of memory issues and confidence in the assessment
            6. Identify suspicious memory allocation locations
            7. Collect and describe detected memory leak patterns

        :param window_size: The window size for trend analysis, defaults to "1s"
        :type window_size: str, optional
        :param fragmentation_threshold: The threshold for memory fragmentation, defaults to 0.7
        :type fragmentation_threshold: float, optional
        :param slope_threshold: The threshold for memory growth slope, defaults to 0.5
        :type slope_threshold: float, optional
        :return: The results of the memory leak analysis
        :rtype: LeakAnalysisResult
        """
        et_df = self._get_events_dataframe()
        if et_df is None:
            self.logger.warning("Events table output not found in the experiment")
            return LeakAnalysisResult(
                severity=MemoryLeakSeverity.NONE,
                confidence_score=0.0,
                metrics=MemoryMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                detected_patterns=[],
                suspicious_locations=pd.DataFrame()
            )

        mu_df = self._get_memory_usage_dataframe()
        if mu_df is None:
            self.logger.warning("Memory usage output not found in the experiment")
            return LeakAnalysisResult(
                severity=MemoryLeakSeverity.NONE,
                confidence_score=0.0,
                metrics=MemoryMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                detected_patterns=[],
                suspicious_locations=pd.DataFrame()
            )

        self.thresholds = MemoryThresholds(
            window_size=window_size,
            fragmentation=fragmentation_threshold,
            growth_slope=slope_threshold
        )

        ptr_tracking = self._track_pointer_lifecycle()
        memory_trend = self._analyze_memory_trend()
        allocation_patterns = self._analyze_allocation_patterns()
        metrics = self._calculate_memory_metrics(ptr_tracking, memory_trend, allocation_patterns)
        severity, confidence = self._evaluate_severity(metrics, memory_trend["p_value"])
        suspicious_locs = self._identify_suspicious_locations(ptr_tracking)
        patterns = self._collect_detected_patterns(memory_trend, allocation_patterns)

        return LeakAnalysisResult(
            severity=severity,
            confidence_score=confidence,
            metrics=metrics,
            detected_patterns=patterns,
            suspicious_locations=suspicious_locs
        )

    def _track_pointer_lifecycle(self) -> pd.DataFrame:
        """
        Track the lifecycle of memory pointers to identify memory leaks.

        :return: A DataFrame containing information about memory leaks
        :rtype: pd.DataFrame
        """
        events_df = self._get_events_dataframe()
        if events_df is None or events_df.empty:
            self.logger.warning("No events data available for memory leak analysis")
            return pd.DataFrame()

        # Separate allocations and deallocations
        allocations = events_df[events_df["event_category"] == "allocation"]
        deallocations = events_df[events_df["event_category"] == "deallocation"]

        # Create base allocation records
        allocation_records = allocations.copy()
        allocation_records["allocation_time"] = allocation_records.index

        # Check for duplicate allocations (potential corruption)
        duplicate_allocs = allocation_records["ptr"].value_counts()
        if any(duplicate_allocs > 1):
            # Keep only the first allocation for duplicates
            allocation_records = allocation_records.groupby("ptr").first().reset_index()

        # Merge with deallocations to get lifecycle
        merged = pd.merge(
            allocation_records,
            deallocations[["ptr"]].assign(deallocation_time=deallocations.index),
            on="ptr",
            how="left"
        )

        # Calculate lifetime
        merged["lifetime"] = (merged["deallocation_time"] - merged["allocation_time"]).dt.total_seconds()
        merged["lifetime"] = merged["lifetime"].clip(lower=0)

        # Rename size column
        merged = merged.rename(columns={"allocation_size": "size"})

        self.ptr_lifecycle = merged
        return self.ptr_lifecycle

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """
        Analyze memory usage trends to detect systematic growth.

        :return: The results of the memory usage trend analysis
        :rtype: Dict[str, Any]
        """
        memory_df = self._get_memory_usage_dataframe()
        if memory_df is None or memory_df.empty:
            self.logger.warning("No memory usage data available for trend analysis")
            return {}

        memory_df = self.data_preprocessor.normalize(memory_df)

        # Convert timestamps to seconds from start
        time_seconds = (memory_df.index - memory_df.index[0]).total_seconds()

        # Calculate rolling statistics
        window_size = pd.Timedelta(self.thresholds.window_size)
        rolling_mean = memory_df.iloc[:, 0].rolling(window=window_size).mean()
        rolling_std = memory_df.iloc[:, 0].rolling(window=window_size).std()

        # Perform linear regression with actual time intervals
        slope, intercept, r_value, p_value, _ = stats.linregress(time_seconds, memory_df.iloc[:, 0].values)
        slope = cast(float, slope)
        intercept = cast(float, intercept)
        r_value = cast(float, r_value)
        p_value = cast(float, p_value)

        # Calculate growth characteristics
        is_significant = p_value < 0.05
        is_increasing = slope > 0
        growth_rate = slope if is_increasing else 0

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "growth_rate": growth_rate,
            "is_significant": is_significant,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std
        }

    def _analyze_allocation_patterns(self) -> Dict[str, Any]:
        """
        Analyze allocation patterns to identify potential issues.

        :return: The results of the allocation pattern analysis
        :rtype: Dict[str, Any]
        """
        events_df = self._get_events_dataframe()
        if events_df is None or events_df.empty:
            self.logger.warning("No events data available for allocation pattern analysis")
            return {}

        allocation_events = events_df[events_df["event_category"] == "allocation"]

        # Check for null values in allocation size
        null_sizes = allocation_events["allocation_size"].isnull().sum()
        if null_sizes > 0:
            self.logger.warning(
                f"Found {null_sizes} allocations with null size. "
                "This might indicate trace corruption or incomplete data."
            )

        # Calculate allocation frequencies
        allocation_freq = allocation_events.resample(self.thresholds.window_size).size()

        # Analyze allocation sizes with null handling
        allocation_sizes = allocation_events["allocation_size"]
        valid_sizes = allocation_sizes.dropna()

        return {
            "allocation_frequency": allocation_freq,
            "mean_allocation_size": valid_sizes.mean() if not valid_sizes.empty else 0,
            "median_allocation_size": valid_sizes.median() if not valid_sizes.empty else 0,
            "size_std": valid_sizes.std() if not valid_sizes.empty else 0,
            "total_allocations": len(allocation_events),
            "unique_sizes": valid_sizes.nunique() if not valid_sizes.empty else 0,
            "null_size_count": null_sizes
        }

    def _calculate_memory_metrics(self, ptr_tracking: pd.DataFrame, memory_trend: Dict, allocation_patterns: Dict) -> MemoryMetrics:
        """
        Calculate comprehensive memory metrics to assess memory issues.
        Here, the metrics include:
            - Unreleased Allocations
            - Total Allocations
            - Leak Rate
            - Average Allocation Size
            - Maximum Continuous Growth Duration
            - Memory Fragmentation Score

        :param ptr_tracking: The DataFrame containing memory leak information
        :type ptr_tracking: pd.DataFrame
        :param memory_trend: The results of the memory usage trend analysis
        :type memory_trend: Dict
        :param allocation_patterns: The results of the allocation pattern analysis
        :type allocation_patterns: Dict
        :return: The calculated memory metrics
        :rtype: MemoryMetrics
        """
        # Calculate unreleased allocations
        unreleased = len(ptr_tracking[ptr_tracking["deallocation_time"].isna()])

        # Calculate memory fragmentation score
        if not ptr_tracking.empty:
            allocated_chunks = len(ptr_tracking)
            concurrent_chunks = len(ptr_tracking[ptr_tracking["deallocation_time"].isna()])
            fragmentation_score = concurrent_chunks / allocated_chunks if allocated_chunks > 0 else 0
        else:
            fragmentation_score = 0

        # Calculate maximum continuous growth duration
        rolling_mean = memory_trend["rolling_mean"]
        growth_periods = (rolling_mean.diff() > 0)
        consecutive_periods = growth_periods.groupby((growth_periods != growth_periods.shift()).cumsum())
        max_growth_duration = (consecutive_periods.apply(lambda x: (x.index[-1] - x.index[0]).total_seconds() if len(x) > 1 else 0).max())

        return MemoryMetrics(
            unreleased_allocations=unreleased,
            total_allocations=allocation_patterns["total_allocations"],
            leak_rate=memory_trend["growth_rate"],
            avg_allocation_size=allocation_patterns["mean_allocation_size"],
            max_continuous_growth_duration=max_growth_duration,
            memory_fragmentation_score=fragmentation_score,
            regression_slope=memory_trend["slope"],
            regression_intercept=memory_trend["intercept"]
        )

    def _evaluate_severity(self, metrics: MemoryMetrics, p_value: float) -> Tuple[MemoryLeakSeverity, float]:
        """
        Evaluate the severity of memory issues and confidence in the assessment.
        Severity indicates the impact of memory issues, while confidence reflects the reliability of the analysis.
        The severity levels are defined as follows:
            - NONE: No memory issues detected
            - LOW: Low impact memory issues detected
            - MEDIUM: Medium impact memory issues detected
            - HIGH: High impact memory issues detected
            - CRITICAL: Critical impact memory issues detected

        :param metrics: The calculated memory metrics
        :type metrics: MemoryMetrics
        :param p_value: The p-value of the memory usage trend analysis
        :type p_value: float
        :return: The severity of memory issues and confidence in the assessment
        :rtype: Tuple[MemoryLeakSeverity, float]
        """
        # Calculate base scores
        growth_score = min(1.0, metrics.leak_rate / self.thresholds.growth_slope)
        unreleased_score = min(1.0, metrics.unreleased_allocations / metrics.total_allocations)
        fragmentation_score = min(1.0, metrics.memory_fragmentation_score / self.thresholds.fragmentation)

        # Weight the scores
        weighted_score = (
            0.4 * growth_score +
            0.4 * unreleased_score +
            0.2 * fragmentation_score
        )

        # Calculate confidence based on data quality
        confidence = min(1.0, (
            0.6 * (1 - p_value) +
            0.4 * (metrics.total_allocations)
        ))

        # Determine severity
        if weighted_score < 0.2:
            severity = MemoryLeakSeverity.NONE
        elif weighted_score < 0.4:
            severity = MemoryLeakSeverity.LOW
        elif weighted_score < 0.6:
            severity = MemoryLeakSeverity.MEDIUM
        elif weighted_score < 0.8:
            severity = MemoryLeakSeverity.HIGH
        else:
            severity = MemoryLeakSeverity.CRITICAL

        return severity, confidence

    def _identify_suspicious_locations(self, ptr_tracking: pd.DataFrame) -> pd.DataFrame:
        """
        Identify suspicious memory allocation locations.
        These locations are characterized by large amounts of unreleased memory.

        :param ptr_tracking: The DataFrame containing memory leak information
        :type ptr_tracking: pd.DataFrame
        :return: The top suspicious memory allocation locations
        :rtype: pd.DataFrame
        """
        events_df = self._get_events_dataframe()
        if events_df is None or events_df.empty:
            self.logger.warning("No events data available for suspicious location analysis")
            return pd.DataFrame()

        # Find allocations without matching deallocations
        unfreed_ptrs = ptr_tracking[ptr_tracking["deallocation_time"].isna()]["ptr"]

        # Get allocation events for unfreed pointers
        suspicious_allocs = events_df[
            (events_df["event_category"] == "allocation") &
            (events_df["ptr"].isin(unfreed_ptrs))
        ]

        # Group by location and calculate metrics
        location_metrics = suspicious_allocs.groupby(["ptr"]).agg({
            "allocation_size": ["sum", "count"],
            "Event type": "first"  # Keep the original event for context
        }).reset_index()

        location_metrics.columns = ["ptr", "total_bytes", "allocation_count", "event_context"]

        return location_metrics.sort_values("total_bytes", ascending=False)

    def _collect_detected_patterns(self, memory_trend: Dict[str, Any], allocation_patterns: Dict[str, Any]) -> List[str]:
        """
        Collect and describe detected memory leak patterns.
        These patterns provide insights into the detected memory issues. For example:
            - Systematic memory growth detected
            - Irregular allocation pattern detected
            - High memory usage volatility detected

        :param memory_trend: The results of the memory usage trend analysis
        :type memory_trend: Dict[str, Any]
        :param allocation_patterns: The results of the allocation pattern analysis
        :type allocation_patterns: Dict[str, Any]
        :return: The detected memory leak patterns
        :rtype: List[str]
        """
        patterns = []

        if memory_trend["is_significant"] and memory_trend["slope"] > 0:
            growth_val, growth_unit = Formatter.format_bytes(memory_trend["growth_rate"])
            patterns.append(f"Systematic memory growth detected: {growth_val:.2f} {growth_unit}/s")

        if allocation_patterns["allocation_frequency"].std() > allocation_patterns["allocation_frequency"].mean():
            patterns.append("Irregular allocation pattern detected")

        if memory_trend["rolling_std"].mean() > memory_trend["rolling_mean"].mean() * 0.1:
            patterns.append("High memory usage volatility detected")

        return patterns

    def interpret(self, analysis_result: LeakAnalysisResult) -> None:
        """Interpret and display memory leak analysis results using the DocumentGenerator."""

        DocumentGenerator.section("Memory Leak Analysis Results")

        DocumentGenerator.metrics_group("Analysis Overview", {
            "Severity": analysis_result.severity.name,
            "Confidence Score": f"{analysis_result.confidence_score:.2f}"
        })

        leak_val, leak_unit = Formatter.format_bytes(analysis_result.metrics.leak_rate)
        avg_val, avg_unit = Formatter.format_bytes(analysis_result.metrics.avg_allocation_size)
        max_growth_val, max_growth_unit = Formatter.format_seconds(analysis_result.metrics.max_continuous_growth_duration)
        DocumentGenerator.metrics_group("Memory Metrics", {
            "Unreleased Allocations": analysis_result.metrics.unreleased_allocations,
            "Total Allocations": analysis_result.metrics.total_allocations,
            "Leak Rate": f"{leak_val:.2f} {leak_unit}/s",
            "Average Allocation Size": f"{avg_val:.2f} {avg_unit}",
            "Max Continuous Growth": f"{max_growth_val:.2f} {max_growth_unit}",
            "Memory Fragmentation": f"{(analysis_result.metrics.memory_fragmentation_score * 100):.2f}%"
        })

        DocumentGenerator.metrics_group("Detected Patterns", {
            f"{i + 1}": pattern for i, pattern in enumerate(analysis_result.detected_patterns)
        })

        if not analysis_result.suspicious_locations.empty:
            suspicious_headers = ["Pointer", "Size", "Allocation Count", "Event Context"]
            suspicious_rows = [
                [row["ptr"], f"{total_bytes:.2f} {unit}", row["allocation_count"], row["event_context"]]
                for _, row in analysis_result.suspicious_locations.head().iterrows()
                for total_bytes, unit in [Formatter.format_bytes(row["total_bytes"])]
            ]
            DocumentGenerator.table(
                suspicious_headers,
                suspicious_rows,
                "Top 5 Suspicious Locations"
            )

        memory_df = self._get_memory_usage_dataframe()
        if memory_df is not None:
            peak_val, peak_unit = Formatter.format_bytes(memory_df.iloc[:, 0].max())
            avg_val, avg_unit = Formatter.format_bytes(memory_df.iloc[:, 0].mean())
            std_val, std_unit = Formatter.format_bytes(memory_df.iloc[:, 0].std())
            DocumentGenerator.metrics_group("Memory Usage Statistics", {
                "Peak Memory Usage": f"{peak_val:.2f} {peak_unit}",
                "Average Memory Usage": f"{avg_val:.2f} {avg_unit}",
                "Memory Usage Std Dev": f"{std_val:.2f} {std_unit}"
            })

        events_df = self._get_events_dataframe()
        if events_df is not None:
            allocation_events = events_df[
                events_df["event_category"] == "allocation"
            ]
            deallocation_events = events_df[
                events_df["event_category"] == "deallocation"
            ]
            unmatched_allocations = allocation_events.loc[
                ~allocation_events["ptr"].isin(deallocation_events["ptr"])
            ]["ptr"].unique()

            DocumentGenerator.metrics_group("Allocation Statistics", {
                "Total Allocations": f"{len(allocation_events):,}",
                "Total Deallocations": f"{len(deallocation_events):,}",
                "Unmatched Allocations": f"{len(unmatched_allocations):,}"
            })

        ptr_tracking = self._track_pointer_lifecycle()
        lifetimes = ptr_tracking["lifetime"].dropna()
        if not lifetimes.empty:
            avg_val, avg_unit = Formatter.format_seconds(lifetimes.mean())
            med_val, med_unit = Formatter.format_seconds(lifetimes.median())
            max_val, max_unit = Formatter.format_seconds(lifetimes.max())
            DocumentGenerator.metrics_group("Pointer Lifetime Statistics", {
                "Average Lifetime": f"{avg_val:.2f} {avg_unit}",
                "Median Lifetime": f"{med_val:.2f} {med_unit}",
                "Maximum Lifetime": f"{max_val:.2f} {max_unit}"
            })

    def plot_memory_leak_analysis(self, analysis_result: LeakAnalysisResult, **kwargs) -> None:
        """
        Plot memory usage trends and analysis results.

        :param analysis_result: The results of the memory leak analysis
        :type analysis_result: LeakAnalysisResult
        :param kwargs: Additional keyword arguments
        """
        memory_df = self._get_memory_usage_dataframe()
        events_df = self._get_events_dataframe()

        fig_size = kwargs.get("fig_size", (15, 5))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab10")

        if memory_df is not None and not memory_df.empty:
            points_per_window = max(len(memory_df) // 10, 1)

            # Scale the trend line as it was trained on normalized data, so we need to scale it back
            analysis_result.metrics.regression_intercept = analysis_result.metrics.regression_intercept * memory_df.iloc[:, 0].max()
            analysis_result.metrics.regression_slope = analysis_result.metrics.regression_slope * memory_df.iloc[:, 0].max()
            trend_line = pd.DataFrame({
                "timestamp": memory_df.index,
                "trend_line": analysis_result.metrics.regression_slope *
                (memory_df.index - memory_df.index[0]).total_seconds() +
                analysis_result.metrics.regression_intercept
            })

            # Plot 1: Memory Usage Over Time
            plots = [
                {
                    "plot_type": "time_series",
                    "data": memory_df,
                    "label": "Memory Usage",
                    "alpha": 0.8,
                    "linewidth": 2,
                    "color": colors(0)
                },
                {
                    "plot_type": "time_series",
                    "data": memory_df.rolling(window=points_per_window, min_periods=1, center=True).mean(),
                    "label": "Rolling Mean",
                    "alpha": 0.9,
                    "linewidth": 2,
                    "color": colors(1)
                },
                {
                    "plot_type": "time_series",
                    "data": trend_line,
                    "x": "timestamp",
                    "y": "trend_line",
                    "label": "Trend",
                    "color": colors(2),
                    "linestyle": "--",
                    "alpha": 0.8,
                    "linewidth": 2
                }
            ]

            self._plot(plots, plot_size=fig_size, dpi=fig_dpi, fig_title="Memory Usage Over Time",
                       fig_xlabel="Time", fig_ylabel="Memory Usage", grid=True)

        if events_df is not None and not events_df.empty:
            allocation_events = events_df[events_df["event_category"] == "allocation"]
            deallocation_events = events_df[events_df["event_category"] == "deallocation"]
            ptr_tracking = self._track_pointer_lifecycle()
            lifetimes = ptr_tracking["lifetime"].dropna()

            # Plot 2: Allocation Patterns
            resample_window = pd.Timedelta(self.thresholds.window_size) / 10
            alloc_series = allocation_events.resample(resample_window).size()
            dealloc_series = deallocation_events.resample(resample_window).size()

            plots = [
                {
                    "plot_type": "time_series",
                    "data": alloc_series,
                    "label": "Allocations",
                    "alpha": 0.6,
                    "color": colors(3),
                    "linewidth": 2.5
                },
                {
                    "plot_type": "time_series",
                    "data": dealloc_series,
                    "label": "Deallocations",
                    "alpha": 0.7,
                    "color": colors(4),
                    "linewidth": 2.5
                }
            ]
            self._plot(plots, plot_size=fig_size, dpi=fig_dpi, fig_title="Memory Operations Over Time",
                       fig_xlabel="Time", fig_ylabel="Operations per Second", grid=True)

            # Plot 3: Pointer Lifetime Distribution
            mean_val, mean_unit = Formatter.format_seconds(lifetimes.mean())
            med_val, med_unit = Formatter.format_seconds(lifetimes.median())
            plots = [
                {
                    "plot_type": "histogram",
                    "data": lifetimes,
                    "bins": 50,
                    "alpha": 0.8,
                    "color": colors(5)
                },
                {
                    "plot_type": "vline",
                    "x": lifetimes.mean(),
                    "label": f"Mean: {mean_val:.2f} {mean_unit}",
                    "color": colors(6),
                    "linestyle": "--",
                    "linewidth": 2.5
                },
                {
                    "plot_type": "vline",
                    "x": lifetimes.median(),
                    "label": f"Median: {med_val:.2f} {med_unit}",
                    "color": colors(7),
                    "linestyle": "--",
                    "linewidth": 2.5
                }
            ]
            self._plot(plots, plot_size=fig_size, dpi=fig_dpi, fig_title="Pointer Lifetime Distribution",
                       fig_xlabel="Lifetime (seconds)", fig_ylabel="Count", grid=True)

            # Plot 4: Memory Fragmentation Analysis
            alloc_cumsum = allocation_events.resample(resample_window).size().cumsum()
            dealloc_cumsum = deallocation_events.resample(resample_window).size().cumsum()
            active_allocations = alloc_cumsum - dealloc_cumsum
            total_ops = alloc_cumsum + dealloc_cumsum
            fragmentation_score = abs(active_allocations / total_ops.replace(0, 1)) * 100

            plots = [
                {
                    "plot_type": "time_series",
                    "data": fragmentation_score,
                    "label": "Fragmentation Score",
                    "color": colors(8),
                    "alpha": 0.8,
                    "linewidth": 2.5
                },
                {
                    "plot_type": "hline",
                    "y": self.thresholds.fragmentation * 100,
                    "label": f"Threshold ({self.thresholds.fragmentation * 100}%)",
                    "color": colors(9),
                    "linestyle": "--",
                    "linewidth": 2.5
                }
            ]
            self._plot(plots, plot_size=fig_size, dpi=fig_dpi, fig_title="Memory Fragmentation Analysis",
                       fig_xlabel="Time", fig_ylabel="Fragmentation Score (%)", grid=True)
