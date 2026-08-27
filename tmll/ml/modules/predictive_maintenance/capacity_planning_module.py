from dataclasses import dataclass
from enum import Enum, auto
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, cast

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, acf, pacf

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.ml.modules.common.statistics import Statistics
from tmll.ml.utils.formatter import Formatter
from tmll.tmll_client import TMLLClient
from tmll.ml.utils.document_generator import DocumentGenerator


class ResourceType(Enum):
    CPU = auto()
    MEMORY = auto()
    DISK = auto()


class ForecastMethod(Enum):
    ARIMA = auto()
    VAR = auto()
    MOVING_AVERAGE = auto()


@dataclass
class ResourceThresholds:
    cpu_threshold: float = 85.0  # percentage
    memory_threshold: float = 10 * 1024 * 1024  # MB
    disk_threshold: float = 50 * 1024 * 1024  # MB/s
    warning_period: str = '1m'  # time-based


@dataclass
class ResourceMetrics:
    current_usage: float
    peak_usage: float
    average_usage: float
    forecast_values: List[float]
    forecast_timestamps: List[pd.Timestamp]
    threshold_violations: List[Tuple[pd.Timestamp, pd.Timestamp, float, bool]]
    utilization_pattern: str


@dataclass
class CapacityForecastResult:
    resource_metrics: Dict[str, ResourceMetrics]
    resource_type: ResourceType
    forecast_period: Dict[str, Optional[pd.Timestamp]]
    thresholds_used: ResourceThresholds
    forecast_method: ForecastMethod


class CapacityPlanning(BaseModule):
    """
    Capacity Planning Module

    This module analyzes system resource utilization to forecast future usage and provide
    capacity planning recommendations. It uses historical trace data to:
        1. Track current resource utilization patterns
        2. Forecast future resource usage
        3. Identify potential capacity bottlenecks
    """

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the capacity planning module.

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

        self.combined_df: Optional[pd.DataFrame] = None
        self.thresholds: ResourceThresholds = ResourceThresholds()
        self.scalers: Dict[str, StandardScaler] = {}

        self.logger.info("Initializing Capacity Planning module.")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        if not outputs:
            outputs = self.experiment.find_outputs(keyword=["cpu usage", "memory usage", "disk"],
                                                   type=["xy"], match_any=True)

        super()._process(outputs=outputs,
                         normalize=False,
                         resample=True,
                         align_timestamps=True,
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        """Post-process the data after loading."""
        if not self.dataframes:
            return

        for key, df in self.dataframes.items():
            df.columns = [key]
        self.combined_df = self.data_preprocessor.combine_dataframes(list(self.dataframes.values()))

        for column in list(self.combined_df.columns):
            self.combined_df[f"{column}_original"] = self.combined_df[column]

            scaler = StandardScaler()
            self.combined_df[column] = scaler.fit_transform(np.array(self.combined_df[column]).reshape(-1, 1))
            self.scalers[column] = scaler

    def _analyze_utilization_pattern(self, series: pd.Series) -> str:
        """
        Analyze the utilization pattern of a resource.

        :param series: Time series data of resource usage
        :type series: pd.Series
        :return: Description of the utilization pattern
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

    def _get_formatted_resource_property(self, resource_property: float, resource_type: ResourceType, separate: bool = False) -> str:
        """
        Format resource property based on its type.

        :param resource_property: Resource property value
        :type resource_property: float
        :param resource_type: Type of resource (CPU, Memory, Disk)
        :type resource_type: ResourceType
        :param separate: Whether to separate the value and unit
        :type separate: bool
        :return: Formatted threshold string
        :rtype: str
        """
        output = ""
        if resource_type.name == ResourceType.CPU.name:
            output = f"{resource_property:.2f} %"
        else:
            val, unit = Formatter.format_bytes(resource_property)
            output = f"{val:.2f} {unit}"

            if resource_type.name == ResourceType.DISK.name:
                output += "/s"

        return output if separate else output.replace(" ", "")

    def _calculate_pdq(self, series: pd.Series, max_lag: int = 20) -> Tuple[int, int, int]:
        """
        Determine the optimal AR (p), I (d), and MA (q) parameters for an ARIMA model.

        :param series: Input time series data
        :type series: pd.Series
        :param max_lag: Maximum lag to consider for ACF and PACF
        :type max_lag: int
        :return: Optimal p, d, and q parameters
        :rtype: Tuple[int, int, int]
        """
        p, d, q = 0, 0, 0

        test_series = series.copy()
        while True:
            if test_series.nunique() == 1:
                break

            adf_result = adfuller(test_series)
            if adf_result[1] < 0.05:  # is stationary
                break

            # Detrend the series
            test_series = test_series.diff().dropna()
            d += 1

        series = series.diff(d).dropna() if d > 0 else series

        acf_values = acf(series, nlags=max_lag, fft=True)
        pacf_values = pacf(series, nlags=max_lag)

        # Determine p (PACF cutoff)
        p = next((lag for lag, value in enumerate(pacf_values) if abs(value) < 0.2), max_lag)

        # Determine q (ACF cutoff)
        q = next((lag for lag, value in enumerate(acf_values) if abs(value) < 0.2), max_lag)

        return p, d, q

    def _get_series_freq(self, series: pd.Series) -> str:
        """
        Get the frequency of a time series.

        :param series: Input time series data
        :type series: pd.Series
        :return: Frequency string
        :rtype: str
        """
        freq = pd.infer_freq(cast(pd.DatetimeIndex, series.index))
        if freq is None or not isinstance(freq, str):
            freq = "1s"
        if not freq[0].isdigit():
            freq = "1" + freq

        return freq

    def _forecast_arima(self, series: pd.Series, forecast_steps: int) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Forecast future values using ARIMA model with dynamic parameters.

        :param series: Input time series data
        :type series: pd.Series
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Tuple of forecast values and timestamps
        :rtype: Tuple[List[float], List[pd.Timestamp]]
        """
        freq = self._get_series_freq(series)
        p, d, q = self._calculate_pdq(series)

        try:
            model = ARIMA(series, order=(p, d, q))
            results = model.fit()

            forecast = results.forecast(steps=forecast_steps)

            last_timestamp = cast(pd.Timestamp, series.index[-1])
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(freq),
                periods=forecast_steps,
                freq=freq
            )

            return forecast.tolist(), future_timestamps.tolist()
        except Exception as e:
            self.logger.error(f"ARIMA forecast failed")
            return [], []

    def _forecast_var(self, data: pd.DataFrame, forecast_steps: int) -> Dict[str, Tuple[List[float], List[pd.Timestamp]]]:
        """
        Forecast future values using VAR model.

        :param data: Input multivariate time series data
        :type data: pd.DataFrame
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Dictionary of forecasts for each variable
        :rtype: Dict[str, Tuple[List[float], List[pd.Timestamp]]]
        """
        freq = self._get_series_freq(data.iloc[:, 0])

        model = VAR(data)
        results = model.fit()

        forecast = results.forecast(data.values, steps=forecast_steps)

        last_timestamp = cast(pd.Timestamp, data.index[-1])
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=forecast_steps,
            freq=freq
        )

        forecasts = {}
        for i, column in enumerate(data.columns):
            forecasts[column] = (forecast[:, i].tolist(), future_timestamps.tolist())

        return forecasts

    def _forecast_moving_average(self, series: pd.Series, forecast_steps: int, window_size: int) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Forecast future values using moving average method.

        :param series: Input time series data
        :type series: pd.Series
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :param window_size: Window size for moving average
        :type window_size: int
        :return: Tuple of forecast values and timestamps
        :rtype: Tuple[List[float], List[pd.Timestamp]]
        """
        freq = self._get_series_freq(series)

        forecast = series.rolling(window=window_size).mean().iloc[-forecast_steps:]
        last_timestamp = cast(pd.Timestamp, series.index[-1])
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=forecast_steps,
            freq=freq
        )

        return forecast.tolist(), future_timestamps.tolist()

    def _detect_threshold_violations(self, values: List[float], timestamps: List[pd.Timestamp],
                                     threshold: float, window_size: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, float, bool]]:
        """
        Detect periods where the forecast exceeds the threshold.

        :param values: Forecasted values
        :type values: List[float]
        :param timestamps: Corresponding timestamps
        :type timestamps: List[pd.Timestamp]
        :param threshold: Threshold value
        :type threshold: float
        :param window_size: Window size for threshold detection
        :type window_size: float
        :return: List of threshold violation periods
        :rtype: List[Tuple[pd.Timestamp, float]]
        """
        window_seconds = Formatter.parse_time_to_seconds(window_size)
        violations = []

        violation_start = None
        violation_max = float('-inf')

        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            if value > threshold:
                # Start or continue a violation period
                if violation_start is None:
                    violation_start = timestamp
                violation_max = max(violation_max, value)

                # If this is the last point or next point doesn't exceed threshold, end the current violation period
                is_last_point = i == len(values) - 1
                next_point_violates = (not is_last_point and values[i + 1] > threshold)

                if is_last_point or not next_point_violates:
                    violation_duration = (timestamp - violation_start).total_seconds()
                    is_sustained = violation_duration >= window_seconds

                    violations.append((
                        violation_start,
                        timestamp,
                        violation_max,
                        is_sustained
                    ))

                    violation_start = None
                    violation_max = float('-inf')

        return violations

    def forecast_capacity(self, resource_types: Optional[List[ResourceType]] = None,
                          method: str = ForecastMethod.ARIMA.name,
                          forecast_steps: int = 100,
                          **kwargs) -> Dict[ResourceType, CapacityForecastResult]:
        """
        Forecast future resource utilization and identify potential capacity issues.

        :param resource_types: List of resource types to analyze
        :type resource_types: Optional[List[ResourceType]]
        :param method: Forecasting method to use (e.g., ARIMA, VAR, Moving Average)
        :type method: ForecastMethod
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Forecast results for each resource type
        :rtype: Dict[ResourceType, CapacityForecastResult]
        """
        if self.combined_df is None or self.combined_df.empty:
            self.logger.warning("No data available for capacity planning analysis")
            return {}

        # Check if method is present in the forecast methods
        if not hasattr(ForecastMethod, method.upper()):
            self.logger.warning(f"Invalid forecast method: {method}. Using default ARIMA.")
            method = ForecastMethod.ARIMA.name
        method = method.upper()

        if resource_types is None:
            resource_types = list(ResourceType)

        self.thresholds = ResourceThresholds(
            cpu_threshold=kwargs.get("cpu_threshold", ResourceThresholds.cpu_threshold),
            memory_threshold=kwargs.get("memory_threshold", ResourceThresholds.memory_threshold),
            disk_threshold=kwargs.get("disk_threshold", ResourceThresholds.disk_threshold),
            warning_period=kwargs.get("warning_period", ResourceThresholds.warning_period)
        )

        results = {}

        all_resource_columns = []
        resource_columns_by_type = {}
        for resource_type in resource_types:
            columns = [
                col for col in self.combined_df.columns
                if (
                    (resource_type == ResourceType.CPU and "cpu" in col.lower()) or
                    (resource_type == ResourceType.MEMORY and "memory" in col.lower()) or
                    (resource_type == ResourceType.DISK and "disk" in col.lower())
                ) and not col.endswith("_original")
            ]
            if columns:
                all_resource_columns.extend(columns)
                resource_columns_by_type[resource_type] = columns
            else:
                self.logger.warning(f"No {resource_type.name} resources found in data")

        if not all_resource_columns:
            self.logger.warning("No resources found in data")
            return {}

        # Generate forecasts based on method
        resource_metrics = {}
        if method in [ForecastMethod.ARIMA.name, ForecastMethod.MOVING_AVERAGE.name]:
            # Generate individual ARIMA or Moving Average forecasts for each column
            for name in all_resource_columns:
                series = self.combined_df[name]
                if method == ForecastMethod.ARIMA.name:
                    forecast_values, forecast_timestamps = self._forecast_arima(series, forecast_steps)
                else:
                    window_size = kwargs.get("window_size", 5)
                    forecast_values, forecast_timestamps = self._forecast_moving_average(series, forecast_steps, window_size)
                resource_metrics[name] = (forecast_values, forecast_timestamps)
        elif method == ForecastMethod.VAR.name:
            # Generate VAR forecasts for all columns at once
            data = self.combined_df[all_resource_columns]
            var_forecasts = self._forecast_var(data, forecast_steps)
            resource_metrics = var_forecasts

        for resource_type, columns in resource_columns_by_type.items():
            if resource_type == ResourceType.CPU:
                threshold = self.thresholds.cpu_threshold
            elif resource_type == ResourceType.MEMORY:
                threshold = self.thresholds.memory_threshold
            else:
                threshold = self.thresholds.disk_threshold

            type_metrics = {}
            forecast_timestamps = []
            for name in columns:
                original_series = self.combined_df[f"{name}_original"]
                forecast_values, forecast_timestamps = resource_metrics[name]

                forecast_values = self.scalers[name].inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten().tolist()  # type: ignore

                violations = self._detect_threshold_violations(
                    forecast_values,
                    forecast_timestamps,
                    threshold,
                    self.thresholds.warning_period
                )

                type_metrics[name] = ResourceMetrics(
                    current_usage=original_series.iloc[-1],
                    peak_usage=original_series.max(),
                    average_usage=original_series.mean(),
                    forecast_values=forecast_values,
                    forecast_timestamps=forecast_timestamps,
                    threshold_violations=violations,
                    utilization_pattern=self._analyze_utilization_pattern(original_series)
                )

            results[resource_type] = CapacityForecastResult(
                resource_metrics=type_metrics,
                resource_type=resource_type,
                forecast_period={
                    "start": forecast_timestamps[0],
                    "end": forecast_timestamps[-1]
                },
                thresholds_used=self.thresholds,
                forecast_method=ForecastMethod[method.upper()]
            )

        return results

    def interpret(self, forecast_results: Dict[ResourceType, CapacityForecastResult]) -> None:
        """
        Interpret capacity planning analysis results and provide recommendations.

        :param forecast_results: Results from capacity forecast analysis
        :type forecast_results: Dict[ResourceType, CapacityForecastResult]
        """
        DocumentGenerator.section("Capacity Planning Analysis Results")

        for resource_type, result in forecast_results.items():
            DocumentGenerator.section(f"{resource_type.name} Capacity Analysis")

            parameters = {
                "Analysis Period Start": result.forecast_period["start"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.forecast_period["start"] else "N/A",
                "Analysis Period End": result.forecast_period["end"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.forecast_period["end"] else "N/A",
                "Forecast Method": result.forecast_method.name,
            }

            parameters[f"{resource_type.name} Threshold"] = self._get_formatted_resource_property(
                result.thresholds_used.cpu_threshold if resource_type.name == ResourceType.CPU.name else
                result.thresholds_used.memory_threshold if resource_type.name == ResourceType.MEMORY.name else
                result.thresholds_used.disk_threshold,
                resource_type
            )

            DocumentGenerator.metrics_group("Analysis Parameters", parameters)

            # Resource metrics
            for resource_id, metrics in result.resource_metrics.items():
                resource_metrics = {
                    "Current Usage": self._get_formatted_resource_property(metrics.current_usage, resource_type),
                    "Peak Usage": self._get_formatted_resource_property(metrics.peak_usage, resource_type),
                    "Average Usage": self._get_formatted_resource_property(metrics.average_usage, resource_type),
                    "Utilization Pattern": metrics.utilization_pattern
                }

                if resource_type.name != ResourceType.CPU.name:
                    for key in ["Current Usage", "Peak Usage", "Average Usage"]:
                        resource_metrics[key] = self._get_formatted_resource_property(metrics.__dict__[key.lower().replace(" ", "_")],
                                                                                      resource_type)

                if metrics.threshold_violations:
                    next_violation = metrics.threshold_violations[0]
                    resource_metrics["Next Threshold Violation"] = (
                        f"{next_violation[0].strftime('%Y-%m-%d %H:%M:%S.%f')} "
                    )
                    resource_metrics["Total Violations"] = str(len(metrics.threshold_violations))

                output = self.experiment.get_output_by_id(resource_id)
                resource_name = output.name if output else resource_id
                DocumentGenerator.metrics_group(f"Resource: {resource_name}", resource_metrics)

                if metrics.threshold_violations:
                    # Sort violations by duration
                    metrics.threshold_violations.sort(key=lambda x: (x[1] - x[0]).total_seconds(), reverse=True)

                    violation_headers = ["Start", "End", "Duration", "Forecasted Usage"]
                    num_sig_violations = min(5, len(metrics.threshold_violations))
                    violation_rows = []
                    for i in range(num_sig_violations):
                        start_time, end_time, max_usage, _ = metrics.threshold_violations[i]
                        duration = (end_time - start_time).total_seconds()
                        if duration == 0:
                            continue

                        time_val, time_unit = Formatter.format_seconds(duration)
                        max_usage = self._get_formatted_resource_property(max_usage, resource_type)

                        violation_rows.append([start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                               end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                               f"{time_val:.2f} {time_unit}", max_usage])
                    DocumentGenerator.table(
                        violation_headers,
                        violation_rows,
                        f"Top Threshold Violations for {resource_name}"
                    )

        # Generate recommendations
        DocumentGenerator.section("Capacity Planning Recommendations")

        DEFAULT_RECOMMENDATION = "Resource utilization is within acceptable limits. No immediate action required."
        recommendations = {
            "Immediate Actions": [],
            "Short-term Planning": [],
            "Long-term Strategy": [],
            "Optimization Opportunities": []
        }

        for resource_type, result in forecast_results.items():
            for resource_id, metrics in result.resource_metrics.items():
                average_usage = metrics.average_usage
                peak_usage = metrics.peak_usage
                pattern = metrics.utilization_pattern
                violations = metrics.threshold_violations

                output = self.experiment.get_output_by_id(resource_id)
                resource_name = output.name if output else resource_id

                threshold = (
                    result.thresholds_used.cpu_threshold if resource_type.name == ResourceType.CPU.name
                    else result.thresholds_used.memory_threshold if resource_type.name == ResourceType.MEMORY.name
                    else result.thresholds_used.disk_threshold
                )

                if violations:
                    if self.combined_df is None or self.combined_df.empty:
                        continue

                    time_to_violation = (violations[0][0] - self.combined_df.index[-1]).total_seconds()

                    # Immediate actions for imminent violations
                    if time_to_violation < 3600:  # Less than 1 hour
                        time_val, time_unit = Formatter.format_seconds(time_to_violation)
                        exceed_value = self._get_formatted_resource_property(threshold, resource_type)
                        if resource_type.name == ResourceType.CPU.name:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit}. "
                                f"Consider immediate load balancing or scaling up CPU capacity."
                            )
                        elif resource_type.name == ResourceType.MEMORY.name:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit} "
                                f"Consider freeing up memory or increasing available memory."
                            )
                        else:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit}. "
                                f"Consider cleanup or adding storage capacity."
                            )

                    # Short-term planning for upcoming violations
                    elif time_to_violation < 86400:  # Less than 24 hours
                        time_val, time_unit = Formatter.format_seconds(time_to_violation)
                        exceed_value = self._get_formatted_resource_property(threshold, resource_type)
                        if resource_type.name == ResourceType.CPU.name:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit} "
                                f"Plan for CPU capacity increase or workload redistribution."
                            )
                        elif resource_type.name == ResourceType.MEMORY.name:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit} "
                                f"Plan for memory upgrade or optimization."
                            )
                        else:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {exceed_value} in {time_val:.2f} {time_unit}. "
                                f"Plan for storage expansion or archival."
                            )

                # Long-term strategy based on utilization patterns
                if pattern == "Highly variable":
                    recommendations["Long-term Strategy"].append(
                        f"{resource_name} shows highly variable usage patterns. "
                        f"Consider implementing auto-scaling or dynamic resource allocation."
                    )

                # Long-term strategy based on peak usage
                if peak_usage > threshold * 0.9:
                    peak_usage_str = self._get_formatted_resource_property(peak_usage, resource_type)
                    threshold_str = self._get_formatted_resource_property(threshold, resource_type)

                    recommendations["Long-term Strategy"].append(
                        f"{resource_name} has peak usage ({peak_usage_str}) approaching or bypassing the threshold ({threshold_str}). "
                        f"Consider increasing capacity or implementing load balancing."
                    )

                # Optimization opportunities
                if pattern == "Stable" and average_usage < threshold * 0.3:  # Less than 30% of threshold
                    recommendations["Optimization Opportunities"].append(
                        f"{resource_name} is consistently underutilized. "
                        f"Consider resource consolidation or downsizing."
                    )

        for category, items in recommendations.items():
            if not items:
                items = [DEFAULT_RECOMMENDATION]
            items = {f"{idx + 1}": item for idx, item in enumerate(items)}
            DocumentGenerator.metrics_group(category, items)

    def plot_capacity_forecast(self, forecast_results: Dict[ResourceType, CapacityForecastResult],
                               zoomed: bool = False, **kwargs) -> None:
        """
        Plot capacity forecast analysis results.

        :param forecast_results: Results from capacity forecast analysis
        :type forecast_results: Dict[ResourceType, CapacityForecastResult]
        :param zoomed: Whether to zoom in on the forecast period (show a smaller time range of original data)
        :type zoomed: bool
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if not forecast_results:
            self.logger.warning("No forecast results to plot")
            return

        if self.combined_df is None or self.combined_df.empty:
            self.logger.warning("No data available for plotting")
            return

        fig_size = kwargs.get("fig_size", (15, 5))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab10")

        for idx, (resource_type, result) in enumerate(forecast_results.items()):
            for resource_id, metrics in result.resource_metrics.items():
                historical_data = self.combined_df[f"{resource_id}_original"]

                # Get the last 10% of the data for zoomed view
                if zoomed:
                    historical_data = historical_data.iloc[-int(len(historical_data) * 0.1):]

                forecast_data = pd.Series(
                    metrics.forecast_values,
                    index=metrics.forecast_timestamps,
                    name=resource_id
                )

                plots = []

                # Plot historical data
                plots.append({
                    "plot_type": "time_series",
                    "data": historical_data,
                    "label": "Historical",
                    "color": colors(idx % 10),
                    "alpha": 0.5,
                    "linewidth": 2
                })

                # Connect the end of historical data to the start of forecast data
                if not historical_data.empty and not forecast_data.empty:
                    connection_data = pd.Series(
                        [historical_data.iloc[-1], forecast_data.iloc[0]],
                        index=[historical_data.index[-1], forecast_data.index[0]]
                    )
                    plots.append({
                        "plot_type": "time_series",
                        "data": connection_data,
                        "label": None,
                        "color": colors(idx % 10),
                        "alpha": 1,
                        "linewidth": 2,
                    })

                # Plot forecast
                plots.append({
                    "plot_type": "time_series",
                    "data": forecast_data,
                    "label": "Forecast",
                    "color": colors(idx % 10),
                    "alpha": 1,
                    "linewidth": 2,
                    "linestyle": "--"
                })

                # Add threshold line
                if resource_type.name == ResourceType.CPU.name:
                    threshold_line = result.thresholds_used.cpu_threshold
                elif resource_type.name == ResourceType.MEMORY.name:
                    threshold_line = result.thresholds_used.memory_threshold
                else:
                    threshold_line = result.thresholds_used.disk_threshold

                threshold_str = self._get_formatted_resource_property(threshold_line, resource_type)

                plots.append({
                    "plot_type": "hline",
                    "y": threshold_line,
                    "color": "red",
                    "linestyle": "--",
                    "label": f"Threshold ({threshold_str})",
                    "alpha": 0.75
                })

                # Highlight threshold violations
                for violation_start, violation_end, _, is_sustained in metrics.threshold_violations:
                    if not is_sustained:
                        continue

                    plots.append({
                        "plot_type": "span",
                        "start": violation_start,
                        "end": violation_end,
                        "color": "red",
                        "alpha": 0.2
                    })

                usage_str = self._get_formatted_resource_property(metrics.average_usage, resource_type)
                peak_str = self._get_formatted_resource_property(metrics.peak_usage, resource_type)

                output = self.experiment.get_output_by_id(resource_id)
                resource_name = output.name if output else resource_id
                title = (
                    f"{resource_type.name} Capacity Forecast: {resource_name}\n"
                    f"(Average Usage: {usage_str}, "
                    f"Peak: {peak_str}, "
                    f"Pattern: {metrics.utilization_pattern})"
                )

                self._plot(
                    plots,
                    plot_size=fig_size,
                    dpi=fig_dpi,
                    grid=True,
                    fig_title=title,
                    fig_xlabel="Time",
                    fig_ylabel=f"Usage",
                    fig_num_yticks=8
                )
