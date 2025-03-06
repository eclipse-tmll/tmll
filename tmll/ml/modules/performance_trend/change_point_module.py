from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures
from typing import Any, List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient

# Disable warnings ruptures costnormal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ruptures.costs.costnormal")


@dataclass
class ChangePointAnalysisResult:
    @dataclass
    class MetricResult:
        change_points: List[int]
        magnitudes: List[float]
        kwargs: Dict[str, Any]

    metrics: Dict[str, MetricResult]


class ChangePointAnalysis(BaseModule):
    """
    Change point detection module for analyzing performance trends.

    Basically, this module aims to indicate the significant changes that happens
    in the performance metrics of a system (e.g., CPU usage, memory usage, etc.).

    The module supports multiple methods for change point detection:
    - Single metric analysis: Detects change points in individual metrics (e.g., CPU usage)
    - Z-score analysis: Combines metrics using z-score normalization and detects change points
    - Voting-based analysis: Combines change points from individual metrics using voting
    - PCA analysis: Uses principal component analysis to detect change points
    """

    AVAILABLE_METRICS = ["single", "zscore", "voting", "pca"]

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the Change Point Analysis module.

        :param client: The TMLL client for data communication.
        :type client: TMLLClient
        :param experiment: The experiment to analyze.
        :type experiment: Experiment
        :param outputs: The list of outputs to analyze.
        :type outputs: Optional[List[Output]]
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        """
        super().__init__(client, experiment)

        self.timestamps: Optional[pd.DatetimeIndex] = None
        self.combined_df: Optional[pd.DataFrame] = None
        self.best_params: Dict[str, Any] = {}

        self.logger.info("Initializing Change Point Analysis module.")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        super()._process(outputs=outputs,
                         normalize=False,
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        # Combine dataframes
        normalized_dataframes = [self.data_preprocessor.normalize(df) for df in self.dataframes.values()]
        self.combined_df = self.data_preprocessor.combine_dataframes(normalized_dataframes)

    def _calculate_changes_magnitude(self, data: np.ndarray, change_points: List[int]) -> List[float]:
        """
        Calculate magnitude of changes at change points

        :param data: Input time series data
        :type data: np.ndarray
        :param change_points: Detected change points
        :type change_points: List[int]
        :return: Magnitudes of changes
        :rtype: List[float]
        """
        magnitudes = []

        for cp in change_points:
            # Calculate window size based on data
            window_size = min(self.window_size, cp, len(data) - cp)

            # Get segments before and after change point
            before_segment = data[max(0, cp - window_size):cp]
            after_segment = data[cp:min(len(data), cp + window_size)]

            if len(before_segment) == 0 or len(after_segment) == 0:
                continue

            # Calculate statistical properties
            before_mean = np.mean(before_segment)
            after_mean = np.mean(after_segment)
            before_std = np.std(before_segment)
            after_std = np.std(after_segment)

            # Calculate magnitude components
            mean_change = abs(after_mean - before_mean)
            var_change = abs(after_std - before_std)
            snr = mean_change / max(before_std, after_std) if max(before_std, after_std) > 0 else mean_change

            # Combined magnitude score
            magnitude = self._calculate_significance(float(mean_change), float(var_change))
            magnitudes.append(magnitude)

        magnitudes = [magnitude for magnitude in magnitudes if abs(magnitude) > 0]

        return magnitudes

    def _calculate_statistical_metrics(self, segment_before: np.ndarray, segment_after: np.ndarray) -> Tuple[float, float]:
        """
        Calculate statistical metrics for segments before and after a change point.
        The statistical metrics are: mean change and variance change

        :param segment_before: Data segment before change point
        :type segment_before: np.ndarray
        :param segment_after: Data segment after change point
        :type segment_after: np.ndarray
        :return: Tuple of (mean_change, var_change)
        :rtype: Tuple[float, float]
        """
        if len(segment_before) == 0 or len(segment_after) == 0:
            return 0.0, 0.0

        before_mean = np.mean(segment_before)
        after_mean = np.mean(segment_after)
        before_std = np.std(segment_before)
        after_std = np.std(segment_after)

        mean_change = abs(after_mean - before_mean)
        var_change = abs(after_std - before_std)

        return float(mean_change), float(var_change)

    def _get_segments(self, data: np.ndarray, change_point: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get segments before and after a change point.

        :param data: Input time series data
        :type data: np.ndarray
        :param change_point: Change point index
        :type change_point: int
        :return: Tuple of (before_segment, after_segment)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        window_size = min(self.window_size, change_point, len(data) - change_point)
        before_segment = data[max(0, change_point - window_size):change_point]
        after_segment = data[change_point:min(
            len(data), change_point + window_size)]
        return before_segment, after_segment

    def _calculate_significance(self, mean_change: float, var_change: float) -> float:
        """
        Calculate significance score from statistical metrics.

        :param mean_change: Change in mean
        :type mean_change: float
        :param var_change: Change in variance
        :type var_change: float
        :return: Combined significance score
        :rtype: float
        """
        return (0.6 * mean_change) + (0.4 * var_change)

    def _detect_changes(self, data: np.ndarray, n_change_points: int = 5,
                        tune_hyperparameters: bool = True, method_key: Optional[str] = None) -> List[int]:
        """
        Detect the most significant change points in order of their importance.
        We want to detect the most significant change points in the data.

        :param data: Input time series data
        :type data: np.ndarray
        :param n_change_points: Number of change points to detect
        :type n_change_points: int
        :param method_key: Key for storing method-specific parameters
        :type method_key: Optional[str]
        :return: Detected change points
        :rtype: List[int]
        """
        # Tune parameters if needed
        if tune_hyperparameters and method_key and method_key not in self.best_params:
            self.best_params[method_key] = self._tune_hyperparameters(data, n_change_points)

        params = self.best_params.get(method_key, {}) if method_key else {}
        model_type = params.get("model", "rbf")
        min_size = params.get("min_size", 5)
        jump = params.get("jump", 2)

        # Create and fit model
        model = ruptures.Binseg(model=model_type, min_size=min_size, jump=jump)
        model.fit(data.reshape(-1, 1))

        # Get extra change points for significance evaluation
        # We want to ensure that we have enough points to evaluate significance
        n_extra = min(len(data) // 2, n_change_points * 5)
        all_change_points = model.predict(n_bkps=n_extra)[:-1]

        if not all_change_points:
            return []

        # Calculate significance for each point
        significance_scores = []
        for cp in all_change_points:
            before_segment, after_segment = self._get_segments(data, cp)
            if len(before_segment) == 0 or len(after_segment) == 0:
                continue

            mean_change, var_change = self._calculate_statistical_metrics(before_segment, after_segment)
            significance = self._calculate_significance(mean_change, var_change)
            significance_scores.append((cp, significance))

        # Select top points by significance
        significant_points = sorted(significance_scores, key=lambda x: x[1], reverse=True)
        return sorted(point[0] for point in significant_points[:n_change_points])

    def get_change_points(self, metrics: Optional[List[str]] = None,
                          n_change_points: int = 3,
                          methods: Optional[List[str]] = None,
                          min_size: int = 5,
                          window_size: int = 5,
                          tune_hyperparameters: bool = True) -> Optional[ChangePointAnalysisResult]:
        """
        Find the most significant change points in the data using multiple analysis methods.
        The methods include:
        - Single metric analysis: Detects change points in individual metrics (e.g., CPU usage)
        - Z-score analysis: Combines metrics using z-score normalization and detects change points (combined score)
        - Voting-based analysis: Combines change points from individual metrics using voting (majority voting)
        - PCA analysis: Uses principal component analysis to detect change points (principal component)

        :param metrics: List of metrics to analyze (None for all metrics)
        :type metrics: Optional[List[str]]
        :param n_change_points: Number of change points to detect
        :type n_change_points: int
        :param methods: Analysis methods to use (e.g., "single", "zscore", "voting", "pca")
        :type methods: Optional[List[str]]
        :param min_size: Minimum size of a segment
        :type min_size: int
        :param window_size: Window size for calculating change magnitude
        :type window_size: int
        :param tune_hyperparameters: Whether to tune hyperparameters for change point detection
        :type tune_hyperparameters: bool
        :return: Analysis results for each method
        :rtype: Optional[ChangePointAnalysisResult]
        """
        if self.combined_df is None:
            self.logger.error("No data available for analysis")
            return None

        self.min_size = min_size
        self.window_size = window_size

        results = ChangePointAnalysisResult(metrics={})

        if metrics:
            for idx, metric in enumerate(metrics):
                output = self.experiment.get_output_by_name(metric)
                if output is not None:
                    metrics[idx] = output.id
                else:
                    metrics[idx] = "unknown"
            metrics = [metric for metric in metrics if metric != "unknown"]

        metrics_to_analyze = metrics if metrics else list(self.dataframes.keys())

        if not methods:
            methods = self.AVAILABLE_METRICS

        # Single metric analysis
        if "single" in methods:
            for metric in metrics_to_analyze:
                if metric not in self.dataframes:
                    self.logger.warning(f"Metric {metric} not found")
                    continue

                data = self.dataframes[metric].iloc[:, 0].values
                change_points = self._detect_changes(data=np.array(data),
                                                     n_change_points=n_change_points,
                                                     tune_hyperparameters=tune_hyperparameters,
                                                     method_key=f"single_{metric}")
                magnitudes = self._calculate_changes_magnitude(np.array(data), change_points)
                if len(change_points) > 0:
                    results.metrics[metric] = ChangePointAnalysisResult.MetricResult(
                        change_points=change_points,
                        magnitudes=magnitudes,
                        kwargs={})

        # Z-score analysis
        if "zscore" in methods:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.combined_df[metrics_to_analyze])
            combined_zscore = np.sqrt(np.mean(np.square(scaled_data), axis=1))

            change_points = self._detect_changes(data=combined_zscore,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key="zscore")
            magnitudes = self._calculate_changes_magnitude(combined_zscore, change_points)
            if len(change_points) > 0:
                results.metrics["zscore"] = ChangePointAnalysisResult.MetricResult(
                    change_points=change_points,
                    magnitudes=magnitudes,
                    kwargs={
                        "combined_score": combined_zscore
                    })

        # Voting-based analysis
        if "voting" in methods:
            vote_matrix = np.zeros(len(self.combined_df))
            individual_changes = {}

            for metric in metrics_to_analyze:
                data = self.combined_df[metric].values
                changes = self._detect_changes(data=np.array(data),
                                               n_change_points=n_change_points,
                                               tune_hyperparameters=tune_hyperparameters,
                                               method_key=f"voting_{metric}")
                individual_changes[metric] = changes

                for cp in changes:
                    window = slice(max(0, cp - self.window_size), min(len(vote_matrix), cp + self.window_size + 1))
                    vote_matrix[window] += 1

            change_points = self._detect_changes(data=vote_matrix,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key="voting")
            magnitudes = self._calculate_changes_magnitude(vote_matrix, change_points)
            if len(magnitudes) > 0:
                results.metrics["voting"] = ChangePointAnalysisResult.MetricResult(
                    change_points=change_points,
                    magnitudes=magnitudes,
                    kwargs={
                        "individual_changes": individual_changes,
                        "vote_matrix": vote_matrix
                    })

        # PCA analysis
        if "pca" in methods:
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(self.combined_df[metrics_to_analyze]).flatten()

            change_points = self._detect_changes(data=principal_component,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key="pca")
            magnitudes = self._calculate_changes_magnitude(principal_component, change_points)
            if len(magnitudes) > 0:
                results.metrics["pca"] = ChangePointAnalysisResult.MetricResult(
                    change_points=change_points,
                    magnitudes=magnitudes,
                    kwargs={
                        "principal_component": principal_component,
                        "explained_variance_ratio": pca.explained_variance_ratio_[0]
                    })

        return results

    def _tune_hyperparameters(self, data: np.ndarray, n_change_points: int,
                              param_grid: Optional[Dict] = None, cv_splits: int = 5, deep_search: bool = False) -> Dict:
        """
        Tune hyperparameters for change point detection using cross-validation.

        :param data: Input time series data
        :type data: np.ndarray
        :param param_grid: Parameter grid for hyperparameter tuning
        :type param_grid: Optional[Dict]
        :param cv_splits: Number of cross-validation splits
        :type cv_splits: int
        :param deep_search: Perform a deeper search for hyperparameters
        :type deep_search: bool
        :return: Best hyperparameters
        :rtype: Dict
        """
        if param_grid is None:
            param_grid = {
                "model": ["rbf", "linear", "normal", "cosine"],  # Model type
                "min_size": [5, 10, 20],  # Minimum size of a segment
                "window_size": [3, 5],  # Window size for calculating change magnitude
                "jump": [2, 3, 4]  # Jump parameter
            }

        if deep_search:
            param_grid["min_size"] = [5, 10, 15, 20, 25, 30, 35, 40]
            param_grid["window_size"] = [3, 5, 7, 10, 15]
            param_grid["jump"] = [2, 3, 4, 5, 6]

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        best_score = float("inf")
        best_params = {}

        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

        for params in param_combinations:
            scores = []

            for train_idx, test_idx in tscv.split(data):
                train_data = data[train_idx]
                test_data = data[test_idx]

                model = ruptures.Binseg(
                    model=params["model"],
                    min_size=params["min_size"],
                    jump=params["jump"])

                try:
                    model.fit(train_data.reshape(-1, 1))
                    bkps = model.predict(n_bkps=n_change_points)[:-1]

                    error = self._calculate_prediction_error(test_data, bkps)
                    scores.append(error)
                except Exception as e:
                    self.logger.warning(f"Error with parameters {params}: {str(e)}")
                    scores.append(float("inf"))

            mean_score = np.mean(scores) if len(scores) > 0 else float("inf")

            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        self.best_params = best_params
        return best_params

    def _calculate_prediction_error(self, data: np.ndarray, change_points: List[int]) -> float:
        """
        Calculate prediction error for a given set of change points.
        Uses mean squared error between actual values and piecewise constant approximation.

        :param data: Input time series data
        :type data: np.ndarray
        :param change_points: Detected change points
        :type change_points: List[int]
        :return: Prediction error
        :rtype: float
        """
        if len(change_points) < 2:
            return float("inf")

        error = 0
        start_idx = 0

        for end_idx in change_points[:-1]:
            segment = data[start_idx:end_idx]
            segment_mean = np.mean(segment)
            error += np.sum((segment - segment_mean) ** 2)
            start_idx = end_idx

        return float(error / len(data))

    def plot_change_points(self, results: Optional[ChangePointAnalysisResult] = None, **kwargs) -> None:
        """
        Plot change points and analysis results.

        :param results: Results from `get_change_points` method
        :type results: ChangePointAnalysisResult
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if results is None:
            self.logger.error("No results provided")
            return

        if self.combined_df is None or self.combined_df.empty:
            self.logger.error("Combined DataFrame is None")
            return

        fig_size = kwargs.get("fig_size", (15, 3))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab10")

        def _get_metric_data(metric: str) -> Tuple[str, Optional[pd.Series]]:
            match metric:
                case "zscore":
                    return "Combined z-score Analysis", \
                        results.metrics[metric].kwargs.get("combined_score") if (
                            "zscore" in results.metrics and results.metrics["zscore"].kwargs is not None) else None
                case "voting":
                    return "Voting-based Analysis", \
                        results.metrics[metric].kwargs.get("vote_matrix", None) if (
                            "voting" in results.metrics and results.metrics["voting"].kwargs is not None) else None
                case "pca":
                    return "PCA Analysis", \
                        results.metrics[metric].kwargs.get("principal_component", None) if (
                            "pca" in results.metrics and results.metrics["pca"].kwargs is not None) else None
                case _:
                    return f"Change Points for {metric}", \
                        self.dataframes[metric].iloc[:, 0] if metric in self.dataframes else None

        for idx, (metric, result) in enumerate(results.metrics.items()):
            plots = []
            title, series = _get_metric_data(metric)
            output = self.experiment.get_output_by_id(metric)
            metric_name = output.name if output else metric
            title = title.replace(metric, metric_name)

            # Add individual metric plots
            plots.append({
                "plot_type": "time_series",
                "data": pd.DataFrame({metric_name: series}, index=self.combined_df.index),
                "title": metric_name,
                "label": metric_name,
                "y": metric_name,
                "color": colors(idx % 10),
                "linewidth": 1.5
            })

            # Add change points to the plot
            for cp, mag in zip(result.change_points, result.magnitudes):
                change_time = self.combined_df.index[cp]

                # Normalize magnitude based on the metric
                if series is not None and np.max(series) != 0:
                    mag = mag / np.max(series)

                # Add change point to plot
                plots.append({
                    "plot_type": "vline",
                    "data": None,
                    "label": f"Change Point (Δ={mag:.2f})",
                    "x": change_time,
                    "color": "red",
                    "linestyle": "--",
                    "alpha": 0.75,
                    "linewidth": 1.5
                })

                # Add annotation to plot
                plots.append({
                    "plot_type": "annotate",
                    "data": None,
                    "xy": change_time,
                    "text": f"Δ={mag:.2f}",
                    "color": "red",
                })

            self._plot(plots=plots, plot_size=fig_size, dpi=fig_dpi,
                       fig_title=title,
                       fig_xlabel="Time",
                       fig_ylabel=metric_name)
