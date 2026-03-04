from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from tmll.common.models.experiment import Experiment
from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy
from tmll.ml.modules.anomaly_detection.strategies.frequency_domain import FrequencyDomainStrategy
from tmll.ml.modules.base_module import BaseModule
from tmll.ml.modules.anomaly_detection.strategies.combined import CombinedStrategy
from tmll.ml.modules.anomaly_detection.strategies.iqr import IQRStrategy
from tmll.ml.modules.anomaly_detection.strategies.iforest import IsolationForestStrategy
from tmll.ml.modules.anomaly_detection.strategies.moving_average import MovingAverageStrategy
from tmll.ml.modules.anomaly_detection.strategies.zscore import ZScoreStrategy
from tmll.ml.modules.anomaly_detection.strategies.seasonality import SeasonalityStrategy
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient


# Minimum number of data points required for anomaly detection analysis to proceed with
MINIMUM_REQUIRED_DATAPOINTS = 10

# Anomaly detection methods supported by the module
DETECTION_METHODS = ["zscore", "iqr", "moving_average", "combined", "iforest", "seasonality"]

# Combination methods supported by the module
COMBINATION_METHODS = ["zscore", "pca"]


@dataclass
class AnomalyDetectionResult:
    anomalies: Dict[str, pd.DataFrame]
    anomaly_periods: Dict[str, List]


class AnomalyDetection(BaseModule):
    """
    A class for performing anomaly detection on time series data.

    This class implements various anomaly detection strategies and provides
    methods for data processing, anomaly detection, and result visualization.

    :param client: The TMLL client for data communication.
    :type client: TMLLClient
    """

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the AnomalyDetection module.

        :param client: The TMLL client for data communication.
        :type client: TMLLClient
        :param experiment: The experiment to analyze.
        :type experiment: Experiment
        :param outputs: The list of outputs to analyze.
        :type outputs: Optional[List[Output]]
        :param method: The anomaly detection method to use.
        :type method: str
        :param kwargs: Additional keyword arguments to pass to the anomaly detection method.
        :type kwargs: dict
        """
        super().__init__(client=client, experiment=experiment)

        self.detection_method: str = ""
        self.strategy_map: Dict[str, AnomalyDetectionStrategy] = {
            "zscore": ZScoreStrategy(),
            "iqr": IQRStrategy(),
            "moving_average": MovingAverageStrategy(),
            "combined": CombinedStrategy([ZScoreStrategy(), IQRStrategy(), MovingAverageStrategy()]),
            "iforest": IsolationForestStrategy(),
            "seasonality": SeasonalityStrategy()
        }

        self.logger.info(f"Initializing AnomalyDetection module.")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        super()._process(outputs=outputs,
                         normalize=False,
                         min_size=kwargs.get("min_size", MINIMUM_REQUIRED_DATAPOINTS),
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        normalized_dataframes = [self.data_preprocessor.normalize(df) for df in self.dataframes.values()]
        df = self.data_preprocessor.combine_dataframes(normalized_dataframes)

        combination_method = kwargs.get("combine_method", COMBINATION_METHODS[0])
        if combination_method not in COMBINATION_METHODS:
            self.logger.warning(f"Unknown combination method: {combination_method}. Using {COMBINATION_METHODS[0]} instead.")
            combination_method = COMBINATION_METHODS[0]

        if combination_method == "zscore":
            scaler = StandardScaler()
            combined_data = np.sqrt(np.mean(np.square(scaler.fit_transform(df)), axis=1))
        else:  # PCA
            combined_data = PCA(n_components=1).fit_transform(df).flatten()

        self.dataframes[f"combined ({combination_method})"] = pd.DataFrame(combined_data, index=df.index, columns=["combined"])

    def find_anomalies(self, method: str = "zscore", **kwargs) -> Optional[AnomalyDetectionResult]:
        """
        Apply the specified anomaly detection method to the dataset.

        This internal method selects the appropriate strategy and applies it to the data.

        :param method: The anomaly detection method to use.
        :type method: str
        :param kwargs: Additional keyword arguments to pass to the detection method.
        :return: None
        """
        self.detection_method = method.lower()
        self.logger.info(f"Detecting anomalies using {self.detection_method} method...")

        strategy = self.strategy_map.get(self.detection_method)
        if not strategy:
            self.logger.error(f"Unknown detection method: {self.detection_method}. Please choose from {DETECTION_METHODS}.")
            return None

        anomalies: Dict[str, pd.DataFrame] = {}
        anomaly_periods: Dict[str, List] = {}
        for name, dataframe in self.dataframes.items():
            anomalies[name], anomaly_periods[name] = strategy.detect_anomalies(dataframe.copy(), **kwargs)

        return AnomalyDetectionResult(anomalies=anomalies, anomaly_periods=anomaly_periods)

    def plot_anomalies(self, anomaly_detection_results: Optional[AnomalyDetectionResult] = None, **kwargs) -> None:
        """
        Plot the original dataframe features along with the anomaly periods.

        :param anomaly_detection_results: The anomaly detection results to plot.
        :type anomaly_detection_results: AnomalyDetectionResult
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if not anomaly_detection_results or not anomaly_detection_results.anomalies or not self.dataframes:
            self.logger.error("No data or anomalies detected.")
            return

        fig_size = kwargs.get("fig_size", (18, 4))
        fig_dpi = kwargs.get("fig_dpi", 500)
        colors = plt.colormaps.get_cmap("tab10")

        for idx, (id, dataframe) in enumerate(self.dataframes.items()):
            output = self.experiment.get_output_by_id(id)
            name = output.name if output else id
            plots = []
            # Plot the original data
            plots.append({
                "plot_type": "time_series",
                "data": dataframe,
                "label": name,
                "color": colors(idx),
                "alpha": 1,
                "linewidth": 2
            })

            # Append the anomaly periods to the plots as span plot
            for start, end in anomaly_detection_results.anomaly_periods[id]:
                plots.append({
                    "label": "Anomaly Period",
                    "plot_type": "span",
                    "data": None,
                    "start": start,
                    "end": end,
                    "color": "red",
                    "alpha": 0.3,
                    "is_top": True
                })

            anomaly_points_list = []
            for point in anomaly_detection_results.anomalies[id].index:
                if not anomaly_detection_results.anomalies[id].loc[point].any():
                    continue

                # Check if the point is within any anomaly period
                in_anomaly_period = False
                for start, end in anomaly_detection_results.anomaly_periods[id]:
                    if start <= point <= end:
                        in_anomaly_period = True
                        break

                if in_anomaly_period:
                    continue

                anomaly_points_list.append({
                    "timestamp": point,
                    "value": dataframe.loc[point].values[0],
                })

            if anomaly_points_list:
                # Create the DataFrame after the loop
                anomaly_points = pd.DataFrame(anomaly_points_list)
                plots.append({
                    "plot_type": "scatter",
                    "data": anomaly_points,
                    "label": "Anomaly Points",
                    "x": "timestamp",
                    "y": "value",
                    "color": "red",
                    "alpha": 0.7,
                    "s": 100
                })

            self._plot(plots,
                       plot_size=fig_size,
                       dpi=fig_dpi,
                       fig_title=f"Anomaly Detection for \"{name}\" using \"{self.detection_method}\" method",
                       fig_xlabel="Time (index)",
                       fig_ylabel=name)
