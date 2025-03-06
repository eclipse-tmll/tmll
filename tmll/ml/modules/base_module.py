from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

from tmll.common.models.output import Output
from tmll.common.services.logger import Logger
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor
from tmll.ml.utils.formatter import Formatter
from tmll.ml.visualization.plot_factory import PlotFactory
from tmll.common.models.experiment import Experiment
from tmll.ml.visualization.utils import PlotUtils
from tmll.tmll_client import TMLLClient


class BaseModule(ABC):

    def __init__(self, client: TMLLClient, experiment: Experiment) -> None:
        """
        Initialize the base module with the given TMLL client.

        :param client: The TMLL client to use
        :type client: TMLLClient
        :param experiment: The experiment to analyze
        :type experiment: Experiment
        """
        self.client: TMLLClient = client
        self.experiment: Experiment = experiment

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.outputs: Optional[List[Output]] = None
        self.data_fetcher: DataFetcher = DataFetcher(client)
        self.data_preprocessor: DataPreprocessor = DataPreprocessor()

        self.logger: Logger = Logger(self.__class__.__name__)

    def _plot(self, plots: List[Dict[str, Any]], plot_size: Tuple[float, float] = (15, 10), **kwargs) -> None:
        """
        Plot the given plots. 

        :param plots: The plots to plot. Each plot should be a dictionary with the following keys:
            - plot_type (PLOT_TYPES): The type of the plot (e.g., "time_series" or "scatter")
            - data (Any): The data to plot
            - x (str): The name of the x-axis column
            - y (str): The name of the y-axis column
            - hue (str, optional): The name of the hue column. Default is None
            - color (str, optional): The color of the plot. Default is "blue"
        :type plots: List[Dict[str, Any]]
        :param plot_size: The size of the plot. Default is (15, 10)
        :type plot_size: Tuple[int, int], optional
        :param kwargs: Additional keyword arguments to pass to the plot
            - title (str): The title of the plot
            - x (str): The name of the x-axis
            - y (str): The name of the y-axis
            - fig_title (str): The title of the figure
            - fig_xlabel (str): The x-axis label of the figure
            - fig_ylabel (str): The y-axis label of the figure
            - legend (bool): Whether to show the legend. Default is True
        :type kwargs: dict
        :return: None
        """
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=plot_size, dpi=kwargs.get("dpi", 100))

        # Plot each plot
        for plot_info in plots:
            plot_type = plot_info.get("plot_type", None)
            data = plot_info.get("data", None)

            # Create the plot
            plot_strategy = PlotFactory.create_plot(plot_type)
            plot_strategy.plot(ax, data, **{k: v for k, v in plot_info.items() if k not in ["plot_type", "data"]})

        # Set the title, x-axis label, and y-axis label of the plot
        ax.set_title(kwargs.get("fig_title", ""))
        ax.set_xlabel(kwargs.get("fig_xlabel", ""))
        ax.set_ylabel(kwargs.get("fig_ylabel", ""))

        if kwargs.get("fig_xticks", None) is not None:
            ax.set_xticks(kwargs.get("fig_xticks"))  # type: ignore

        if kwargs.get("fig_yticks", None) is not None:
            ax.set_yticks(kwargs.get("fig_yticks"))  # type: ignore
        else:
            y_min, y_max = ax.get_ylim()
            if isinstance(y_min, (int, float)) and isinstance(y_max, (int, float)):
                num_yticks = kwargs.get("fig_num_yticks", 6)
                yticks = PlotUtils.get_formatted_ticks(y_min, y_max, num_yticks)
                ax.set_yticks(yticks)

                data_range = yticks[-1] - yticks[0]
                padding = data_range * 0.025
                ax.set_ylim(yticks[0] - padding, yticks[-1] + padding)
                ax.set_yticklabels([f"{val:.2f}{unit}" for tick in yticks for val, unit in [Formatter.format_large_number(tick)]])

        if kwargs.get("fig_xticklabels", None) is not None:
            ax.set_xticklabels(kwargs.get("fig_xticklabels"))  # type: ignore
        if kwargs.get("fig_yticklabels", None) is not None:
            ax.set_yticklabels(kwargs.get("fig_yticklabels"))  # type: ignore
        if kwargs.get("fig_xticklabels_rotation", None) is not None:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=kwargs.get("fig_xticklabels_rotation"))
        if kwargs.get("fig_yticklabels_rotation", None) is not None:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=kwargs.get("fig_yticklabels_rotation"))

        # Add the legend to the plot (remove duplicates)
        if kwargs.get("legend", True):
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            PlotUtils.set_standard_legend_style(ax, by_label.values(), by_label.keys(), title=kwargs.get("legend_title", None))
        else:
            if ax.get_legend():
                ax.get_legend().remove()

        ax.grid(kwargs.get("grid", True))

        # Display the plot
        plt.tight_layout()
        plt.show()

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Base processing method that handles common data fetching and preprocessing tasks.

        :param outputs: Optional list of outputs to process
        :type outputs: Optional[List[Output]]
        :param kwargs: Additional keyword arguments for specific module processing
        """
        self.dataframes.clear()

        data, outputs = self.data_fetcher.fetch_data(
            experiment=self.experiment,
            target_outputs=outputs,
            resample_freq=kwargs.get("resample_freq", "1s"),
            **kwargs.get("fetch_params", {}))

        if data is None:
            self.logger.error("No data fetched")
            return

        self.outputs = outputs

        # Process each output
        for output_key, output_data in data.items():
            output = next(iter(output for output in outputs if output.id == output_key.split("$")[0]), None) if outputs else None
            if output is None:
                continue

            if output.id not in self.dataframes:
                df = output_data

                if output.type == "TIME_GRAPH":
                    df = df.rename({"start_time": "timestamp"}, axis=1)
                    df["end_time"] = pd.to_datetime(df["end_time"])

                # Apply common preprocessing steps
                if kwargs.get("normalize", True):
                    df = self.data_preprocessor.normalize(df)
                if kwargs.get("convert_datetime", True):
                    df = self.data_preprocessor.convert_to_datetime(df)
                if kwargs.get("resample", True):
                    df = self.data_preprocessor.resample(df, frequency=kwargs.get("resample_freq", "1s"))
                if kwargs.get("remove_minimum", False):
                    df = self.data_preprocessor.remove_minimum(df)

                self.dataframes[output.id] = df

        # Filter out dataframes with less than min_size instances
        min_size = kwargs.get("min_size", 1)
        self.dataframes = {k: v for k, v in self.dataframes.items() if len(v) >= min_size}

        # Align timestamps if needed
        if kwargs.get("align_timestamps", True) and self.dataframes:
            self.dataframes, self.timestamps = DataPreprocessor.align_timestamps(self.dataframes)

        # Call module-specific post-processing
        self._post_process(**kwargs)

    @abstractmethod
    def _post_process(self, **kwargs) -> None:
        """
        Abstract method for module-specific post-processing.
        Should be implemented by each module.

        :param kwargs: Additional keyword arguments for specific module processing
        """
        pass
