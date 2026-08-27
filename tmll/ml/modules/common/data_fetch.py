import pandas as pd
from typing import Any, List, Dict, Optional, Tuple, Union, cast

from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.common.models.tree.tree import Tree
from tmll.tmll_client import TMLLClient


class DataFetcher:
    def __init__(self, client: TMLLClient) -> None:
        self.client = client
        self.logger = client.logger

    def fetch_data(self, experiment: Experiment, target_outputs: Optional[List[Output]] = None,
                   **kwargs) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[List[Output]]]:
        """
        Fetch and process data for the given outputs.

        :param experiment: The experiment to fetch the data from
        :type experiment: Experiment
        :param target_outputs: The target outputs to fetch the data from
        :type target_outputs: Optional[List[Output]]
        :param force_reload: Whether to force reload the data
        :type force_reload: bool
        :param kwargs: Additional keyword arguments
        :type kwargs: Dict
        :return: The fetched data and the outputs
        :rtype: Tuple[Optional[Dict[str, pd.DataFrame]], Optional[List[Output]]]
        """
        total_outputs = None
        custom_output_ids = [output.id for output in target_outputs] if target_outputs else None
        total_outputs = self.client.fetch_outputs_with_tree(experiment=experiment, custom_output_ids=custom_output_ids)

        if not total_outputs:
            self.logger.error("No outputs fetched")
            return None, []

        if not target_outputs:
            target_outputs = [cast(Output, output["output"]) for output in total_outputs]

        self.logger.info("Fetching data...")
        data = self._fetch_data(experiment=experiment,
                                experiment_outputs=total_outputs,
                                target_outputs=target_outputs,
                                **kwargs)
        if not data:
            self.logger.error("No data fetched")
            return None, None

        dataframes = {}
        for output in target_outputs:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            if isinstance(data[output.id], dict):
                for key, value in data[output.id].items():
                    if isinstance(value, pd.DataFrame):
                        dataframe = value
                        dataframe = dataframe.rename(columns={'y': output.id, 'x': 'timestamp'})
                        if dataframes.get(f"{output.id}${key}", None) is None:
                            dataframes[f"{output.id}${key}"] = dataframe
                        else:
                            dataframes[f"{output.id}${key}"] = pd.merge(dataframes[f"{output.id}_{key}"], dataframe, on='timestamp', how='outer')
            elif isinstance(data[output.id], pd.DataFrame):
                dataframes[output.id] = data[output.id]

        return dataframes, [cast(Output, output["output"]) for output in total_outputs]

    def _fetch_data(self, experiment: Experiment, experiment_outputs: List[Dict[str, Output | Tree]],
                    target_outputs: List[Output], **kwargs: Dict[str, Any]) -> Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Fetch the data from the given outputs.

        :param experiment: The experiment to fetch the data from
        :type experiment: Experiment
        :param experiment_outputs: The outputs of the experiment
        :type experiment_outputs: List[Dict[str, Output | Tree]]
        :param target_outputs: The target outputs to fetch the data from
        :type target_outputs: List[Output]
        :param force_reload: Whether to force reload the data
        :type force_reload: bool
        :param kwargs: Additional keyword arguments
        :type kwargs: Dict
        :return: The fetched data
        :rtype: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        """
        custom_output_ids = [output.id for output in target_outputs]
        return self.client.fetch_data(experiment=experiment,
                                      outputs=experiment_outputs,
                                      custom_output_ids=custom_output_ids,
                                      separate_columns=True,
                                      **kwargs)
