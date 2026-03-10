"""
Trace Server Protocol (TSP) Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of TSP.
The library is implemented as a set of Python classes that can be used to interact with Trace Server Protocol (TSP) and apply machine learning techniques on the data.
"""
import time
import re

import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Tuple, Union, cast

from tmll.common.models.data.table.column import TableDataColumn
from tmll.common.models.data.table.table import TableData
from tmll.common.models.output import Output
from tmll.common.models.timegraph.timegraph import TimeGraph
from tmll.common.models.trace import Trace
from tmll.common.models.experiment import Experiment
from tmll.common.models.tree.node import NodeTree
from tmll.common.models.tree.tree import Tree

from tmll.tsp.tsp.indexing_status import IndexingStatus
from tmll.tsp.tsp.response import ResponseStatus
from tmll.tsp.tsp.tsp_client import TspClient

from tmll.services.tsp_installer import TSPInstaller
from tmll.services.instrumentation import TMLLInstrumentation

from tmll.utils.name_generator import NameGenerator

from tmll.common.services.logger import Logger


class TMLLClient:

    def __init__(self, tsp_server_host: str = "localhost", tsp_server_port: int = 8080,
                 install_tsp_server: bool = False, force_install: bool = False,
                 verbose: bool = True, **kwargs) -> None:
        """
        Constructor for the TMLLClient class.

        :param tsp_server_host: Host of the TSP server
        :type tsp_server_host: str
        :param tsp_server_port: Port of the TSP server
        :type tsp_server_port: int
        :param install_tsp_server: Flag to install the TSP server if it is not running
        :type install_tsp_server: bool
        :param force_install: Flag to force the installation of the TSP server
        :type force_install: bool
        :param verbose: Flag to enable/disable the verbose mode
        :type verbose: bool
        :param kwargs: Additional parameters
        :type kwargs: Dict
        """

        base_url = f"http://{tsp_server_host}:{tsp_server_port}/tsp/api/"
        self.tsp_client = TspClient(base_url=base_url)

        self.logger = Logger("TMLLClient", verbose)

        # Check if the TSP server is running (i.e., check if server is reachable)
        self.health_status = 400
        try:
            self.health_status = self.tsp_client.fetch_health().status_code
            if self.health_status != 200:
                self.logger.error(f"TSP server is not running properly. Check its health status.")
                return
        except Exception as e:
            # If the TSP server is not running and the user has not specified to install the TSP server, return
            if not install_tsp_server:
                self.logger.error(
                    "Failed to connect to the TSP server. Please make sure that the TSP server is running. If you want to install the TSP server, set the 'install_tsp_server' parameter to True.")
                return

            self.logger.warning("TSP server is not running. Trying to run the TSP server, and if not installed, install it.")

            tsp_installer = TSPInstaller()
            tsp_installer.install()

            # Check if the TSP server is installed successfully
            self.health_status = self.tsp_client.fetch_health().status_code
            if self.health_status != 200:
                self.logger.error("Failed to install the TSP server.")
                return

        self.logger.info("Connected to the TSP server successfully.")

        """
        THESE ARE THE TEMPORARY METHODS THAT WILL BE REMOVED IN THE FINAL VERSION
        """
        def _delete_experiments():
            response = self.tsp_client.fetch_experiments()
            for experiment in response.model.experiments:
                self.tsp_client.delete_experiment(experiment.UUID)

        def _delete_traces():
            response = self.tsp_client.fetch_traces()
            for trace in response.model.traces:
                # Do not also delete the trace from disk; file part of this repo.
                self.tsp_client.delete_trace(trace.UUID, False)

        if kwargs.get("delete_all", False):
            _delete_experiments()
            _delete_traces()

    def create_experiment(self, traces: List[Dict[str, str]], experiment_name: str) -> Union[None, Experiment]:
        """
        Import traces into the Trace Server Protocol (TSP) server.

        :param traces: List of traces to import
        :type traces: List[Dict[str, str]]
        :param experiment_name: Name of the experiment to create
        :type experiment_name: str
        """
        # For each trace, add it to the TSP server
        opened_traces = []
        for trace in traces:
            if "path" not in trace:
                self.logger.error("The 'path' parameter is required for each trace.")
                continue

            # If the name is not provided for the trace, generate a name for the trace
            if "name" not in trace:
                trace["name"] = NameGenerator.generate_name(base=trace["path"])

            response = self.tsp_client.open_trace(name=trace["name"], path=trace["path"])
            if response.status_code != 200:
                self.logger.error(f"Failed to open trace '{trace['name']}'. Error: {response.status_text}")
                continue

            opened_traces.append(Trace.from_tsp_trace(response.model))
            self.logger.info(f"Trace '{trace['name']}' opened successfully.")

        # If no trace is opened, return None
        if not opened_traces:
            self.logger.error("No trace is opened. Please check the traces.")
            return None

        # If create_experiment is True, create an experiment with the opened traces
        trace_uuids = [trace.uuid for trace in opened_traces]

        opened_experiment = self.tsp_client.open_experiment(name=experiment_name, traces=trace_uuids)
        if opened_experiment.status_code != 200:
            self.logger.error(f"Failed to open experiment '{experiment_name}'. Error: {opened_experiment.status_text}")
            return None

        # Check the status of the experiment periodically until it is completed
        # P.S.: It's not an efficient way to check the status. However, the tsp server does not provide a way to get the status of the experiment directly.
        self.logger.info(f"Checking the indexing status of the experiment '{experiment_name}'.")
        experiment = None
        while (True):
            status = self.tsp_client.fetch_experiment(opened_experiment.model.UUID)
            if status.status_code != 200:
                self.logger.error(f"Failed to fetch experiment. Error: {status.status_text}")
                return None

            if status.model.indexing_status.name == IndexingStatus.COMPLETED.name:
                experiment = Experiment.from_tsp_experiment(status.model)
                self.logger.info(f"Experiment '{experiment_name}' is loaded completely.")
                break

            # Wait for 1 second before checking the status again
            time.sleep(1)

        # If the experiment is loaded, assign the outputs to the experiment
        if experiment:
            experiment.assign_outputs(self._fetch_outputs(experiment))

        return experiment

    def _fetch_outputs(self, experiment: Experiment) -> List[Output]:
        """
        Fetch the outputs of the experiment.

        :param experiment: Experiment to fetch the outputs
        :type experiment: Experiment
        :return: List of outputs
        :rtype: List[Output]
        """

        if experiment is None:
            self.logger.error("Experiment is not loaded. Please load the experiment first by calling the 'create_experiment' method.")
            return []

        # Get the outputs of the experiment
        outputs = self.tsp_client.fetch_experiment_outputs(experiment.uuid)
        if outputs.status_code != 200:
            self.logger.error(f"Failed to fetch experiment outputs. Error: {outputs.status_text}")
            return []

        fetched_outputs = [Output.from_tsp_output(output) for output in outputs.model.descriptors]
        self.logger.info("Outputs are fetched successfully.")

        return fetched_outputs

    def fetch_outputs_with_tree(self, experiment: Experiment, custom_output_ids: Optional[List[str]] = None) -> Union[None, List[Dict[str, Union[Output, Tree]]]]:
        """
        Fetch the outputs of the experiment.

        :param experiment: Experiment to fetch the outputs
        :type experiment: Experiment
        :param custom_output_ids: List of custom output IDs to fetch
        :type custom_output_ids: Optional[List[str]]
        :param force_reload: Flag to force reload the outputs
        :type force_reload: bool
        """

        if experiment is None:
            self.logger.error("Experiment is not loaded. Please load the experiment first by calling the 'create_experiment' method.")
            return None

        # Get the outputs of the experiment
        outputs = experiment.outputs
        if not outputs:
            outputs = self._fetch_outputs(experiment)

        fetched_outputs = []
        for output in outputs:
            # Check if the custom_output_ids is specified and the output is not in the custom_output_ids
            if custom_output_ids and output.id not in custom_output_ids:
                continue

            # Get the trees of the outputs
            match output.type:
                case "TABLE":
                    response = self.tsp_client.fetch_virtual_table_columns(exp_uuid=experiment.uuid, output_id=output.id)
                    if response.status_code != 200:
                        self.logger.error(f"Failed to fetch table columns. Error: {response.status_text}")
                        continue

                    columns = response.model.model.columns
                    tree = Tree([NodeTree(name=col.name, id=col.id, parent_id=-1) for col in columns])
                    fetched_outputs.append({
                        "output": output,
                        "tree": tree
                    })
                    self.logger.info(f"Output '{output.name}' and its tree fetched successfully.")
                    continue

                case "DATA_TREE":
                    while True:
                        response = self.tsp_client.fetch_datatree(exp_uuid=experiment.uuid, output_id=output.id)
                        if response.status_code != 200:
                            response = self.tsp_client.fetch_timegraph_tree(exp_uuid=experiment.uuid, output_id=output.id)
                            if response.status_code != 200:
                                self.logger.error(f"Failed to fetch data tree. Error: {response.status_text}")
                                break

                        # Wait until the model is completely fetched (i.e., status is COMPLETED)
                        if response.model.status.name == ResponseStatus.COMPLETED.name:
                            break

                        time.sleep(1)

                case "TIME_GRAPH":
                    while True:
                        response = self.tsp_client.fetch_timegraph_tree(exp_uuid=experiment.uuid, output_id=output.id)
                        if response.status_code != 200:
                            self.logger.error(f"Failed to fetch time graph tree. Error: {response.status_text}")
                            break

                        # Wait until the model is completely fetched (i.e., status is COMPLETED)
                        if response.model.status.name == ResponseStatus.COMPLETED.name:
                            break

                        time.sleep(1)

                case "TREE_TIME_XY":
                    while True:
                        response = self.tsp_client.fetch_xy_tree(exp_uuid=experiment.uuid, output_id=output.id)
                        if response.status_code != 200:
                            self.logger.error(f"Failed to fetch XY tree. Error: {response.status_text}")
                            break

                        # Wait until the model is completely fetched (i.e., status is COMPLETED)
                        if response.model.status.name == ResponseStatus.COMPLETED.name:
                            break

                        time.sleep(1)
                case _:
                    self.logger.warning(f"Output type '{output.type}' is not supported.")
                    continue

            if response.model is None:
                self.logger.warning(f"Tree of the output '{output.name}' is None.")
                continue

            model = response.model.model
            if model is None:
                self.logger.warning(f"Tree of the output '{output.name}' is None.")
                continue

            tree = Tree.from_tsp_tree(model)

            fetched_outputs.append({
                "output": output,
                "tree": tree
            })
            self.logger.info(f"Output '{output.name}' and its tree fetched successfully.")

        self.logger.info("Outputs are fetched successfully.")

        return fetched_outputs

    def fetch_data(self, experiment: Experiment, outputs: List[Dict[str, Union[Output, Tree]]], custom_output_ids: Optional[List[str]] = None, **kwargs) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Fetch the data for the outputs.

        :param experiment: Experiment to fetch the data
        :type experiment: Experiment
        :param outputs: List of outputs to fetch the data
        :type outputs: List[Dict[str, Union[Output, Tree]]]
        :param custom_output_ids: List of custom output IDs to fetch
        :type custom_output_ids: Optional[List[str]]
        :param kwargs: Additional parameters for fetching the data
        :type kwargs: Dict
        :return: Dictionary of processed DataFrames
        :rtype: Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]
        """

        # Check if the experiment is loaded
        if experiment is None:
            self.logger.error("Experiment is not loaded. Please load the experiment first.")
            return None

        # Check if the outputs are fetched
        if not outputs:
            self.logger.error("Outputs are not fetched. Please fetch the outputs first.")
            return None

        datasets = {}
        for output in outputs:
            # Get the output and tree from the outputs
            o_output = cast(Output, output["output"])
            o_tree = cast(Tree, output["tree"])

            # If custom_output_ids is specified, only fetch the data of the specified outputs
            if custom_output_ids and o_output.id not in custom_output_ids:
                continue

            match o_output.type:
                case "TREE_TIME_XY":
                    # Prepare the parameters for the TSP server
                    item_ids = list(map(int, [node.id for node in o_tree.nodes]))
                    time_range_start = kwargs.get("start", experiment.start)
                    time_range_end = kwargs.get("end", experiment.end)
                    resample_freq = kwargs.get("resample_freq", "1s")

                    intervals = self._calculate_intervals(time_range_start, time_range_end, resample_freq)

                    for interval in intervals:
                        parameters = {
                            TspClient.PARAMETERS_KEY: {
                                TspClient.REQUESTED_ITEM_KEY: item_ids,
                                TspClient.REQUESTED_TIME_RANGE_KEY: {
                                    TspClient.REQUESTED_TIME_RANGE_START_KEY: interval[0],
                                    TspClient.REQUESTED_TIME_RANGE_END_KEY: interval[1],
                                    TspClient.REQUESTED_TIME_RANGE_NUM_TIMES_KEY: interval[2]
                                }
                            }
                        }

                        # Send a request to the TSP server to fetch the XY data
                        data = self.tsp_client.fetch_xy(exp_uuid=experiment.uuid, output_id=o_output.id, parameters=parameters)
                        if data.status_code != 200 or data.model.model is None:
                            self.logger.error(f"Failed to fetch XY data. Error: {data.status_text}")
                            break  # Exit the loop if there's an error

                        if not data.model.model.series:
                            break  # Exit the loop if no more data is returned

                        x, y = None, None
                        for serie in data.model.model.series:
                            # Get the x and y values from the data
                            x, y = serie.x_values, serie.y_values

                            # Create a pandas DataFrame from the data
                            dataset = pd.DataFrame(data=np.c_[x, y], columns=["x", "y"])

                            # Get the series name
                            series_name = Tree.get_node_by_id(o_tree, serie.series_id)
                            if series_name:
                                prefix = series_name.name
                                while True:
                                    parent_node = o_tree.get_node_parent(series_name.id)
                                    if parent_node:
                                        prefix = f"{parent_node.name}->{prefix}"
                                        series_name = parent_node
                                    else:
                                        break
                                series_name = prefix
                            else:
                                series_name = serie.series_name

                            # Add the dataset to the datasets dictionary
                            datasets[o_output.id] = datasets.get(o_output.id, {})
                            datasets[o_output.id][series_name] = pd.concat([datasets[o_output.id].get(series_name, pd.DataFrame()), dataset])

                case "TIME_GRAPH":
                    items = list(map(int, [node.id for node in o_tree.nodes]))
                    time_range_start = kwargs.get("start", experiment.start)
                    time_range_end = kwargs.get("end", experiment.end)
                    resample_freq = kwargs.get("resample_freq", "1s")
                    strategy = kwargs.get("strategy", "DEEP")

                    intervals = self._calculate_intervals(time_range_start, time_range_end, resample_freq)

                    for interval in intervals:
                        parameters = {
                            TspClient.PARAMETERS_KEY: {
                                TspClient.REQUESTED_ITEM_KEY: items,
                                TspClient.REQUESTED_TIME_RANGE_KEY: {
                                    TspClient.REQUESTED_TIME_RANGE_START_KEY: interval[0],
                                    TspClient.REQUESTED_TIME_RANGE_END_KEY: interval[1],
                                    TspClient.REQUESTED_TIME_RANGE_NUM_TIMES_KEY: interval[2]
                                },
                                "filter_query_parameters": {
                                    "strategy": strategy
                                }
                            }
                        }

                        # Send a request to the TSP server to fetch the time graph data
                        data = self.tsp_client.fetch_timegraph_states(exp_uuid=experiment.uuid, output_id=o_output.id, parameters=parameters)
                        if data.status_code != 200 or data.model.model is None:
                            self.logger.error(f"Failed to fetch time graph data. Error: {data.status_text}")
                            break

                        # Check if the time graph data is empty
                        if not data.model.model.rows:
                            break

                        timegraph = TimeGraph.from_tsp_time_graph(data.model.model)

                        # Create a pandas DataFrame from the time graph data
                        data = []
                        for row in timegraph.rows:
                            for state in row.states:
                                node = o_tree.get_node_by_id(row.entry_id)
                                parent_node = o_tree.get_node_parent(row.entry_id)
                                parent_id = parent_node.id if parent_node else row.entry_id
                                data.append({
                                    "entry_id": row.entry_id,
                                    "entry_name": node.name if node else None,
                                    "parent_id": parent_id,
                                    "start_time": state.start,
                                    "end_time": state.end,
                                    "label": state.label
                                })

                        dataset = pd.DataFrame(data)

                        # Add the dataset to the datasets dictionary
                        if o_output.id not in datasets:
                            datasets[o_output.id] = pd.DataFrame()

                        datasets[o_output.id] = pd.concat([datasets[o_output.id], dataset])

                case "TABLE" | "DATA_TREE":
                    columns = self.tsp_client.fetch_virtual_table_columns(exp_uuid=experiment.uuid, output_id=o_output.id)
                    if columns.status_code != 200:
                        self.logger.error(f"Failed to fetch '{o_output.name}' virtual table columns. Error: {columns.status_text}")
                        continue

                    columns = [TableDataColumn.from_tsp_table_column(column) for column in columns.model.model.columns]

                    # We keep the initial columns to use them in the DataFrame creation
                    initial_table_columns = []

                    start_index = int(kwargs.get("table_line_start_index", 0))  # Start index of the table data. Default is 0 (i.e., the first row)
                    line_count = int(kwargs.get("table_line_count", 65536))  # 65536 is the maximum value that the TSP server accepts
                    column_ids = list(map(int, kwargs.get("table_line_column_ids", [c.id for c in columns])))  # Which columns to fetch from the table
                    search_direction = kwargs.get("table_line_search_direction", "NEXT")  # Search direction for the table data (i.e., NEXT or PREVIOUS)

                    # Convert table_line_column_names into column_ids if it is provided
                    if not column_ids and "table_line_column_names" in kwargs:
                        column_ids = [int(c.id) for c in columns if c.name.lower() in map(str.lower, kwargs.get("table_line_column_names", []))]

                    while True:
                        # Prepare the parameters for the TSP server
                        parameters = {
                            TspClient.PARAMETERS_KEY: {
                                TspClient.REQUESTED_TABLE_LINE_INDEX_KEY: start_index,
                                TspClient.REQUESTED_TABLE_LINE_COUNT_KEY: line_count,
                                TspClient.REQUESTED_TABLE_LINE_COLUMN_IDS_KEY: column_ids,
                                TspClient.REQUESTED_TABLE_LINE_SEACH_DIRECTION_KEY: search_direction
                            }
                        }

                        # Send a request to the TSP server to fetch the virtual table data
                        table_request = self.tsp_client.fetch_virtual_table_lines(
                            exp_uuid=experiment.uuid, output_id=o_output.id, parameters=parameters)

                        # If the request is not successful or the table data is None, break the loop
                        if table_request.status_code != 200 or table_request.model.model is None:
                            self.logger.error(f"Failed to fetch '{o_output.name}' virtual table data. Error: {table_request.status_text}")
                            break

                        # Create the table model
                        table = TableData.from_tsp_table(table_request.model.model)

                        # If the DataFrame for the output is not created yet, create it
                        if o_output.id not in datasets:
                            initial_table_columns = [c.name for c_id in table.columns for c in columns if c.id == c_id]
                            if not initial_table_columns:
                                initial_table_columns = [f"Column {c}" for c in table.columns]

                            datasets[o_output.id] = pd.DataFrame(columns=initial_table_columns)

                        # If there are no rows in the table, break the loop since there is no more data to fetch
                        if not table.rows:
                            break

                        # Convert the table rows to a DataFrame
                        row_data = pd.DataFrame([row.values for row in table.rows], columns=initial_table_columns)

                        # If the "separate_columns" parameter is True, extract the features from the columns
                        # For example, if the column contains "key=value" pairs, extract the key and value as separate columns
                        separate_columns = kwargs.get("separate_columns", False)
                        if separate_columns:
                            processor = TableProcessor()
                            row_data = processor.extract_features_from_columns(row_data)

                        # Concatenate the row data to the DataFrame of the output
                        datasets[o_output.id] = pd.concat([datasets[o_output.id], row_data], ignore_index=True)

                        # Update the start index for the next iteration (i.e., next batch of data)
                        start_index += line_count
                case _:
                    self.logger.warning(f"Output type '{o_output.type}' is not supported.")
                    continue

            self.logger.info(f"Data fetched for the output '{o_output.name}'.")

        self.logger.info("All data fetched successfully.")

        return datasets

    def _calculate_intervals(self, start: int, end: int, resample_freq: str) -> List[Tuple[int, int, int]]:
        """
        Based on the resampling frequency, we should indicate how many items we want to fetch for XY/TIMEGRAPH data
        from the TSP server. For example, if the resampling frequency is 1s, we should fetch 1 item per second.

        Now, based on the start and end timestamps, we should calculate the number of items
        to fetch from the TSP server. For example, if the start timestamp is 0 and the end timestamp is 1000,
        and the resampling frequency is 1s, we should fetch 1000 items.

        The formula to calculate the number of items is:
            num_items = (end - start) / resample_freq
        where:
            - start: Start timestamp in nano-seconds
            - end: End timestamp in nano-seconds
            - resample_freq: Resampling frequency in string format (e.g., "1s", "1ms", "1us", etc.)

        The catch is that, the tsp server can handle up to 65536 items at a time. So, we need to return the list
        of intervals to fetch the XY data from the TSP server. For example, if the number of items is 100000, we should
        return a list of two intervals with the following values:
            - start: the start timestamp of the interval
            - end: the end timestamp of the interval
            - num_items: the number of items to fetch from the TSP server
        So, the list should be like:
            [(start1, end1, 65536), (start2, end2, 34464)]

        :param start: Start timestamp
        :type start: int
        :param end: End timestamp
        :type end: int
        :param resample_freq: Resampling frequency
        :type resample_freq: str
        :return: List of intervals
        :rtype: List[Tuple[int, int, int]]
        """
        resample_freq_timedelta = pd.to_timedelta(resample_freq).value

        num_items = int((end - start) / resample_freq_timedelta)
        intervals = []

        while num_items > 0:
            batch_size = min(num_items, 65536)
            interval_end = start + batch_size * resample_freq_timedelta
            intervals.append((start, interval_end, batch_size))
            start = interval_end
            num_items -= batch_size

        return intervals

    @staticmethod
    def enable_instrumentation(instrumentation_file: Optional[str] = None, instrument_kernel: bool = False, verbose: bool = True) -> None:
        """
        Enable the instrumentation for user-space (i.e., Python), and if specified, for kernel-space (via LTTng).

        :param instrumentation_file: Instrumentation file to use for storing the instrumentation data
        :type instrumentation_file: Optional[str]
        :param instrument_kernel: Flag to enable the kernel-space instrumentation (i.e., LTTng) - Only for Linux
        :type instrument_kernel: bool
        :param verbose: Flag to enable/disable the verbose mode for LTTng
        :type verbose: bool
        """
        TMLLInstrumentation.enable(instrumentation_file, instrument_kernel, verbose)

    @staticmethod
    def disable_instrumentation(convert_to_json: bool = False) -> None:
        """
        Disable the instrumentation for user-space (i.e., Python) and kernel-space (via LTTng).

        :param convert_to_json: Flag to convert the instrumentation data (user-space) to JSON format
        :type convert_to_json: bool
        """
        TMLLInstrumentation.disable(convert_to_json)


class TableProcessor:
    """
    TableProcessor class is used to process the table data.
    Using parallel processing, it extracts features from the columns of the table data efficiently.
    """

    def __init__(self):
        self.key_cleanup_pattern = re.compile(r'[^\w\s]')

    def _parse_value(self, value_str: str) -> str:
        """
        Parse the value string and return the value.

        :param value_str: Value string to parse
        :type value_str: str
        :return: Parsed value
        :rtype: str
        """
        if not isinstance(value_str, str):
            return value_str
        value_str = value_str.strip('[]')
        if '=' in value_str and not value_str.startswith(('[', '{')):
            return ','.join(part.strip() for part in value_str.split(',') if '=' in part)
        return value_str

    def _process_part(self, part: str, base_column: str = '') -> Dict[str, str]:
        """
        Process the part of the column and return the extracted features.

        :param part: Part of the column to process
        :type part: str
        :param base_column: Base column name
        :type base_column: str
        :return: Extracted features
        :rtype: Dict[str, str]
        """
        if not isinstance(part, str) or '=' not in part:
            return {}

        try:
            key, value = part.split('=', 1)
        except ValueError:
            return {}

        key = self.key_cleanup_pattern.sub('', key).strip()
        col_name = f"{base_column}_{key}" if base_column else key

        if isinstance(value, str) and value.startswith('['):
            parts = value.strip('[]').split(',')
            result = {}
            for p in parts:
                if '=' in p:
                    result.update(self._process_part(p.strip(), col_name))
            return result
        return {col_name: self._parse_value(value)}

    def extract_features_from_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the columns of the table data.

        :param dataframe: DataFrame to process
        :type dataframe: pd.DataFrame
        :return: Processed DataFrame
        :rtype: pd.DataFrame
        """
        if len(dataframe) == 0:
            return dataframe

        all_columns = set()
        extracted_data = {}

        for column in dataframe.columns:
            series = dataframe[column].astype(str).replace('\n', '').str.strip()

            sample_rows = series.str.split(', ')
            for row in sample_rows:
                if not isinstance(row, list):
                    continue

                for part in row:
                    extracted = self._process_part(part)
                    all_columns.update(extracted.keys())

        for col in all_columns:
            extracted_data[col] = [np.nan] * len(dataframe)

        for column in dataframe.columns:
            series = dataframe[column].astype(str).replace('\n', '').str.strip()

            for idx, row in enumerate(series.str.split(', ')):
                if not isinstance(row, list):
                    continue

                for part in row:
                    extracted = self._process_part(part)
                    for key, value in extracted.items():
                        if key in extracted_data:
                            extracted_data[key][idx] = value

        result = pd.DataFrame(extracted_data, index=dataframe.index)

        columns_to_keep = set(dataframe.columns) - set(result.columns)
        if columns_to_keep:
            result = pd.concat([dataframe[list(columns_to_keep)], result], axis=1)

        return result
