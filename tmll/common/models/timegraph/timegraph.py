from typing import List, Optional

from tmll.common.models.timegraph.row import TimeGraphRow
from tmll.common.models.timegraph.arrow import TimeGraphArrow


class TimeGraph:
    """A class to represent a time graph from the TSP server.

    Attributes:
        rows (List[TimeGraphRow]): The rows of the time graph.
        arrows (Optional[List[TimeGraphArrow]]): The arrows of the time graph.
    """

    def __init__(self, rows: List[TimeGraphRow], arrows: Optional[List[TimeGraphArrow]] = None) -> None:
        self.rows = rows
        self.arrows = arrows if arrows is not None else []

    def __repr__(self) -> str:
        return f"TimeGraph(rows={self.rows}, arrows={self.arrows})"

    @classmethod
    def from_tsp_time_graph(cls, tsp_time_graph):
        rows = [TimeGraphRow.from_tsp_row(row) for row in tsp_time_graph.rows]
        return cls(rows)

    @staticmethod
    def parse_tsp_arrows(tsp_arrows):
        arrows = [TimeGraphArrow.from_tsp_arrow(arrow) for arrow in tsp_arrows]
        return arrows
