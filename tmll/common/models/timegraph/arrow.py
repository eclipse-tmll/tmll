from typing import Optional


class TimeGraphArrow:
    """A class to represent an arrow in a time graph.

    Attributes:
        source_id (int): The ID of the source node.
        target_id (int): The ID of the target node.
        start (int): The start time of the arrow.
        end (int): The end time of the arrow.
        duration (Optional[int]): The duration of the arrow.
        style (Optional[dict]): The style of the arrow.
    """

    def __init__(self, source_id: int, target_id: int, start: int, end: int, duration: Optional[int] = None, style: Optional[dict] = None) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.start = start
        self.end = end
        self.duration = duration
        self.style = style

    def __repr__(self) -> str:
        return f"TimeGraphArrow(source_id={self.source_id}, target_id={self.target_id}, start={self.start}, end={self.end}, duration={self.duration}, style={self.style})"

    @classmethod
    def from_tsp_arrow(cls, tsp_arrow):
        return cls(tsp_arrow.source_id, tsp_arrow.target_id, tsp_arrow.start, tsp_arrow.end,
                   tsp_arrow.duration if hasattr(
                       tsp_arrow, 'duration') else None,
                   tsp_arrow.style if hasattr(tsp_arrow, 'style') else None)
