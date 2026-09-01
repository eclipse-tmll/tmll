from typing import Optional


class TimeGraphState:
    """A class to represent a state in a time graph.

    Attributes:
        start (int): The start time of the state.
        end (int): The end time of the state.
        label (Optional[str]): The label of the state.
        style (Optional[dict]): The style of the state.
        value (Optional[list]): The values associated with the state.
        tags (Optional[int]): The tags associated with the state.
    """

    def __init__(self, start: int, end: int, label: Optional[str] = None, style: Optional[dict] = None, value: Optional[list] = None, tags: Optional[int] = None) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.style = style
        self.value = value
        self.tags = tags

    def __repr__(self) -> str:
        return f"TimeGraphState(start={self.start}, end={self.end}, label={self.label}, style={self.style}, value={self.value}, tags={self.tags})"

    @classmethod
    def from_tsp_state(cls, tsp_state):
        return cls(tsp_state.start_time, tsp_state.end_time,
                   tsp_state.label if hasattr(tsp_state, 'label') else None,
                   tsp_state.style if hasattr(tsp_state, 'style') else None,
                   tsp_state.value if hasattr(tsp_state, 'value') else None,
                   tsp_state.tags if hasattr(tsp_state, 'tags') else None)
