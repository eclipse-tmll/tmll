"""Shared Gantt-chart rendering logic for the CLI 'timeline -p' flag and the
'plot_timeline' MCP tool."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

COLOR_PALETTE = ["green", "red", "blue", "orange", "purple", "gray"]
BAR_HEIGHT = 9


def states_to_bars(states):
    return [(state["start_time"], state["end_time"] - state["start_time"]) for state in states]


def group_states_by_entry(states):
    grouped = {}
    for state in states:
        entry = state["entry_name"]
        if entry not in grouped:
            grouped[entry] = []
        grouped[entry].append(state)
    return grouped


def _clean_label(label):
    if label is None or (isinstance(label, float) and label != label):
        return "(no label)"
    return label


def get_label_colors(states):
    label_colors = {}
    for state in states:
        label = _clean_label(state["label"])
        if label not in label_colors:
            label_colors[label] = COLOR_PALETTE[len(
                label_colors) % len(COLOR_PALETTE)]
    return label_colors


def build_gantt_figure(states):
    """Build a Gantt-style matplotlib Figure from a list of states. Does not
    save or close the figure. The caller decides where it goes (file, buffer)."""
    label_colors = get_label_colors(states)
    grouped = group_states_by_entry(states)

    fig, ax = plt.subplots(figsize=(12, max(4, len(grouped) * 0.5)))
    ax.set_title("TMLL Timeline Gantt Chart")
    ax.set_xlabel("Time (ns)")

    yticks = []
    ylabels = []
    for i, (entry_name, entry_states) in enumerate(grouped.items()):
        bars = states_to_bars(entry_states)
        colors = [label_colors[_clean_label(state["label"])]
                  for state in entry_states]
        y_pos = i * 10
        ax.broken_barh(bars, (y_pos, BAR_HEIGHT), facecolors=colors)
        yticks.append(y_pos + BAR_HEIGHT / 2)
        ylabels.append(entry_name)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-1, len(grouped) * 10)

    handles = [mpatches.Patch(color=color, label=label)
               for label, color in label_colors.items()]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    return fig
