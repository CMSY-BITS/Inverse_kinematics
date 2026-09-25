"""NeedlePick: grasp the needle at a specific pose along its curve and lift
it — the highest-precision of the four tasks, since grasp point and needle
orientation both matter, not just tip position."""
import numpy as np

from .base_task_node import BaseTaskNode, spin_task_node


class NeedlePickNode(BaseTaskNode):
    task_name = "needle_pick"

    def _load_goal(self) -> np.ndarray:
        # TODO: needle pose + grasp-point offset from the world plugin;
        # TODO: orient the wrist along the needle's local tangent (needs
        # orientation, not just position, in the CEM cost — see
        # eval/metrics.py:orientation_error_deg for how it's scored).
        return np.array([0.01, 0.02, 0.145])


def main():
    spin_task_node(NeedlePickNode)


if __name__ == "__main__":
    main()
