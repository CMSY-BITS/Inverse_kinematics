"""PegTransfer: move a ring/peg from one post to another. Two-phase goal
(approach source, then approach target) — the standard dVRK training task,
here as a two-waypoint sequence for the planner."""
import numpy as np

from .base_task_node import BaseTaskNode, spin_task_node


class PegTransferNode(BaseTaskNode):
    task_name = "peg_transfer"

    def __init__(self):
        self._phase = 0  # 0: approach source peg, 1: approach target peg
        super().__init__()

    def _load_goal(self) -> np.ndarray:
        # TODO: source/target peg poses from the world plugin; TODO: phase
        # transition (0 -> 1) once tip error < success threshold and jaw
        # has closed on the ring.
        waypoints = [np.array([0.02, -0.03, 0.14]), np.array([-0.02, 0.03, 0.14])]
        return waypoints[self._phase]


def main():
    spin_task_node(PegTransferNode)


if __name__ == "__main__":
    main()
