"""NeedleReach: move the tool tip to the needle's grasp point without
grasping. Simplest of the four tasks — a pure reaching motion, good for the
first ID/OOD smoke test of a new policy."""
import numpy as np

from .base_task_node import BaseTaskNode, spin_task_node


class NeedleReachNode(BaseTaskNode):
    task_name = "needle_reach"

    def _load_goal(self) -> np.ndarray:
        # TODO: subscribe to the needle model's pose from the Gazebo world
        # plugin instead of a fixed point once worlds/needle_reach.sdf's
        # task-prop pose publisher is wired in.
        return np.array([0.04, 0.01, 0.14])


def main():
    spin_task_node(NeedleReachNode)


if __name__ == "__main__":
    main()
