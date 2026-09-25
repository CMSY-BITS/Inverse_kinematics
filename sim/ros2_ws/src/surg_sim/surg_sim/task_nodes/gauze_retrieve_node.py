"""GauzeRetrieve: grasp a piece of gauze and lift it clear of the tissue
bed. Adds jaw actuation (not modeled in the base node's 6-DoF planner yet)
on top of NeedleReach-style positioning."""
import numpy as np

from .base_task_node import BaseTaskNode, spin_task_node


class GauzeRetrieveNode(BaseTaskNode):
    task_name = "gauze_retrieve"

    def _load_goal(self) -> np.ndarray:
        # TODO: gauze pose from the world plugin; TODO: jaw close command
        # once jaw_angle (joint 7) is added to BaseTaskNode's action space.
        return np.array([-0.03, 0.04, 0.13])


def main():
    spin_task_node(GauzeRetrieveNode)


if __name__ == "__main__":
    main()
