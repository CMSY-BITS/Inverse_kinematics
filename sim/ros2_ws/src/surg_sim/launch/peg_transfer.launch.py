import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import generate_task_launch_description  # noqa: E402


def generate_launch_description():
    return generate_task_launch_description("peg_transfer", "peg_transfer_node")
