# engine/executor.py
from typing import List

class Executor:
    """
    Determines execution order for scripts.
    Can optionally simulate execution.
    """
    def __init__(self, execution_order: List[str]):
        self.execution_order = execution_order

    def run_simulation(self):
        print("Simulated execution order:")
        for idx, f in enumerate(self.execution_order, 1):
            print(f"{idx}. {f}")
