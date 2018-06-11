from experiments.baseline.system import BaselineSystem
from experiments.latest.system import LatestSystem
from experiments.sing_only.system import SingOnlySystem


class PluralResolutionDemo:
    def __init__(self, iteration_num=1, demo_only=True):
        self.sing_only_system = SingOnlySystem(iteration_num, use_test_params=demo_only)
        self.baseline_system = BaselineSystem(iteration_num, use_test_params=demo_only)
        self.latest_system = LatestSystem(iteration_num, use_test_params=demo_only)

    def exe(self):
        self._run_sing_only()
        self._run_baseline()
        self._run_latest()

    def _run_sing_only(self):
        self.sing_only_system.run()

    def _run_baseline(self):
        self.baseline_system.run()

    def _run_latest(self):
        self.latest_system.run()
