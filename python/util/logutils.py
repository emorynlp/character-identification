import os
import logging

from constants.paths import Paths
from definitions import ROOT_DIR
from util.pathutil import to_dir_name


def init_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def init_log_package_for_run(experiment_type, iteration_num):
    experiment_path = to_dir_name(ROOT_DIR) + Paths.Logs.get_log_dir() + to_dir_name(experiment_type.value)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    iteration_dir_name = Paths.Logs.get_iteration_dir_name(iteration_num)
    iteration_path = experiment_path + to_dir_name(iteration_dir_name)

    if not os.path.exists(iteration_path):
        os.mkdir(iteration_path)
    # else:
    #     msg = "Run {0} of the {1} approach has already been initiated".format(iteration_num, experiment_type.name)
    #     raise Exception(msg)
