from constants import *
from util.pathutil import *


class Paths(object):
    class Resources(object):
        _dir = "resources/"

        GenderData = _dir + 'gender.data'
        Fasttext50d = _dir + 'fast_50d.bin'
        Fasttext100d = _dir + 'fast_100d.bin'
        Word2vec50d = _dir + 'w2v_50d.bin'
        AnimateUnigram = _dir + 'animate.unigrams.txt'
        InanimateUnigram = _dir + 'inanimate.unigrams.txt'
        SingularPersonalNouns = _dir + "singular_personal_nouns.txt"

    class Transcripts(object):
        _dir = "data/enhanced-jsons/"
        _transcript_name_template = "friends_season_{0:0>2}.json"
        _num_of_seasons = 4

        @staticmethod
        def get_input_transcript_paths():
            transcript_name_template = Paths.Transcripts._transcript_name_template
            return [
                (
                    Paths.Transcripts._dir + transcript_name_template.format(s + 1),
                    range(1, 20),
                    range(20, 22)
                )
                for s in range(Paths.Transcripts._num_of_seasons)
            ]

    class CorefModels(object):
        _dir = "trained_models/"
        _model_name_template = "{0}.f1-4.{1}.m"
        _feat_map_name_template = "{0}.f1-4.{1}.ft.p"

        @staticmethod
        def get_model_export_path(experiment_type, iteration_num):
            return Paths.CorefModels._dir + \
                   Paths.CorefModels._model_name_template.format(experiment_type.value, iteration_num)

        @staticmethod
        def get_feat_map_export_path(experiment_type, iteration_num):
            return Paths.CorefModels._dir + \
                   Paths.CorefModels._feat_map_name_template.format(experiment_type.value, iteration_num)

    class Params(object):
        _dir = "params/"
        _params_name_template = "{0}-{1}-params.json"

        @staticmethod
        def get_params_path(experiment_type, subsystem_type):
            if experiment_type.name not in ExperimentTypes.__members__:
                assert 0, "Bad experiment type: " + experiment_type

            if subsystem_type.name not in SubsystemTypes.__members__:
                assert 0, "Bad subsystem type: " + subsystem_type

            return Paths.Params._dir + \
                Paths.Params._params_name_template.format(experiment_type.value, subsystem_type.value)

        @staticmethod
        def get_test_params_path(experiment_type, subsystem_type):
            if experiment_type.name not in ExperimentTypes.__members__:
                assert 0, "Bad experiment type: " + experiment_type.name

            if subsystem_type.name not in SubsystemTypes.__members__:
                assert 0, "Bad subsystem type: " + subsystem_type.name

            return Paths.Params._dir + \
                "test-" + \
                Paths.Params._params_name_template.format(experiment_type.value, subsystem_type.value)

    class Logs(object):
        _dir = "logs/"
        _log_name_template = "{0}-{1}.{2}.log"
        _iteration_name_template = "run-{0:0>3}"

        @staticmethod
        def get_log_dir():
            return Paths.Logs._dir

        @staticmethod
        def get_log_path(experiment_type, subsystem_type, iteration_num):
            if experiment_type.name not in ExperimentTypes.__members__:
                assert 0, "Bad experiment type: " + experiment_type.name

            if subsystem_type.name not in SubsystemTypes.__members__:
                assert 0, "Bad subsystem type: " + subsystem_type.name

            return Paths.Logs._dir + \
                to_dir_name(experiment_type.value) + \
                to_dir_name(Paths.Logs.get_iteration_dir_name(iteration_num)) + \
                Paths.Logs._log_name_template.format(experiment_type.value, subsystem_type.value, iteration_num)

        @staticmethod
        def get_iteration_dir_name(iteration_num):
            return Paths.Logs._iteration_name_template.format(iteration_num)
