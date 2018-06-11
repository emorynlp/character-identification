import pickle
from abc import ABC, abstractmethod

from component.features import MentionFeatureExtractor
from constants import SubsystemTypes
from util import Timer
from util.loadutils import *
from util.logutils import *


class ExperimentSystem(ABC):
    def __init__(self, iteration_num=1, use_test_params=True):
        self.iteration_num = iteration_num

        self.coref_logger, self.export_clusters_logger, self.entity_linking_logger = self.init_system_logging()

        self.coref_params, self.linking_params = self.init_params(use_test_params=use_test_params)

        self.coref_feat_map_save_path = Paths.CorefModels.get_feat_map_export_path(self._experiment_type(), self.iteration_num)
        self.coref_model_save_path = Paths.CorefModels.get_model_export_path(self._experiment_type(), self.iteration_num)

        self.timer = Timer()

        self.trn_coref_states = []
        self.dev_coref_states = []
        self.tst_coref_states = []

        self.other_label = "#other#"
        self.general_label = "#general#"
        self.linking_labels = ['monica geller', 'judy geller', 'jack geller', 'lily buffay', 'rachel green',
                               'joey tribbiani', 'phoebe buffay', 'carol willick', 'ross geller', 'chandler bing',
                               'gunther', 'ben geller', 'barry farber', 'richard burke', 'kate miller', 'peter becker',
                               'emily waltham'] + [self.other_label, self.general_label]

    def init_params(self, use_test_params=True):
        if use_test_params:
            coref_params_path = Paths.Params.get_test_params_path(self._experiment_type(), SubsystemTypes.COREF)
            linking_params_path = Paths.Params.get_test_params_path(self._experiment_type(), SubsystemTypes.ENTITY_LINKING)
        else:
            coref_params_path = Paths.Params.get_params_path(self._experiment_type(), SubsystemTypes.COREF)
            linking_params_path = Paths.Params.get_params_path(self._experiment_type(), SubsystemTypes.ENTITY_LINKING)

        coref_params = load_json_from_path(coref_params_path)
        linking_params = load_json_from_path(linking_params_path)

        return coref_params, linking_params

    def init_system_logging(self):
        init_log_package_for_run(self._experiment_type(), self.iteration_num)

        coref_logger = init_logger(
            "%s.%s" % (self.__class__.__name__, SubsystemTypes.COREF),
            Paths.Logs.get_log_path(self._experiment_type(), SubsystemTypes.COREF, self.iteration_num)
        )
        export_clusters_logger = init_logger(
            "%s.%s" % (self.__class__.__name__, SubsystemTypes.EXPORT_CLUSTERS),
            Paths.Logs.get_log_path(self._experiment_type(), SubsystemTypes.EXPORT_CLUSTERS, self.iteration_num)
        )
        entity_linking_logger = init_logger(
            "%s.%s" % (self.__class__.__name__, SubsystemTypes.ENTITY_LINKING),
            Paths.Logs.get_log_path(self._experiment_type(), SubsystemTypes.ENTITY_LINKING, self.iteration_num)
        )

        return coref_logger, export_clusters_logger, entity_linking_logger

    def set_model_iteration(self, model_num):
        self.iteration_num = model_num

    def set_feat_map_save_path(self, save_path):
        self.coref_feat_map_save_path = save_path

    def set_coref_model_save_path(self, save_path):
        self.coref_model_save_path = save_path

    @abstractmethod
    def _experiment_type(self):
        pass

    @abstractmethod
    def _load_transcripts(self):
        pass

    def _load_coref_resources(self):
        self.timer.start("load_w2v")
        w2v = load_word_vecs()
        self.coref_logger.info("Fasttext data loaded - %.2fs" % self.timer.end("load_w2v"))

        self.timer.start("load_w2g")
        w2g = load_gender_data()
        self.coref_logger.info("Gender data loaded   - %.2fs" % self.timer.end("load_w2g"))

        self.timer.start("load_animacy_dicts")
        ani = load_animate_data()
        ina = load_inanimate_data()
        self.coref_logger.info("Animacy data loaded  - %.2fs" % self.timer.end("load_animacy_dicts"))

        return w2v, w2g, ani, ina

    def _extract_coref_features(self, spks, poss, ners, deps, save_feats=True):
        w2v, w2g, ani, ina = self._load_coref_resources()
        feat_extractor = MentionFeatureExtractor(w2v, w2g, spks, poss, ners, deps, ani, ina)

        self.timer.start("feature_extraction")
        for s in sum([self.trn_coref_states, self.dev_coref_states, self.tst_coref_states], []):
            s.pfts = {m: dict() for m in s}

            for i, m in enumerate(s):
                m.id, (efts, mft) = i, feat_extractor.extract_mention(m)
                m.feat_map['efts'], m.feat_map['mft'] = efts, mft

                for a in s[:i]:
                    s.pfts[a][m] = feat_extractor.extract_pairwise(a, m)
        self.coref_logger.info("Feature extracted    - %.2fs\n" % self.timer.end("feature_extraction"))

        if save_feats:
            self.timer.start("dump_feature_extractor")

            with open(self.coref_feat_map_save_path, 'wb') as fout:
                pickle.dump(feat_extractor, fout, protocol=2)

            self.coref_logger.info("Feature extractor saved to %s - %.2fs" %
                                   (self.coref_feat_map_save_path, self.timer.end("dump_feature_extractor")))

    def _get_coref_feature_shapes(self):
        m1, m2 = self.trn_coref_states[0][1], self.trn_coref_states[0][2]
        efts, mft = m1.feat_map["efts"], m1.feat_map["mft"]

        eftdims = list(map(lambda x: x.shape, efts))
        mftdim, pftdim = len(mft), len(self.trn_coref_states[0].pfts[m1][m2])

        return eftdims, mftdim, pftdim

    @abstractmethod
    def run_coref(self):
        pass

    @abstractmethod
    def extract_learned_coref_features(self):
        pass

    @abstractmethod
    def run_entity_linking(self):
        pass

    def run(self):
        self.run_coref()
        self.extract_learned_coref_features()
        self.run_entity_linking()
