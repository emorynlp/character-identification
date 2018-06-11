from constants import ExperimentTypes
from constants.paths import Paths
from experiments.latest.model.coref import NoClusterFeatsPluralACNN
from experiments.latest.model.linking import MentionClusterEntityLinker
from experiments.latest.model.linking_joint import JointMentionClusterEntityLinker
from experiments.latest.tools.ioutils import SpliceReader, StateWriter
from experiments.latest.tools.mention import init_super_mentions
from experiments.latest.tools.state import PluralCorefState
from experiments.latest.tools.evaluators import *
from experiments.system import ExperimentSystem
from util import *
from util.pathutil import *


class LatestSystem(ExperimentSystem):
    def __init__(self, iteration_num=1, use_test_params=True):
        ExperimentSystem.__init__(self, iteration_num, use_test_params)

    def _experiment_type(self):
        return ExperimentTypes.LATEST

    def _load_transcripts(self):
        data_in = Paths.Transcripts.get_input_transcript_paths()

        self.timer.start("load_transcript")
        reader = SpliceReader()
        spks, poss, deps, ners = set(), set(), set(), set()
        for d_in in data_in:
            es, ms = reader.read_season_json(d_in[0])

            spks.update(TranscriptUtils.collect_speakers(es))
            poss.update(TranscriptUtils.collect_pos_tags(es))
            ners.update(TranscriptUtils.collect_ner_tags(es))
            deps.update(TranscriptUtils.collect_dep_labels(es))

            keys, d_trn, d_dev, d_tst = set(), dict(), dict(), dict()

            for m in ms:
                eid = m.tokens[0].parent_episode().id
                sid = m.tokens[0].parent_scene().id

                target = d_trn if eid in d_in[1] \
                    else d_dev if eid in d_in[2] \
                    else d_tst

                key = eid * 100 + sid
                if key not in target:
                    target[key] = []
                target[key].append(m)
                keys.add(key)

            for key in sorted(keys):
                if key in d_trn:
                    self.trn_coref_states.append(PluralCorefState(d_trn[key], extract_gold=True))
                if key in d_dev:
                    self.dev_coref_states.append(PluralCorefState(d_dev[key], extract_gold=True))
                if key in d_tst:
                    self.tst_coref_states.append(PluralCorefState(d_tst[key], extract_gold=True))
            self.coref_logger.info("Transcript loaded: %s w/ %d mentions" % (d_in[0], len(ms)))

        trnc = sum(map(len, self.trn_coref_states))
        devc = sum(map(len, self.dev_coref_states))
        tstc = sum(map(len, self.tst_coref_states))

        self.coref_logger.info(
            "%d transcript(s) loaded with %d speakers and %d mentions (Trn/Dev/Tst: %d(%d)/%d(%d)/%d(%d)) - %.2fs\n"
            % (
                len(data_in),
                len(spks),
                trnc + devc + tstc,
                len(self.trn_coref_states),
                trnc,
                len(self.dev_coref_states),
                devc,
                len(self.tst_coref_states),
                tstc,
                self.timer.end("load_transcript")
            )
        )

        return spks, poss, deps, ners

    def run_coref(self, seed_path=""):
        spks, poss, deps, ners = self._load_transcripts()
        self._extract_coref_features(spks, poss, deps, ners)
        eftdims, mftdim, pftdim = self._get_coref_feature_shapes()

        init_super_mentions(eftdims, mftdim, pftdim)

        model = NoClusterFeatsPluralACNN(eftdims,
                                         mftdim,
                                         pftdim,
                                         self.coref_params["number_of_filters"],
                                         self.coref_params["gpu_number"],
                                         self.coref_logger,
                                         gpu=self.coref_params["gpu_settings"])

        if type(seed_path) == str and len(seed_path) == 0:
            model.train_ranking(self.trn_coref_states,
                                self.dev_coref_states,
                                nb_epoch=self.coref_params["number_of_epochs"],
                                batch_size=self.coref_params["batch_size"],
                                model_out=self.coref_model_save_path)
        else:
            model.load_model_weights(self.coref_model_save_path)

        # Model evaluation
        self.coref_logger.info('\nEvaluating trained model on Tst')
        model.decode_clusters([s.reset() for s in self.tst_coref_states])

        for s in self.tst_coref_states:
            s.create_singletons()

        golds, autos, = [s.gCs for s in self.tst_coref_states], [s.auto_clusters() for s in self.tst_coref_states]

        p, r, f = BCubeEvaluator().evaluate_documents(golds, autos)
        self.coref_logger.info('Bcube - %.4f/%.4f/%.4f' % (p, r, f))

        p, r, f = CeafeEvaluator().evaluate_documents(golds, autos)
        self.coref_logger.info('Ceafe - %.4f/%.4f/%.4f' % (p, r, f))

        p, r, f = BlancEvaluator().evaluate_documents(golds, autos)
        self.coref_logger.info('Blanc - %.4f/%.4f/%.4f' % (p, r, f))

    def extract_learned_coref_features(self):
        eftdims, mftdim, pftdim = self._get_coref_feature_shapes()

        model = NoClusterFeatsPluralACNN(eftdims,
                                         mftdim,
                                         pftdim,
                                         self.coref_params["number_of_filters"],
                                         self.coref_params["gpu_number"],
                                         self.export_clusters_logger,
                                         gpu=self.coref_params["gpu_settings"])

        model.load_model_weights(self.coref_model_save_path)

        all_states = sum([self.trn_coref_states, self.dev_coref_states, self.tst_coref_states], [])
        ms = sum(all_states, [])

        m_efts = np.array([m.feat_map['efts'] for m in ms])

        m_mfts = np.array([m.feat_map['mft'] for m in ms])
        m_efts = [np.stack(m_efts[:, g]) for g in range(len(m_efts[0]))]

        for m, r in zip(ms, model.get_mreprs(m_efts + [m_mfts])):
            m.feat_map['mrepr'] = r

        for s in all_states:
            pairs, s.mpairs = [], {m: dict() for m in s}
            m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [[] for _ in range(4)], [[] for _ in range(4)], [], [], []

            if len(s) > 1:
                for i, cm in enumerate(s[1:], 1):
                    cefts, cmft = cm.feat_map['efts'], cm.feat_map['mft']
                    for am in s[:i]:
                        pefts, pmft, pft = am.feat_map['efts'], am.feat_map['mft'], s.pfts[am][cm]

                        for l, e in zip(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts],
                                        pefts + cefts + [pmft, cmft, pft]):
                            l.append(e)

                        pairs.append((am, cm))

                m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [np.array(g) for g in m1_efts], \
                                                              [np.array(g) for g in m2_efts], \
                                                              np.array(m1_mfts), \
                                                              np.array(m2_mfts), \
                                                              np.array(mp_pfts)

                mpairs = model.get_mpairs(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts])
                for mp, (am, cm) in zip(mpairs, pairs):
                    s.mpairs[am][cm] = mp

    def run_entity_linking(self):
        self.entity_linking_logger.info("Beginning baseline entity linker...")
        self.entity_linking_logger.info("-" * 40)

        self._run_baseline_linking()

        self.entity_linking_logger.info("Beginning joint entity linker...")
        self.entity_linking_logger.info("-" * 40)

        self._run_joint_linking()

    def _run_joint_linking(self):
        all_states = sum([self.trn_coref_states, self.dev_coref_states, self.tst_coref_states], [])

        for m in sum(all_states, []):
            m.gold_refs = [self.other_label
                           if gref.lower() not in self.linking_labels
                           else gref.lower()
                           for gref in m.gold_refs]

        m1, m2 = self.trn_coref_states[0][0], self.trn_coref_states[0][1]
        mrepr_dim = len(m1.feat_map['mrepr'])
        mpair_dim = len(self.trn_coref_states[0].mpairs[m1][m2])

        model = JointMentionClusterEntityLinker(self.linking_params["number_of_filters"],
                                                mrepr_dim,
                                                mpair_dim,
                                                self.linking_labels,
                                                self.entity_linking_logger,
                                                gpu=self.linking_params["gpu_settings"])

        model.train_linking(self.trn_coref_states,
                            self.dev_coref_states,
                            nb_epoch=self.linking_params["number_of_epochs"],
                            batch_size=self.linking_params["batch_size"],
                            model_out="")

        self.entity_linking_logger.info('\nEvaluating trained model')
        scorer = LinkingMicroF1Evaluator(self.linking_labels)
        model.do_linking(self.tst_coref_states)
        scores = scorer.evaluate_states(self.tst_coref_states)
        avg = np.mean(list(scores.values()), axis=0)

        sacc, pacc = model.accuracy(self.tst_coref_states)
        self.entity_linking_logger.info('Test accuracy: %.4f/%.4f\n' % (sacc, pacc))
        for l, s in scores.items():
            self.entity_linking_logger.info("%10s : %.4f %.4f %.4f" % (l, s[0], s[1], s[2]))
        self.entity_linking_logger.infoprint('\n%10s : %.4f %.4f %.4f' % ('avg', avg[0], avg[1], avg[2]))

        macro_scorer = LinkingMacroF1Evaluator()
        p, r, f = macro_scorer.evaluate_states(self.tst_coref_states)
        self.entity_linking_logger.info("\n%10s : %.4f %.4f %.4f" % ("macro", p, r, f))

        results_file = "joint-linking-results.txt"
        results_path = Paths.Logs.get_log_dir() + \
            to_dir_name(Paths.Logs.get_iteration_dir_name(self.iteration_num)) + \
            results_file

        writer = StateWriter()
        writer.open_file(results_path)
        writer.write_states(self.tst_coref_states)

    def _run_baseline_linking(self):
        all_states = sum([self.trn_coref_states, self.dev_coref_states, self.tst_coref_states], [])

        for m in sum(all_states, []):
            m.gold_refs = [self.other_label
                           if gref.lower() not in self.linking_labels
                           else gref.lower()
                           for gref in m.gold_refs]

        m1, m2 = self.trn_coref_states[0][0], self.trn_coref_states[0][1]
        mrepr_dim = len(m1.feat_map['mrepr'])
        mpair_dim = len(self.trn_coref_states[0].mpairs[m1][m2])

        model = MentionClusterEntityLinker(self.linking_params["number_of_filters"],
                                           mrepr_dim,
                                           mpair_dim,
                                           self.linking_labels,
                                           self.entity_linking_logger,
                                           gpu=self.linking_params["gpu_settings"])

        model.train_linking(self.trn_coref_states,
                            self.dev_coref_states,
                            nb_epoch=self.linking_params["number_of_epochs"],
                            batch_size=self.linking_params["batch_size"],
                            model_out="")

        self.entity_linking_logger.info('\nEvaluating trained model')
        scorer = LinkingMicroF1Evaluator(self.linking_labels)
        model.do_linking(self.tst_coref_states)
        scores = scorer.evaluate_states(self.tst_coref_states)
        avg = np.mean(list(scores.values()), axis=0)

        self.entity_linking_logger.info('Test accuracy: %.4f\n' % model.accuracy(self.tst_coref_states))
        for l, s in scores.items():
            self.entity_linking_logger.info("%10s : %.4f %.4f %.4f" % (l, s[0], s[1], s[2]))
        self.entity_linking_logger.info('\n%10s : %.4f %.4f %.4f' % ('avg', avg[0], avg[1], avg[2]))

        macro_scorer = LinkingMacroF1Evaluator()
        p, r, f = macro_scorer.evaluate_states(self.tst_coref_states)
        self.entity_linking_logger.info("\n%10s : %.4f %.4f %.4f" % ("macro", p, r, f))

        results_file = "baseline-linking-results.txt"
        results_path = Paths.Logs.get_log_dir() + \
            to_dir_name(Paths.Logs.get_iteration_dir_name(self.iteration_num)) + \
            results_file

        writer = StateWriter()
        writer.open_file(results_path)
        writer.write_states(self.tst_coref_states)
