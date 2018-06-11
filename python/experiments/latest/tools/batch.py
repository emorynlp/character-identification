import numpy as np
from experiments.latest.tools.mention import other, general
from experiments.latest.tools.cluster import PluralCluster
from experiments.latest.tools.test import test_plural_batch_fidelity


def construct_batch(states):
    m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts, probs = [[] for _ in range(4)], [[] for _ in range(4)], [], [], [], []

    for s in [s.reset() for s in states]:
        ms, m2cs = [other, general] + s, s.m2_gCs

        # mention cluster is represented as set here
        m2dcs = {}
        p2drs = {}
        mset = set()

        for idx, m in enumerate(ms[2:], 2):
            cefts, cmft, ccs = m.feat_map['efts'], m.feat_map['mft'], m2cs[m]

            for a in ms[:idx]:
                pefts, pmft, acs = a.feat_map['efts'], a.feat_map['mft'], m2cs.get(a, [PluralCluster()])
                pft = s.pfts[a][m] if not a.is_other() and not a.is_general() else a.feat_map["pft"]

                # map(lambda l, i: l.append(i),
                #     [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts],
                #     [pefts, cefts, pmft, cmft, pft])

                for l, i in zip(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], pefts + cefts + [pmft, cmft, pft]):
                    l.append(i)

                if not a.plural and m.plural and acs[0] in ccs:
                    if m in p2drs:
                        p2drs[m].append(a.gold_refs[0])
                    else:
                        p2drs[m] = [a.gold_refs[0]]

                    probs.append(1)
                elif a.plural and not m.plural and ccs[0] in acs and m.gold_refs[0] not in p2drs.get(a, []):
                    if a in p2drs:
                        p2drs[a].append(m.gold_refs[0])
                    else:
                        p2drs[a] = [m.gold_refs[0]]

                    probs.append(2)
                elif not a.plural and not m.plural and acs[0] == ccs[0]:
                    probs.append(1)
                elif a.is_other() and "#other#" in m.gold_refs:
                    probs.append(1)
                elif a.is_general() and len(m.gold_refs) == 1 and m.gold_refs[0] == "#general#":
                    probs.append(1)
                else:
                    probs.append(0)

                pred = probs[-1]

                # predict antecedent needs to create new cluster
                if pred == 1:
                    if a.is_other() or a.is_general():
                        mc = PluralCluster([m])
                        if m in m2dcs:
                            m2dcs[m].append(mc)
                        else:
                            m2dcs[m] = [mc]
                    else:
                        if a in m2dcs:
                            ante_cs = m2dcs[a]
                            ante_cs[0].append(m)

                            if m in m2dcs and ante_cs[0] not in m2dcs[m]:
                                m2dcs[m].append(ante_cs[0])
                            elif m not in m2dcs:
                                m2dcs[m] = [ante_cs[0]]
                        else:
                            mc = PluralCluster([a, m])
                            m2dcs[a] = [mc]

                            if m in m2dcs:
                                m2dcs[m].append(mc)
                            else:
                                m2dcs[m] = [mc]

                    mset.add(m)
                    mset.add(a)
                # predict current mention needs to create new cluster
                elif pred == 2:
                    if m in m2dcs:
                        m2dcs[m][0].append(a)

                        if a in m2dcs and m2dcs[m][0] not in m2dcs[a]:
                            m2dcs[a].append(m2dcs[m][0])
                        elif a not in m2dcs:
                            m2dcs[a] = [m2dcs[m][0]]
                    else:
                        mc = PluralCluster([m, a])
                        m2dcs[m] = [mc]

                        if a in m2dcs:
                            m2dcs[a].append(mc)
                        else:
                            m2dcs[a] = [mc]

                    mset.add(m)
                    mset.add(a)

        diffset = set(s) - mset

        for dm in diffset:
            m2dcs[dm] = [PluralCluster([dm])]

        test_plural_batch_fidelity(m2dcs, s.m2_gCs)

    print("Clusters recreated with 100% fidelity.")

    # c, probs = len(m1_efts[0]), np.array(probs)
    # m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = map(np.array, [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts])
    # m1_efts, m2_efts = [np.stack(m1_efts[:, g]) for g in range(c)], [np.stack(m2_efts[:, g]) for g in range(c)]

    c, probs = len(m1_efts), np.array(probs)
    m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [np.array(g) for g in m1_efts], \
                                                  [np.array(g) for g in m2_efts], \
                                                  np.array(m1_mfts), \
                                                  np.array(m2_mfts), \
                                                  np.array(mp_pfts)

    return m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], probs[:, np.newaxis]


class BatchTrainer(object):
    def __init__(self, states):
        self.states = states
        self.state2dynamic_ref_tables = {s: {} for s in states}

    def construct_dynamic_batch(self):
        ntdone = [s for s in self.states if not s.done()]
        return get_features(ntdone)[0], get_training_labels(ntdone, self.state2dynamic_ref_tables)

        # creprs, cp_reprs, m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [], [], [], [], [], [], []
        # probs = []
        #
        # for s in ntdone:
        #     p2drs = self.state2dynamic_ref_tables[s]
        #
        #     antes, m = s.current()
        #     antes = [other, general] + antes
        #
        #     i, cefts, cmft, ccs = len(antes), m.feat_map['efts'], m.feat_map['mft'], s.m2_gCs[m]
        #
        #     for a in antes:
        #         pefts, pmft, acs = a.feat_map['efts'], a.feat_map['mft'], s.m2_gCs.get(a, [FullMentionCluster()])
        #         pft = s.pfts[a][m] if not a.is_other() and not a.is_general() else a.feat_map["pft"]
        #
        #         crepr, cp_repr = get_crepr(a, m, s, s.m2_aCs)
        #
        #         map(lambda l, e: l.append(e),
        #             [creprs, cp_reprs, m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts],
        #             [crepr, cp_repr, pefts, cefts, pmft, cmft, pft])
        #
        #         if not a.plural and m.plural and acs[0] in ccs:
        #             if m in p2drs:
        #                 p2drs[m].append(a.gold_refs[0])
        #             else:
        #                 p2drs[m] = [a.gold_refs[0]]
        #
        #             probs.append(1)
        #         elif a.plural and not m.plural and ccs[0] in acs and m.gold_refs[0] not in p2drs.get(a, []):
        #             if a in p2drs:
        #                 p2drs[a].append(m.gold_refs[0])
        #             else:
        #                 p2drs[a] = [m.gold_refs[0]]
        #
        #             probs.append(2)
        #         elif not a.plural and not m.plural and acs[0] == ccs[0]:
        #             probs.append(1)
        #         elif a.is_other() and "#other#" in m.gold_refs:
        #             probs.append(1)
        #         elif a.is_general() and len(m.gold_refs) == 1 and m.gold_refs[0] == "#general#":
        #             probs.append(1)
        #         else:
        #             probs.append(0)
        #
        # c, probs = len(m1_efts[0]), np.array(probs)
        # creprs, cp_reprs = np.array(creprs), np.array(cp_reprs)
        # m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = map(np.array, [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts])
        # m1_efts, m2_efts = [np.stack(m1_efts[:, g]) for g in range(c)], [np.stack(m2_efts[:, g]) for g in range(c)]
        #
        # dat = [creprs, cp_reprs] + m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], probs[:, np.newaxis]
        #
        # return get_features(ntdone)[0], get_training_labels(ntdone, self.ref_tables)
        #
        # for d, nd in zip(dat[0], new_dat[0]):
        #     if not np.array_equal(d, nd):
        #         raise Exception("Old and new data are not the same")
        #
        # if not np.array_equal(dat[1], new_dat[1]):
        #     raise Exception("Old and new data are not the same")
        #
        # return dat

    def advance(self):
        for s in self.states:
            s.advance()

    def reset(self):
        self.state2dynamic_ref_tables = {s: {} for s in self.states}

        for s in self.states:
            s.reset()

    def done(self):
        return True if len([s for s in self.states if not s.done()]) == 0 else False


def get_features(states):
    creprs, cp_reprs, m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts, pranges = [], [], [], [], [], [], [], []

    for s in states:
        antes, m = s.current()
        antes = [other, general] + antes
        i, cefts, cmft = len(antes), m.feat_map['efts'], m.feat_map['mft']

        pranges.append((len(mp_pfts), len(mp_pfts) + i))
        for a in antes:
            pefts, pmft = a.feat_map['efts'], a.feat_map['mft']
            pft = s.pfts[a][m] if not a.is_other() and not a.is_general() else a.feat_map["pft"]

            crepr, cp_repr = get_crepr(a, m, s, s.m2_aCs)

            map(lambda l, e: l.append(e),
                [creprs, cp_reprs, m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts],
                [crepr, cp_repr, pefts, cefts, pmft, cmft, pft])

    c = len(m1_efts[0])
    creprs, cp_reprs = np.array(creprs), np.array(cp_reprs)
    m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = map(np.array, [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts])
    m1_efts, m2_efts = [np.stack(m1_efts[:, g]) for g in range(c)], [np.stack(m2_efts[:, g]) for g in range(c)]

    return [creprs, cp_reprs] + m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], pranges


def get_training_labels(states, state2dynamic_ref_tables):
    probs = []

    for s in states:
        p2drs = state2dynamic_ref_tables[s]

        antes, m = s.current()
        antes = [other, general] + antes

        # clusters which contain current mention m
        ccs = s.m2_gCs[m]

        for a in antes:
            # clusters which contain antecedent mention a; if there are no clusters, then use empty cluster
            acs = s.m2_gCs.get(a, [PluralCluster()])

            if not a.plural and m.plural and acs[0] in ccs:
                if m in p2drs:
                    p2drs[m].append(a.gold_refs[0])
                else:
                    p2drs[m] = [a.gold_refs[0]]

                probs.append(1)
            elif a.plural and not m.plural and ccs[0] in acs and m.gold_refs[0] not in p2drs.get(a, []):
                if a in p2drs:
                    p2drs[a].append(m.gold_refs[0])
                else:
                    p2drs[a] = [m.gold_refs[0]]

                probs.append(2)
            elif not a.plural and not m.plural and acs[0] == ccs[0]:
                probs.append(1)
            elif a.is_other() and "#other#" in m.gold_refs:
                probs.append(1)
            elif a.is_general() and len(m.gold_refs) == 1 and m.gold_refs[0] == "#general#":
                probs.append(1)
            else:
                probs.append(0)

    return np.array(probs)[:, np.newaxis]


def get_crepr(antecedent, mention, state, m2dcs):
    a = antecedent
    m = mention

    mftdim = len(a.feat_map["mft"])
    pftdim = len(state.pfts[a][m]) if not a.is_other() and not a.is_general() else len(a.feat_map["pft"])
    filter_size = 280

    if a not in m2dcs:
        cluster_repr = np.zeros(shape=(filter_size + mftdim,)).astype("float32")
        cluster_pair_repr = np.zeros(shape=(filter_size + pftdim,)).astype("float32")
    else:
        cluster_mat = np.array([c.repr for c in m2dcs[a] if c.repr is not None], dtype="float32")
        cluster_repr = np.mean(cluster_mat, axis=0) \
            if len(cluster_mat) > 0 \
            else np.zeros(shape=(filter_size + mftdim,)).astype("float32")

        cluster_pair_mat = np.array([c.pair_repr for c in m2dcs[a] if c.pair_repr is not None], dtype="float32")
        cluster_pair_repr = np.mean(cluster_pair_mat, axis=0) \
            if len(cluster_pair_mat) > 0 \
            else np.zeros(shape=(filter_size + pftdim,)).astype("float32")

    return cluster_repr, cluster_pair_repr
