from pydash import flatten

from experiments.latest.tools.cluster import PluralCluster
from experiments.latest.tools.mention import other, general


class PluralCorefState(list):
    def __init__(self, mentions, extract_gold=False, pfts=None, mpairs=None):
        list.__init__(self, mentions)

        # current mention index, pair features map {mention -> {antecedent mention -> pair features}}, idk what this is
        self.cmid, self.pfts, self.mpairs = 0, pfts, mpairs
        # all gold clusters, mention to gold clusters map, all auto clusters, mention to auto clusters map
        self.gCs, self.m2_gCs, self.aCs, self.m2_aCs = None, None, set(), {}

        # mentions which have been visited already
        self.amset = set()

        self.ambiguous_labels = ["#other#", "#general#"]

        if extract_gold:
            labels = set(flatten([m.gold_refs for m in mentions]))
            # label to gold clusters map
            self.m_l2c = dict([(l, PluralCluster()) for l in labels])

            for m in mentions:
                for gref in m.gold_refs:
                    if gref not in self.ambiguous_labels:
                        self.m_l2c[gref].append(m)

            self.m2_gCs = {m: [self.m_l2c[gref]
                               if gref not in self.ambiguous_labels
                               else PluralCluster([m])
                               for gref in m.gold_refs]
                           for m in mentions}

            self.gCs = list(set(flatten(list(self.m2_gCs.values()))))

        self.reset()

    def __hash__(self):
        return hash(tuple(self))

    def multi_link_wo_cfeats(self, preds):
        """
        Coref linking for plural model without cluster features
        :param preds: multi-class predictions
        :return: self
        """
        m = self[self.cmid]
        for a, pred in zip([other, general] + self[:self.cmid], preds):
            # predict antecedent needs to create new cluster
            if pred == 1:
                # if antecedent is "other" mention or "general" mention, put mention as a singleton
                if a.is_other() or a.is_general():
                    mc = PluralCluster([m])

                    if m in self.m2_aCs:
                        self.m2_aCs[m].append(mc)
                    else:
                        self.m2_aCs[m] = [mc]
                else:
                    if a in self.m2_aCs:
                        # antecedent cluster
                        ac = self.m2_aCs[a][0]
                        ac.append(m)

                        if m in self.m2_aCs and ac not in self.m2_aCs[m]:
                            self.m2_aCs[m].append(ac)
                        elif m not in self.m2_aCs:
                            self.m2_aCs[m] = [ac]
                    else:
                        mc = PluralCluster([a, m])

                        self.m2_aCs[a] = [mc]

                        if m in self.m2_aCs:
                            self.m2_aCs[m].append(mc)
                        else:
                            self.m2_aCs[m] = [mc]

                self.amset.add(m)
                self.amset.add(a)
            # predict current mention needs to create new cluster
            elif pred == 2:
                if m in self.m2_aCs:
                    mc = self.m2_aCs[m][0]
                    mc.append(a)

                    if a in self.m2_aCs and mc not in self.m2_aCs[a]:
                        self.m2_aCs[a].append(mc)
                    elif a not in self.m2_aCs:
                        self.m2_aCs[a] = [mc]
                else:
                    mc = PluralCluster([m, a])

                    self.m2_aCs[m] = [mc]

                    if a in self.m2_aCs:
                        self.m2_aCs[a].append(mc)
                    else:
                        self.m2_aCs[a] = [mc]

                self.amset.add(m)
                self.amset.add(a)

        return self

    def multi_link(self, preds, creprs=None, cp_reprs=None):
        """
        Coref linking for plural model with cluster features
        :param preds: predictions
        :param creprs: predicted cluster embeddings
        :param cp_reprs: predicted cluster pair embeddings
        :return: self
        """
        m = self[self.cmid]

        with_cfeats = True if creprs is not None and cp_reprs is not None else False

        zip_gen = zip([other, general] + self[:self.cmid], preds, creprs, cp_reprs) \
            if with_cfeats else zip([other, general] + self[:self.cmid], preds)

        for pred_tuple in zip_gen:
            if with_cfeats:
                a, pred, crepr, cp_repr = pred_tuple
            else:
                a, pred = pred_tuple
                crepr = None
                cp_repr = None

            # predict antecedent needs to create new cluster
            if pred == 1:
                # if antecedent is "other" mention or "general" mention, put mention as a singleton
                if a.is_other() or a.is_general():
                    mc = PluralCluster([m])

                    if with_cfeats:
                        mc.repr = crepr
                        mc.pair_repr = cp_repr

                    if m in self.m2_aCs:
                        self.m2_aCs[m].append(mc)
                    else:
                        self.m2_aCs[m] = [mc]
                else:
                    if a in self.m2_aCs:
                        # antecedent cluster
                        ac = self.m2_aCs[a][0]
                        ac.append(m)

                        if with_cfeats:
                            ac.repr = crepr
                            ac.pair_repr = cp_repr

                        if m in self.m2_aCs and ac not in self.m2_aCs[m]:
                            self.m2_aCs[m].append(ac)
                        elif m not in self.m2_aCs:
                            self.m2_aCs[m] = [ac]
                    else:
                        mc = PluralCluster([a, m])

                        if with_cfeats:
                            mc.repr = crepr
                            mc.pair_repr = cp_repr

                        self.m2_aCs[a] = [mc]

                        if m in self.m2_aCs:
                            self.m2_aCs[m].append(mc)
                        else:
                            self.m2_aCs[m] = [mc]

                self.amset.add(m)
                self.amset.add(a)
            # predict current mention needs to create new cluster
            elif pred == 2:
                if m in self.m2_aCs:
                    mc = self.m2_aCs[m][0]
                    mc.append(a)

                    if with_cfeats:
                        mc.repr = crepr
                        mc.pair_repr = cp_repr

                    if a in self.m2_aCs and mc not in self.m2_aCs[a]:
                        self.m2_aCs[a].append(mc)
                    elif a not in self.m2_aCs:
                        self.m2_aCs[a] = [mc]
                else:
                    mc = PluralCluster([m, a])

                    if with_cfeats:
                        mc.repr = crepr
                        mc.pair_repr = cp_repr

                    self.m2_aCs[m] = [mc]

                    if a in self.m2_aCs:
                        self.m2_aCs[a].append(mc)
                    else:
                        self.m2_aCs[a] = [mc]

                self.amset.add(m)
                self.amset.add(a)

        return self

    def create_singletons(self):
        mset = set(self)
        diffset = mset - self.amset

        for dm in diffset:
            mc = PluralCluster([dm])
            self.m2_aCs[dm] = [mc]

    def auto_clusters(self):
        for c in flatten(list(self.m2_aCs.values())):
            self.aCs.add(c)

        return list(self.aCs)

    def advance(self):
        self.cmid += 1
        return self

    def current(self):
        return self[:self.cmid], self[self.cmid]

    def reset(self):
        self.cmid = 0
        self.aCs = set()
        self.m2_aCs = {}
        self.amset = set()

        return self

    def done(self):
        return self.cmid >= len(self)
