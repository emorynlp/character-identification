from abc import *
import numpy as np
from pydash import flatten_deep
from sklearn.utils.linear_assignment_ import linear_assignment


class AbstractEvaluator(object):
    @abstractmethod
    def evaluate_documents(self, gold_documents, auto_documents):
        return

    @abstractmethod
    def evaluate_clusters(self, gold_clusters, auto_clusters):
        return

    @staticmethod
    def f1_score(precision, recall):
        return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    @staticmethod
    def create_mention2cluster_map(clusters):
        m2cs = dict()
        for c in clusters:
            for m in c:
                if m in m2cs:
                    m2cs[m].append(c)
                else:
                    m2cs[m] = [c]
        return m2cs
        # return dict((m, c) for c in clusters for m in c)


class BCubeEvaluator(AbstractEvaluator):
    def evaluate_documents(self, gold_documents, auto_documents):
        return self.evaluate_clusters(sum(gold_documents, []), sum(auto_documents, []))

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        gold_m2c_map = self.create_mention2cluster_map(gold_clusters)
        auto_m2c_map = self.create_mention2cluster_map(auto_clusters)
        mentions = auto_m2c_map.keys()

        pc = rc = 0
        for mention in mentions:
            gcs = gold_m2c_map.get(mention)
            acs = auto_m2c_map.get(mention)

            # for each plural noun, use all clusters that contain said plural noun
            agg_gold_cluster = set(flatten_deep(gcs))
            agg_auto_cluster = set(flatten_deep(acs))

            correct = len(set(agg_gold_cluster).intersection(set(agg_auto_cluster)))
            pc += float(correct) / len(agg_auto_cluster) if agg_auto_cluster else 0.0
            rc += float(correct) / len(agg_gold_cluster) if agg_gold_cluster else 0.0

        p = pc / len(mentions)
        r = rc / len(mentions)

        return p, r, self.f1_score(p, r)


class BlancEvaluator(AbstractEvaluator):
    def evaluate_documents(self, gold_documents, auto_documents):
        # coreferent / non-coreferent indices
        c, n = 0, 1
        confusion = np.zeros((2, 2), dtype="int32")

        # get confusion matrix for each scene b/c model trains only on scene-level
        # i.e. cannot consider m-m links across scenes
        for gdoc, adoc in zip(gold_documents, auto_documents):
            confusion += self.evaluate_clusters(gdoc, adoc)

        print(confusion)

        pc = float(confusion[c, c]) / (confusion[c, c] + confusion[n, c]) \
            if confusion[c, c] + confusion[n, c] > 0 \
            else 0.0
        pn = float(confusion[n, n]) / (confusion[c, n] + confusion[n, n]) \
            if confusion[c, n] + confusion[n, n] > 0 \
            else 0.0
        p = float(pc + pn) / 2

        rc = float(confusion[c, c]) / (confusion[c, c] + confusion[c, n]) \
            if confusion[c, c] + confusion[c, n] > 0 \
            else 0.0
        rn = float(confusion[n, n]) / (confusion[n, c] + confusion[n, n]) \
            if confusion[n, c] + confusion[n, n] > 0 \
            else 0.0
        r = float(rc + rn) / 2

        fc = AbstractEvaluator.f1_score(pc, rc)
        fn = AbstractEvaluator.f1_score(pn, rn)
        f = float(fc + fn) / 2

        return p, r, f

    def total_num_links(self, gold_clusters, auto_clusters):
        gold_ms = {m for gc in gold_clusters for m in gc}
        auto_ms = {m for ac in auto_clusters for m in ac}
        num_ms = len(gold_ms.union(auto_ms))
        num_links = (num_ms * (num_ms - 1)) / 2

        return num_links

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        def get_links(cluster):
            if len(cluster) > 1:
                # prevent duplicate pairs from being considered as different links
                # i.e. (m1, m2) = (m2, m1)
                links = {(m1, m2) if m1.id < m2.id else (m2, m1) for i, m1 in enumerate(cluster) for m2 in cluster[i + 1:]}
                return links
            else:
                return set()

        gold_links = set.union(*map(get_links, gold_clusters))
        auto_links = set.union(*map(get_links, auto_clusters))

        num_links = self.total_num_links(gold_clusters, auto_clusters)

        # coreferent / non-coreferent indices
        c, n = 0, 1
        confusion = np.zeros((2, 2), dtype="int32")

        confusion[c, c] = len(auto_links & gold_links)  # intersection of links
        confusion[n, c] = len(auto_links.difference(gold_links))    # (auto union gold) \ gold
        confusion[c, n] = len(gold_links.difference(auto_links))    # (auto union gold) \ auto
        confusion[n, n] = num_links - (confusion[c, c] + confusion[n, c] + confusion[c, n])

        return confusion


class CeafeEvaluator(AbstractEvaluator):
    def evaluate_documents(self, gold_documents, auto_documents):
        return self.evaluate_clusters(sum(gold_documents, []), sum(auto_documents, []))

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        def phi4(c1, c2):
            return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

        # auto_clusters = [c for c in auto_clusters if len(c) != 1]
        scores = np.zeros((len(gold_clusters), len(auto_clusters)))
        for i in range(len(gold_clusters)):
            for j in range(len(auto_clusters)):
                scores[i, j] = phi4(gold_clusters[i], auto_clusters[j])
        matching = linear_assignment(-scores)
        similarity = float(sum(scores[matching[:, 0], matching[:, 1]]))

        p = similarity / len(auto_clusters) if similarity else 0.0
        r = similarity / len(gold_clusters) if similarity else 0.0

        return p, r, self.f1_score(p, r)


class LinkingMicroF1Evaluator(object):
    def __init__(self, labels):
        self.labels = labels

    def evaluate_states(self, states):
        gold_links = {l: [] for l in self.labels}
        auto_links = {l: [] for l in self.labels}

        for m in sum(states, []):
            gold_links[m.gold_ref].append(m)

            auto_links[m.auto_ref].append(m)

        scores = {}
        for l in self.labels:
            g, a = gold_links[l], auto_links[l]
            c = float(len(set(g).intersection(set(a))))

            p = c / len(a) if a else 0.0
            r = c / len(g) if g else 0.0
            f = AbstractEvaluator.f1_score(p, r)

            scores[l] = (p, r, f)

        return scores


class LinkingMacroF1Evaluator(object):
    def __init__(self, labels):
        self.labels = labels

    def evaluate_states(self, states):
        c = 0.0
        m_all = sum(states, [])
        for m in m_all:
            if m.gold_ref == m.auto_ref:
                c += 1.0

        p = r = f = float(c) / len(m_all) if m_all and len(m_all) > 0 else 0.0
        return p, r, f
