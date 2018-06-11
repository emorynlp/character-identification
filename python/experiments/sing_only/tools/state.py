from experiments.sing_only.tools.cluster import SingOnlyMentionCluster


class SingOnlyCorefState(list):
    def __init__(self, mentions, extract_gold=False, pfts=None, mpairs=None):
        list.__init__(self, mentions)

        self.cmid, self.pfts, self.mpairs = 0, pfts, mpairs
        self.gCs, self.m2_gC, self.aCs, self.m2_aC = None, None, [], {}

        self.ambiguous_labels = ["#other#", "#general#"]

        if extract_gold:
            labels = set([m.gold_ref for m in mentions])
            m_l2c = dict([(l, SingOnlyMentionCluster()) for l in labels])

            for m in mentions:
                if m.gold_ref not in self.ambiguous_labels:
                    m_l2c[m.gold_ref].append(m)

            self.m2_gC = {m: m_l2c[m.gold_ref]
                          if m.gold_ref not in self.ambiguous_labels
                          else SingOnlyMentionCluster([m])
                          for m in mentions}

            self.gCs = list(set(self.m2_gC.values()))
            # print([[m.gold_ref for m in gc] for gc in self.gCs])

        self.reset()

    def link(self, aid=None, mid=None):
        m = self[mid] if mid else self[self.cmid]
        a = self[aid] if aid and 0 <= aid < len(self) else None
        c = self.m2_aC.get(a, SingOnlyMentionCluster())

        self.m2_aC[m] = c
        if len(c) == 0:
            self.aCs.append(c)
        c.append(m)
        return self

    def advance(self):
        self.cmid += 1
        return self

    def current(self):
        return self[:self.cmid], self[self.cmid]

    def reset(self):
        c = SingOnlyMentionCluster([self[0]])
        self.cmid, self.aCs, self.m2_aC = 1, [c], {self[0]: c}
        return self

    def done(self):
        return self.cmid >= len(self)
