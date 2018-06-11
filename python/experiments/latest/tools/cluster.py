from structure.cluster import MentionCluster


class PluralCluster(MentionCluster):
    def __init__(self, *args):
        MentionCluster.__init__(self, *args)

        self._mset = set(self)
        self.repr = None
        self.pair_repr = None

    def __hash__(self):
        return hash(tuple(self))

    def append(self, mention):
        if mention not in self._mset:
            super(PluralCluster, self).append(mention)
            self._mset.add(mention)
