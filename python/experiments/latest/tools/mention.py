import numpy as np

from structure import TokenNode


class PluralMentionNode(object):
    def __init__(self, id, tokens, gold_refs, auto_refs=None, feat_map=None, plural=False):
        self.id = id
        self.tokens = tokens

        self.gold_refs = [gref.lower() for gref in gold_refs]
        self.auto_refs = [aref.lower() for aref in auto_refs] if auto_refs else []
        self.feat_map = feat_map if feat_map is not None else dict()

        self.plural = plural

    def __lt__(self, other_mention):
        return self.id < other_mention.id

    def __gt__(self, other_mention):
        return other_mention.__lt__(self)

    def __str__(self):
        return " ".join(map(str, self.tokens)) + " %d" % self.id

    def __repr__(self):
        return self.__str__()

    def is_other(self):
        return True if self.id == -1 and self.gold_refs == ["#other#"] else False

    def is_general(self):
        return True if self.id == -1 and self.gold_refs == ["#general#"] else False


other = PluralMentionNode(-1, [TokenNode(-1, "#other#")], ["#other#"], feat_map={})

general = PluralMentionNode(-1, [TokenNode(-1, "#general#")], ["#general#"], feat_map={})


def init_super_mentions(eftdims, mftdim, pftdim):
    other.feat_map["efts"] = [np.random.rand(d1, d2) for d1, d2 in eftdims]
    other.feat_map["mft"] = np.random.rand(mftdim)
    other.feat_map["pft"] = np.random.rand(pftdim)

    general.feat_map["efts"] = [np.random.rand(d1, d2) for d1, d2 in eftdims]
    general.feat_map["mft"] = np.random.rand(mftdim)
    general.feat_map["pft"] = np.random.rand(pftdim)
