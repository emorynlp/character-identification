class SingOnlyMentionNode(object):
    @staticmethod
    def root():
        return SingOnlyMentionNode(0, [], 'root', {})

    def __init__(self, id, tokens, gold_ref, auto_ref=None, feat_map=None):
        self.id = id
        self.tokens = tokens

        self.gold_ref = str(gold_ref).lower()
        self.auto_ref = str(auto_ref).lower() if auto_ref else ''
        self.feat_map = feat_map if feat_map is not None else dict()

    def isRoot(self):
        return self.id == 0 or self.gold_ref == 'root'

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __str__(self):
        return " ".join(map(str, self.tokens))

    def __repr__(self):
        return self.__str__()
