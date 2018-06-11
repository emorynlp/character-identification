class MentionCluster(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def __hash__(self):
        return hash(self[0] if self else -1)

    def append(self, mention):
        super(MentionCluster, self).append(mention)
