from structure.cluster import MentionCluster


class SingEvalMentionCluster(MentionCluster):
    def __init__(self, *args):
        MentionCluster.__init__(self, *args)
