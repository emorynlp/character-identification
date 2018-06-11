from structure.cluster import MentionCluster


class SingOnlyMentionCluster(MentionCluster):
    def __init__(self, *args):
        MentionCluster.__init__(self, *args)
