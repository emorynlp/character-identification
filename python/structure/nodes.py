class TokenNode(object):
    def __init__(self, id, word_form, pos_tag='_', ner_tag='_', dep_label='_', dep_head=-1, episode=None, scene=None, utterance=None):
        self.id = int(id)
        self.word_form = str(word_form)

        self.pos_tag = str(pos_tag)
        self.ner_tag = str(ner_tag)

        self.dep_label = str(dep_label)
        self.dep_head = dep_head

        self._scene = scene
        self._episode = episode
        self._utterance = utterance

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __str__(self):
        return self.word_form

    def __repr__(self):
        return self.__str__()

    def parent_utterance(self):
        return self._utterance

    def parent_scene(self):
        return self._scene

    def parent_episode(self):
        return self._episode

    def tsv_string(self):
        return '\t'.join(map(str, [self.id, self.word_form, self.pos_tag, self.dep_label, self.dep_head, self.ner_tag]))
