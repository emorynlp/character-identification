class Episode(object):
    def __init__(self, id, scenes=None, previous=None, next=None):
        self.id = int(id)
        self.scenes = scenes if scenes is not None else []

        self._previous = previous
        self._next = next

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def previous_episode(self):
        return self._previous

    def next_episode(self):
        return self._next


class Scene(object):
    def __init__(self, id, utterances=None, episode=None, previous=None, next=None):
        self.id = int(id)
        self.utterances = utterances if utterances is not None else []

        self._episode = episode

        self._previous = previous
        self._next = next

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def previous_scene(self):
        return self._previous

    def next_scene(self):
        return self._next

    def parent_episode(self):
        return self._episode


class Utterance(object):
    def __init__(self, speakers, utterances=None, statements=None, scene=None, previous=None, next=None):
        self.speakers = speakers
        self.utterances = utterances if utterances is not None else []
        self.statements = statements if statements is not None else []

        self._scene = scene

        self._previous = previous
        self._next = next

    def previous_utterance(self):
        return self._previous

    def next_utterance(self):
        return self._next

    def parent_scene(self):
        return self._scene
