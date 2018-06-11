import re
import numpy as np


# Male/Female/Neutral
class GenderDataReader(object):
    @staticmethod
    def load(fin, word_only=False, normalize=False):
        word_regex, d = re.compile(b'^[A-Za-z]+$'), dict()

        for line in fin.readlines():
            string, data = line.lower().split(b'\t')
            string = string.replace(b"!", b"").strip()

            if not word_only or word_regex.match(string) is not None:
                vector = list(map(lambda x: int(x), data.split()[:3]))
                vector = np.array(vector).astype('float32')
                d[string] = d.get(string, np.zeros(len(vector))) + vector

        if normalize:
            for s, v in d.items():
                tc = float(sum(v))
                d[s] = v / tc if tc != 0.0 else v
        return d


class DictionaryReader(object):
    @staticmethod
    def load_string_set(fin):
        return set([line.strip() for line in fin.readlines()])