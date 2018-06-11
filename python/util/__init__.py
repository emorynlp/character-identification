import numpy as np
from time import time


class Timer:
    @staticmethod
    def now():
        return time()

    def __init__(self):
        self.timers = dict()

    def start(self, *args):
        start_time = time()
        for arg in args:
            if arg not in self.timers:
                self.timers[arg] = start_time
        return [start_time] * len(args)

    def end(self, *args):
        end_time, times = time(), []
        for arg in args:
            if arg in self.timers:
                times.append(end_time - self.timers[arg])
                del self.timers[arg]
            else:
                times.append(0.0)
        return times if len(times) > 1 else times[0]


class TranscriptUtils(object):

    @staticmethod
    def collect_speakers(episodes):
        return set([spk for e in episodes for s in e.scenes for u in s.utterances for spk in u.speakers])

    @staticmethod
    def collect_from_nodes(episodes, f):
        us = [u for e in episodes for s in e.scenes for u in s.utterances]
        return set([f(n) for u in us for ns in u.statements for n in ns])

    @staticmethod
    def collect_pos_tags(episodes):
        return TranscriptUtils.collect_from_nodes(episodes, lambda n: n.pos_tag)

    @staticmethod
    def collect_ner_tags(episodes):
        return TranscriptUtils.collect_from_nodes(episodes, lambda n: n.ner_tag)

    @staticmethod
    def collect_dep_labels(episodes):
        return TranscriptUtils.collect_from_nodes(episodes, lambda n: n.dep_label)


class StringUtils(object):

    @staticmethod
    def lcs(a, b):
        lengths = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        result = ""
        x, y = len(a), len(b)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            else:
                assert a[x - 1] == b[y - 1]
                result = a[x - 1] + result
                x -= 1
                y -= 1
        return result


class DSUtils(object):

    @staticmethod
    def create_lists(num, rows=1):
        ls = [[[] for _ in range(num)] for _ in range(rows)]
        return ls if rows > 1 else ls[0]

    @staticmethod
    def convert_to_batch(X, Y):
        Xb = [np.concatenate(col, axis=0) for col in X]
        Yb = [np.concatenate(col, axis=0) for col in Y]

        count, starts, ends = 0, [], []
        for size in [len(y) for y in Y[0]]:
            starts += [count] * size
            ends += [count + size] * size
            count += size

        starts = np.expand_dims(np.array(starts).astype('int32'), axis=1)
        ends = np.expand_dims(np.array(ends).astype('int32'), axis=1)
        return Xb, Yb + [starts, ends]

    @staticmethod
    def create_batches(X, Y, bsize, nb_epoch, shuffle=True):
        dsize, bpe = len(Y[0]), len(Y[0]) / bsize + 1

        indices = np.arange(dsize)
        for e in range(nb_epoch):
            if shuffle:
                np.random.shuffle(indices)

            for bidx in range(bpe):
                sidx, eidx = bidx * bsize, min((bidx + 1) * bsize, dsize)
                selected = set(indices[sidx:eidx])

                Xi = [[x for i, x in enumerate(col) if i in selected] for col in X]
                Yi = [[y for i, y in enumerate(col) if i in selected] for col in Y]

                Xb = [np.concatenate(col, axis=0) for col in Xi]
                Yb = [np.concatenate(col, axis=0) for col in Yi]

                count, starts, ends = 0, [], []
                for size in [len(y) for y in Yi[0]]:
                    starts += [count] * size
                    ends += [count + size] * size
                    count += size

                starts = np.expand_dims(np.array(starts).astype('int32'), axis=1)
                ends = np.expand_dims(np.array(ends).astype('int32'), axis=1)

                yield Xb, Yb + [starts, ends]


class DebugUtils(object):

    @staticmethod
    def shape_strings(iterable):
        return str(map(lambda e: np.array(e).shape, iterable))
