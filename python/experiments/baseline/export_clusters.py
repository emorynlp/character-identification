import sys
import pickle
import fasttext

from experiments.baseline.model.coref import MentionMentionCNN
from experiments.baseline.tools.evaluators import BCubeEvaluator
from experiments.baseline.tools.ioutils import SpliceReader
from experiments.baseline.tools.state import SingEvalCorefState

from component.features import *
from constants.params import CorefParams
from constants.paths import Paths
from util.readers import *


data_in = Paths.Transcripts.get_input_transcript_paths()

nb_fltrs = CorefParams.nb_fltrs
gpu = CorefParams.gpu

gpu_num = 1

pt_model_in = 'pretrained_models/baseline.f1-4.pt.m'
pt_ftmp_in = 'pretrained_models/baseline.f1-4.pt.ft.p'

data_out = 'data/baseline.f1-4.states.p'


timer = Timer()

# Loading transcripts
timer.start('load_transcript')
Strn, Sdev, Stst, reader = [], [], [], SpliceReader()
spks, poss, deps, ners = set(), set(), set(), set()
for d_in in data_in:
    # json_in = open(d_in[0], 'rb')
    es, ms = reader.read_season_json(d_in[0])

    spks.update(TranscriptUtils.collect_speakers(es))
    poss.update(TranscriptUtils.collect_pos_tags(es))
    ners.update(TranscriptUtils.collect_ner_tags(es))
    deps.update(TranscriptUtils.collect_dep_labels(es))

    keys, d_trn, d_dev, d_tst = set(), dict(), dict(), dict()
    # ms = [m for m in ms if m.gold_ref is not 'collective']
    for m in ms:
        eid = m.tokens[0].parent_episode().id
        sid = m.tokens[0].parent_scene().id

        target = d_trn if eid in d_in[1] \
            else d_dev if eid in d_in[2] \
            else d_tst

        # key = eid * 100
        key = eid * 100 + sid
        if key not in target:
            target[key] = []
        target[key].append(m)
        keys.add(key)

    for key in sorted(keys):
        if key in d_trn:
            Strn.append(SingEvalCorefState(d_trn[key], extract_gold=True))
        if key in d_dev:
            Sdev.append(SingEvalCorefState(d_dev[key], extract_gold=True))
        if key in d_tst:
            Stst.append(SingEvalCorefState(d_tst[key], extract_gold=True))
    print("Transcript loaded: %s w/ %d mentions" % (d_in[0], len(ms)))

trnc, devc, tstc = sum(map(len, Strn)), sum(map(len, Sdev)), sum(map(len, Stst))
print("%d transcript(s) loaded with %d speakers and %d mentions (Trn/Dev/Tst: %d(%d)/%d(%d)/%d(%d)) - %.2fs\n" \
      % (len(data_in), len(spks), trnc + devc + tstc, len(Strn), trnc, len(Sdev), devc, len(Stst), tstc,
         timer.end('load_transcript')))
Sall = sum([Strn, Sdev, Stst], [])

timer.start('load_w2v')
w2v = fasttext.load_model(Paths.Resources.Fasttext50d)
print("Fasttext data loaded - %.2fs" % timer.end('load_w2v'))

timer.start('load_feature_map')
with open(pt_ftmp_in, 'rb') as fin:
    mft_map = pickle.load(fin)
mft_map.w2v = w2v
print('Feature map loaded   - %.2fs' % timer.end('load_feature_map'))

# Extracting features
timer.start('feature_extraction')
for s in sum([Strn, Sdev, Stst], []):
    s.pfts = {m: dict() for m in s}
    for i, m in enumerate(s):
        m.id, (efts, mft) = i, mft_map.extract_mention(m)
        m.feat_map['efts'], m.feat_map['mft'] = efts, mft

        for a in s[:i]:
            s.pfts[a][m] = mft_map.extract_pairwise(a, m)

# Collection feature shape information
m1, m2 = Strn[0][1], Strn[0][2]
efts, mft = m1.feat_map['efts'], m1.feat_map['mft']
eftdims = list(map(lambda x: x.shape, efts))
mftdim, pftdim = len(mft), len(Strn[0].pfts[m1][m2])

# Initialize model
model = MentionMentionCNN(eftdims, mftdim, pftdim, nb_fltrs, gpu_num, gpu=gpu)
model.load_model_weights(pt_model_in)

# Extract representations
ms = sum(Sall, [])
m_efts = np.array([m.feat_map['efts'] for m in ms])
print(m_efts.shape)
m_mfts = np.array([m.feat_map['mft'] for m in ms])
m_efts = [np.stack(m_efts[:, g]) for g in range(len(m_efts[0]))]
print(m_efts[0].shape, m_efts[1].shape, m_efts[2].shape, m_efts[3].shape)
for m, r in zip(ms, model.get_mreprs(m_efts + [m_mfts])):
    m.feat_map['mrepr'] = r

c = len(eftdims)
for s in Sall:
    pairs, s.mpairs = [], {m: dict() for m in s}
    # m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [], [], [], [], []
    m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [[] for _ in range(4)], [[] for _ in range(4)], [], [], []

    if len(s) > 1:
        for i, cm in enumerate(s[1:], 1):
            cefts, cmft = cm.feat_map['efts'], cm.feat_map['mft']
            for am in s[:i]:
                pefts, pmft, pft = am.feat_map['efts'], am.feat_map['mft'], s.pfts[am][cm]
                # map(lambda l, e: l.append(e),
                #     [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts], [pefts, cefts, pmft, cmft, pft])
                for l, e in zip(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], pefts + cefts + [pmft, cmft, pft]):
                    l.append(e)

                pairs.append((am, cm))

        # m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = map(np.array, [m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts])
        # m1_efts, m2_efts = [np.stack(m1_efts[:, g]) for g in range(c)], [np.stack(m2_efts[:, g]) for g in range(c)]

        m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [np.array(g) for g in m1_efts], \
                                                      [np.array(g) for g in m2_efts], \
                                                      np.array(m1_mfts), \
                                                      np.array(m2_mfts), \
                                                      np.array(mp_pfts)

        mpairs = model.get_mpairs(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts])
        for mp, (am, cm) in zip(mpairs, pairs):
            s.mpairs[am][cm] = mp

print("Feature extracted    - %.2fs\n" % timer.end('feature_extraction'))

# Model evaluation
print('\nEvaluating trained model')
model.decode_clusters([s.reset() for s in Sall])

p, r, f = BCubeEvaluator().evaluate_documents([s.gCs for s in Strn], [s.aCs for s in Strn])
print('Trn - %.4f/%.4f/%.4f' % (p, r, f))

p, r, f = BCubeEvaluator().evaluate_documents([s.gCs for s in Sdev], [s.aCs for s in Sdev])
print('Dev - %.4f/%.4f/%.4f' % (p, r, f))

p, r, f = BCubeEvaluator().evaluate_documents([s.gCs for s in Stst], [s.aCs for s in Stst])
print('Tst - %.4f/%.4f/%.4f' % (p, r, f))

default_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(default_recursion_limit * 2)

with open(data_out, 'wb') as fout:
    pickle.dump([Strn, Sdev, Stst], fout, protocol=2)

sys.setrecursionlimit(default_recursion_limit)
