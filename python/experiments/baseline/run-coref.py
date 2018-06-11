import pickle
import fasttext

from experiments.baseline.model.coref import MentionMentionCNN
from experiments.baseline.tools.evaluators import *
from experiments.baseline.tools.ioutils import SpliceReader
from experiments.baseline.tools.state import SingEvalCorefState

from component.features import *
from constants.params import CorefParams
from constants.paths import *
from util.readers import *


data_in = Paths.Transcripts.get_input_transcript_paths()

nb_fltrs = CorefParams.nb_fltrs
nb_epoch = CorefParams.nb_epoch
batch_size = CorefParams.batch_size
gpu = CorefParams.gpu
eval_only = CorefParams.eval_only

experiment_name = "baseline"
model_num = 1
gpu_num = 3

model_out = Paths.CorefModels.get_model_export_path(experiment_name, model_num)
ftmap_out = Paths.CorefModels.get_feat_map_export_path(experiment_name, model_num)


def main():
    timer = Timer()

    # Loading transcripts
    timer.start('load_transcript')
    Strn, Sdev, Stst, reader = [], [], [], SpliceReader()
    spks, poss, deps, ners = set(), set(), set(), set()

    for d_in in data_in:
        es, ms = reader.read_season_json(d_in[0])

        spks.update(TranscriptUtils.collect_speakers(es))
        poss.update(TranscriptUtils.collect_pos_tags(es))
        ners.update(TranscriptUtils.collect_ner_tags(es))
        deps.update(TranscriptUtils.collect_dep_labels(es))

        keys, d_trn, d_dev, d_tst = set(), dict(), dict(), dict()
        for m in ms:
            eid = m.tokens[0].parent_episode().id
            sid = m.tokens[0].parent_scene().id

            target = d_trn if eid in d_in[1] \
                else d_dev if eid in d_in[2] \
                else d_tst

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

    # Loading word2vec
    timer.start('load_w2v')
    w2v = fasttext.load_model(Paths.Resources.Fasttext50d)
    print("Fasttext data loaded - %.2fs" % timer.end('load_w2v'))

    w2g = ani = ina = None
    if ftmap_out and not eval_only:
        # Loading gender data
        timer.start('load_w2g')
        w2g_in = open(Paths.Resources.GenderData, 'rb')
        w2g = GenderDataReader.load(w2g_in, True, True)
        print("Gender data loaded   - %.2fs" % timer.end('load_w2g'))

        # Loading animacy data
        timer.start('load_animacy_dicts')
        ani_in = open(Paths.Resources.AnimateUnigram, 'rb')
        ani = DictionaryReader.load_string_set(ani_in)
        ina_in = open(Paths.Resources.InanimateUnigram, 'rb')
        ina = DictionaryReader.load_string_set(ina_in)
        print("Animacy data loaded  - %.2fs" % timer.end('load_animacy_dicts'))

    # Extracting features
    if ftmap_out and eval_only:
        timer.start('load_feature_map')
        with open(ftmap_out, 'rb') as fin:
            mft_map = pickle.load(fin)
        mft_map.w2v = w2v
        print('Feature map loaded   - %.2fs' % timer.end('load_feature_map'))
    else:
        mft_map = MentionFeatureExtractor(w2v, w2g, spks, poss, ners, deps, ani, ina)

    timer.start('feature_extraction')
    for s in sum([Strn, Sdev, Stst], []):
        s.pfts = {m: dict() for m in s}
        for i, m in enumerate(s):
            m.id, (efts, mft) = i, mft_map.extract_mention(m)
            m.feat_map['efts'], m.feat_map['mft'] = efts, mft

            for a in s[:i]:
                s.pfts[a][m] = mft_map.extract_pairwise(a, m)
    print("Feature extracted    - %.2fs\n" % timer.end('feature_extraction'))

    if ftmap_out and not eval_only:
        timer.start('dump_feature_extractor')
        with open(ftmap_out, 'wb') as fout:
            pickle.dump(mft_map, fout, protocol=2)
        print("Feature extractor saved to %s - %.2fs" % (ftmap_out, timer.end('dump_feature_extractor')))

    # Collection feature shape information
    m1, m2 = Strn[0][1], Strn[0][2]
    efts, mft = m1.feat_map['efts'], m1.feat_map['mft']
    eftdims = list(map(lambda x: x.shape, efts))
    mftdim, pftdim = len(mft), len(Strn[0].pfts[m1][m2])

    model = MentionMentionCNN(eftdims, mftdim, pftdim, nb_fltrs, gpu_num, gpu=gpu)
    if not eval_only:
        # Model training
        model.train_ranking(Strn, Sdev, nb_epoch=nb_epoch, batch_size=batch_size, model_out=model_out)
    else:
        model.load_model_weights(model_out)

    # Model evaluation
    print('\nEvaluating trained model on Tst')
    model.decode_clusters([s.reset() for s in Stst])
    golds, autos, = [s.gCs for s in Stst], [s.aCs for s in Stst]

    p, r, f = BCubeEvaluator().evaluate_documents(golds, autos)
    print('Bcube - %.4f/%.4f/%.4f' % (p, r, f))

    p, r, f = CeafeEvaluator().evaluate_documents(golds, autos)
    print('Ceafe - %.4f/%.4f/%.4f' % (p, r, f))

    p, r, f = BlancEvaluator().evaluate_documents(golds, autos)
    print('Blanc - %.4f/%.4f/%.4f' % (p, r, f))


if __name__ == "__main__":
    main()
