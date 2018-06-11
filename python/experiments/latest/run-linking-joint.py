import pickle

from experiments.latest.model.linking_joint import JointMentionClusterEntityLinker
from experiments.latest.tools.evaluators import LinkingMicroF1Evaluator, LinkingMacroF1Evaluator
from experiments.latest.tools.ioutils import StateWriter

from util import *
from constants.params import LinkingParams


states_in = 'data/latest.f1-4.states.p'

nb_fltrs = LinkingParams.nb_fltrs
nb_epoch = LinkingParams.nb_epoch
batch_size = LinkingParams.batch_size
gpu = LinkingParams.gpu
eval_only = LinkingParams.eval_only

model_num = 3

model_out = ''

OTHER = '#other#'
GENERAL = '#general#'

labels = ['monica geller', 'judy geller', 'jack geller', 'lily buffay', 'rachel green', 'joey tribbiani',
          'phoebe buffay', 'carol willick', 'ross geller', 'chandler bing', 'gunther',
          'ben geller', 'barry farber', 'richard burke', 'kate miller', 'peter becker',
          'emily waltham'] + [OTHER, GENERAL]


def main():
    timer = Timer()

    timer.start('load_states')
    with open(states_in, 'rb') as fin:
        Strn, Sdev, Stst = pickle.load(fin)
    Sall = Strn + Sdev + Stst
    print('States loaded - %.2fs' % timer.end('load_states'))

    # Referent label reassignment
    for m in sum(Sall, []):
        m.gold_refs = [OTHER if gref.lower() not in labels else gref.lower() for gref in m.gold_refs]

    m1, m2 = Strn[0][0], Strn[0][1]
    mrepr_dim = len(m1.feat_map['mrepr'])
    mpair_dim = len(Strn[0].mpairs[m1][m2])

    model = JointMentionClusterEntityLinker(nb_fltrs, mrepr_dim, mpair_dim, labels, gpu=gpu)

    if not eval_only:
        model.train_linking(Strn, Sdev, nb_epoch=nb_epoch, batch_size=batch_size, model_out=model_out)
    else:
        model.load_model_weights(model_out + ".sing", model_out + ".pl")

    print('\nEvaluating trained model')
    scorer = LinkingMicroF1Evaluator(labels)
    model.do_linking(Stst)
    scores = scorer.evaluate_states(Stst)
    avg = np.mean(list(scores.values()), axis=0)

    sacc, pacc = model.accuracy(Stst)
    print('Test accuracy: %.4f/%.4f\n' % (sacc, pacc))
    for l, s in scores.items():
        print("%10s : %.4f %.4f %.4f" % (l, s[0], s[1], s[2]))
    print('\n%10s : %.4f %.4f %.4f' % ('avg', avg[0], avg[1], avg[2]))

    macro_scorer = LinkingMacroF1Evaluator()
    p, r, f = macro_scorer.evaluate_states(Stst)
    print("\n%10s : %.4f %.4f %.4f" % ("macro", p, r, f))

    results_path = "./jel-noc.results.f1-4.%d.txt" % model_num
    writer = StateWriter()
    writer.open_file(results_path)
    writer.write_states(Stst)


if __name__ == "__main__":
    main()
