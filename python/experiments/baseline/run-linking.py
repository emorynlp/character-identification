import pickle
import numpy as np

from experiments.baseline.tools.evaluators import LinkingMicroF1Evaluator, LinkingMacroF1Evaluator
from experiments.baseline.model.linking import MentionClusterEntityLinker

from constants.params import LinkingParams
from util import *


states_in = 'data/baseline.f1-4.states.p'

nb_fltrs = LinkingParams.nb_fltrs
nb_epoch = LinkingParams.nb_epoch
batch_size = LinkingParams.batch_size
gpu = LinkingParams.gpu
eval_only = LinkingParams.eval_only

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

    # print({m.gold_ref for m in sum(Sall, [])})
    # print("Num of mentions - {}".format(len([m for m in sum(Sall, [])])))
    # print({gref for m in sum(Sall, []) for gref in m.all_gold_refs})

    # Referent label reassignment
    for m in sum(Sall, []):
        if m.gold_ref not in labels:
            # m.gold_ref = "[u'{}']".format(OTHER)
            m.gold_ref = OTHER
        m.all_gold_refs = [OTHER if gref.lower() not in labels else gref.lower() for gref in m.all_gold_refs]

    # print({m.gold_ref for m in sum(Sall, [])})
    # print({gref for m in sum(Sall, []) for gref in m.all_gold_refs})
    # gold_links = {l: [] for l in labels}
    #
    # for m in sum(Sall, []):
    #     for gref in m.all_gold_refs:
    #         gold_links[gref].append(m)
    #
    # for l in labels:
    #     print(l)
    #     print(len(gold_links[l]))
    #     s = 0
    #     for m in gold_links[l]:
    #         if len(m.all_gold_refs) > 1:
    #             s += 1
    #     print(s)
    #     print(float(s) / len(gold_links[l]))
    #     print()

    m1, m2 = Strn[0][0], Strn[0][1]
    mrepr_dim = len(m1.feat_map['mrepr'])
    mpair_dim = len(Strn[0].mpairs[m1][m2])

    model = MentionClusterEntityLinker(nb_fltrs, mrepr_dim, mpair_dim, labels, gpu=gpu)

    if not eval_only:
        model.train_linking(Strn, Sdev, nb_epoch=nb_epoch, batch_size=batch_size, model_out=model_out)
    else:
        model.load_model_weights(model_out)

    print('\nEvaluating trained model')
    micro_scorer = LinkingMicroF1Evaluator(labels)
    model.do_linking(Stst)
    scores = micro_scorer.evaluate_states(Stst)
    avg = np.mean(list(scores.values()), axis=0)

    print('Test accuracy: %.4f\n' % model.accuracy(Stst))
    for l, s in scores.items():
        print("%10s : %.4f %.4f %.4f" % (l, s[0], s[1], s[2]))
    print('\n%10s : %.4f %.4f %.4f' % ('avg', avg[0], avg[1], avg[2]))

    macro_scorer = LinkingMacroF1Evaluator(labels)
    p, r, f = macro_scorer.evaluate_states(Stst)
    print("\n%10s : %.4f %.4f %.4f" % ("macro", p, r, f))


if __name__ == "__main__":
    main()
