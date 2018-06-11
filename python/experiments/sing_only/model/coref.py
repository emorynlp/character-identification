import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
from keras.layers import Input, Reshape, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

from experiments.sing_only.tools.evaluators import BCubeEvaluator
from util import *


class MentionMentionCNN(object):
    def __init__(self, eftdims, mftdim, pftdim, nb_fltrs, logger, gpu=None):
        self.nb_filters, self.nb_efts = nb_fltrs, len(eftdims)
        self.eftdims, self.mftdim, self.pftdim = eftdims, mftdim, pftdim
        self.ranking_model, self.mrepr_model, self.mpair_model = None, None, None

        self.logger = logger

        gpu_opts = tf.GPUOptions(allow_growth=True, visible_device_list=','.join(map(str, gpu) if gpu else []))
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opts))
        K.set_session(sess)

        with tf.device('/gpu:0'):
            mm_pft = Input(shape=(pftdim,))
            m1_mft, m2_mft = Input(shape=(mftdim,)), Input(shape=(mftdim,))

            m1_efts = [Input(shape=(r, d)) for r, d in eftdims]
            m2_efts = [Input(shape=(r, d)) for r, d in eftdims]

            with tf.name_scope('mrepr'):
                expand_dims, reshape_df, dropout = [], Reshape((nb_fltrs,)), Dropout(0.8)
                eft_c1rs, eft_c2rs, eft_c3rs, eft_p1rs, eft_p2rs, eft_p3rs = [], [], [], [], [], []
                for r, d in eftdims:
                    expand_dims.append(Reshape((r, d, -1)))
                    eft_c1rs.append(Conv2D(nb_fltrs, (1, d), activation='tanh'))
                    eft_c2rs.append(Conv2D(nb_fltrs, (2, d), activation='tanh'))
                    eft_c3rs.append(Conv2D(nb_fltrs, (3, d), activation='tanh'))
                    eft_p1rs.append(MaxPooling2D(pool_size=(r - 0, 1)))
                    eft_p2rs.append(MaxPooling2D(pool_size=(r - 1, 1)))
                    eft_p3rs.append(MaxPooling2D(pool_size=(r - 2, 1)))

                nb_rows, mrepr_conv = 3 * self.nb_efts, Conv2D(nb_fltrs, (1, nb_fltrs), activation='tanh')
                reshape_cefts, mrepr_pool = Reshape((nb_rows, nb_fltrs, -1)), MaxPooling2D(pool_size=(nb_rows, 1))

                def mrepr(m_efts, m_mft, name=None):
                    eft_v1rs, eft_v2rs, eft_v3rs = [], [], []
                    for expand_dim, eft, c1r, c2r, c3r, p1r, p2r, p3r \
                            in zip(expand_dims, m_efts, eft_c1rs, eft_c2rs, eft_c3rs, eft_p1rs, eft_p2rs, eft_p3rs):
                        eft_e = expand_dim(eft)
                        eft_v1rs.append(dropout(reshape_df(p1r(c1r(eft_e)))))
                        eft_v2rs.append(dropout(reshape_df(p2r(c2r(eft_e)))))
                        eft_v3rs.append(dropout(reshape_df(p3r(c3r(eft_e)))))

                    eft_mat = reshape_cefts(concatenate(eft_v1rs + eft_v2rs + eft_v3rs))
                    eft_vec = reshape_df(mrepr_pool(mrepr_conv(eft_mat)))

                    return concatenate([eft_vec, m_mft], name=name)

                m1_mrepr = mrepr(m1_efts, m1_mft, name='m1_mrpr')
                m2_mrepr = mrepr(m2_efts, m2_mft, name='m2_mrpr')

            with tf.name_scope('mpair'):
                mpair_conv = Conv2D(nb_fltrs, (1, nb_fltrs + mftdim), activation='tanh')
                reshape_reprs, mpair_pool = Reshape((2, nb_fltrs + mftdim, -1)), MaxPooling2D(pool_size=(2, 1))

                def mpair(mrepr1, mrepr2, pwft, name=None):
                    mat = reshape_reprs(concatenate([mrepr1, mrepr2]))
                    pft_vec = reshape_df(mpair_pool(mpair_conv(mat)))
                    return concatenate([pft_vec, pwft], name=name)

                mm_mpair = mpair(m1_mrepr, m2_mrepr, mm_pft, name='mm_mpair')

            with tf.name_scope('preds'):
                mm_hidden = dropout(Dense(nb_fltrs, activation='relu')(mm_mpair))
                mm_probs = Dense(1, activation='sigmoid', name='mm_probs', kernel_regularizer=l2(0.005))(mm_hidden)

        mrepr_inputs = m2_efts + [m2_mft]
        self.mrepr_model = Model(inputs=mrepr_inputs, outputs=[m2_mrepr], name='mm_mrepr_model')

        mpair_inputs = m1_efts + m2_efts + [m1_mft, m2_mft, mm_pft]
        self.mpair_model = Model(inputs=mpair_inputs, outputs=[mm_mpair], name='mm_mpair_model')

        ranking_inputs = m1_efts + m2_efts + [m1_mft, m2_mft, mm_pft]
        self.ranking_model = Model(inputs=ranking_inputs, outputs=[mm_probs], name='mm_ranking_model')
        self.ranking_model.compile(optimizer=RMSprop(), loss=['mse'], metrics=['binary_accuracy'])

    def get_mreprs(self, instances):
        return self.mrepr_model.predict(instances)

    def get_mpairs(self, instances):
        return self.mpair_model.predict(instances)

    def predict(self, instances):
        return self.ranking_model.predict(instances)

    def load_model_weights(self, path):
        self.ranking_model.load_weights(path)

    def save_model_weights(self, path):
        self.ranking_model.save_weights(path)

    def decode_clusters(self, states):
        ntdone = [s for s in states if not s.done()]

        while ntdone:
            m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts, pranges = [[] for _ in range(4)], [[] for _ in range(4)], [], [], [], []
            for s in ntdone:
                (antes, m), m2c = s.current(), s.m2_aC
                i, cefts, cmft = len(antes), m.feat_map['efts'], m.feat_map['mft']

                pranges.append((len(mp_pfts), len(mp_pfts) + i))
                for a in antes:
                    pefts, pmft, pft = a.feat_map['efts'], a.feat_map['mft'], s.pfts[a][m]
                    for l, i in zip(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], pefts + cefts + [pmft, cmft, pft]):
                        l.append(i)

            m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [np.array(g) for g in m1_efts], \
                                                          [np.array(g) for g in m2_efts], \
                                                          np.array(m1_mfts), \
                                                          np.array(m2_mfts), \
                                                          np.array(mp_pfts)

            preds = self.predict(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts])
            for s, (ps, pe) in zip(ntdone, pranges):
                nh = not np.any(np.round(preds[ps:pe, 0]))
                pmid = np.argmax(preds[ps:pe, 0]) if not nh else None
                s.link(pmid).advance()
            ntdone = [s for s in ntdone if not s.done()]

    def train_ranking(self, Strn, Sdev, nb_epoch=20, batch_size=32, model_out=None):
        (Xtrn, Ytrn), (Xdev, Ydev) = self.construct_batch(Strn), self.construct_batch(Sdev)

        Sall, m, timer, bcube = Strn + Sdev, self.ranking_model, Timer(), BCubeEvaluator()
        best_trn, best_dev, best_epoch, best_weights, = [0, 0, 0], [0, 0, 0], 0, None

        timer.start('all_training')
        for e in range(nb_epoch):
            timer.start('epoch_training')
            h = m.fit(Xtrn, Ytrn, validation_data=(Xdev, Ydev), batch_size=batch_size, epochs=1, verbose=0)
            tl, ta = h.history['loss'][0], h.history['binary_accuracy'][0]
            dl, da = h.history['val_loss'][0], h.history['val_binary_accuracy'][0]
            ttrn = timer.end('epoch_training')

            timer.start('epoch_decoding')
            self.decode_clusters([s.reset() for s in Sall])
            trn_b3 = bcube.evaluate_documents([s.gCs for s in Strn], [s.aCs for s in Strn])
            dev_b3 = bcube.evaluate_documents([s.gCs for s in Sdev], [s.aCs for s in Sdev])

            tclus = timer.end('epoch_decoding')

            if dev_b3[2] > best_dev[2]:
                best_weights = self.ranking_model.get_weights()
                best_epoch, best_trn, best_dev = e, trn_b3, dev_b3
                if model_out:
                    self.save_model_weights(model_out)

            self.logger.info('Epoch %3d - trn: %4.2fs, clstr: %4.2fs, Trn - %.4f/%.4f/%.4f, Dev - %.4f/%.4f/%.4f'
                             % (e+1, ttrn, tclus, trn_b3[0], trn_b3[1], trn_b3[2], dev_b3[0], dev_b3[1], dev_b3[2]))
            self.logger.info('\tTrn - ls: %.4f ac %.4f, Dev - ls: %.4f ac: %.4f' % (tl, ta, dl, da))
        tall = timer.end('all_training')

        m.set_weights(best_weights)
        self.logger.info('Summary - Trn: %.4f/%.4f/%.4f, Dev: %.4f/%.4f/%.4f @ Epoch %d - %.2fs'
                         % (best_trn[0], best_trn[1], best_trn[2], best_dev[0], best_dev[1], best_dev[2], best_epoch + 1, tall))

    def construct_batch(self, states):
        m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts, probs = [[] for _ in range(4)], [[] for _ in range(4)], [], [], [], []

        for s in [s.reset() for s in states]:
            ms, m2c = s, s.m2_gC
            for idx, m in enumerate(ms[1:], 1):
                cefts, cmft, cc = m.feat_map['efts'], m.feat_map['mft'], m2c[m]
                for a in reversed(ms[:idx]):
                    pefts, pmft, pft, ac = a.feat_map['efts'], a.feat_map['mft'], s.pfts[a][m], m2c[a]

                    for l, i in zip(m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], pefts + cefts + [pmft, cmft, pft]):
                        l.append(i)

                    probs.append(1.0 if ac == cc else 0.0)

        c, probs = len(m1_efts), np.array(probs)
        m1_efts, m2_efts, m1_mfts, m2_mfts, mp_pfts = [np.array(g) for g in m1_efts], \
                                                      [np.array(g) for g in m2_efts], \
                                                      np.array(m1_mfts), \
                                                      np.array(m2_mfts), \
                                                      np.array(mp_pfts)

        return m1_efts + m2_efts + [m1_mfts, m2_mfts, mp_pfts], [probs]
