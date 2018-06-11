import tensorflow as tf
import keras.backend as K

from keras.layers import Input, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from util import *


class JointMentionClusterEntityLinker(object):
    def __init__(self, nb_fltrs, mrepr_dim, mpair_dim, labels, logger, gpu=None):
        self.nb_fltrs, self.mrepr_dim, self.mpair_dim = nb_fltrs, mrepr_dim, mpair_dim
        self.nb_labels = nb_labels = len(labels)
        self.label2idx = {l: i for i, l in zip(range(nb_labels), labels)}
        self.idx2label = {i: l for i, l in zip(range(nb_labels), labels)}

        self.logger = logger

        gpu_opts = tf.GPUOptions(allow_growth=True, visible_device_list=','.join(map(str, gpu) if gpu else []))
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opts))
        K.set_session(sess)

        with tf.device('/gpu:0'):
            mrepr = Input(shape=(mrepr_dim,))
            crepr = Input(shape=(2, mrepr_dim,))
            cmmft = Input(shape=(2, mpair_dim,))

            mrd_exp, mpd_exp = Reshape((2, mrepr_dim, -1)), Reshape((2, mpair_dim, -1))
            crepr_conv = Conv2D(nb_fltrs, (1, mrepr_dim), activation='tanh')
            cmmft_conv = Conv2D(nb_fltrs, (1, mpair_dim), activation='tanh')
            reshape_nbf, r2_pool = Reshape((nb_fltrs, )), MaxPooling2D(pool_size=(2, 1))

            crepr_vec = reshape_nbf(r2_pool(crepr_conv(mrd_exp(crepr))))
            cmmft_vec = reshape_nbf(r2_pool(cmmft_conv(mpd_exp(cmmft))))

            dense_dim, dropout = 2 * nb_fltrs + mrepr_dim, Dropout(0.8)
            cm_vec = concatenate([dropout(mrepr), dropout(crepr_vec), dropout(cmmft_vec)])

            hidden1 = Dense(dense_dim, activation='relu')(cm_vec)
            hidden2 = Dense(dense_dim, activation='relu')(hidden1)
            # probs = Dense(nb_labels, activation='sigmoid')(hidden2)
            sing_probs = Dense(nb_labels + 1, activation='softmax')(hidden2)
            pl_probs = Dense(nb_labels, activation='sigmoid')(hidden2)

        self.slinking_model = Model(inputs=[mrepr, crepr, cmmft], outputs=[sing_probs])
        self.slinking_model.compile(optimizer=RMSprop(),
                                    loss=['sparse_categorical_crossentropy'],
                                    metrics=["sparse_categorical_accuracy"])

        self.plinking_model = Model(inputs=[mrepr, crepr, cmmft], outputs=[pl_probs])
        self.plinking_model.compile(optimizer=RMSprop(), loss=['binary_crossentropy'], metrics=["accuracy"])

    def predict(self, instances):
        return self.slinking_model.predict(instances), self.plinking_model.predict(instances)

    def accuracy(self, states):
        X, Ys, Yp = self.construct_batch(states)
        return self.slinking_model.test_on_batch(X, Ys)[1], self.plinking_model.test_on_batch(X, Yp)[1]

    def load_model_weights(self, sing_path, pl_path):
        self.slinking_model.load_weights(sing_path)
        self.plinking_model.load_weights(pl_path)

    def save_model_weights(self, sing_path, pl_path):
        self.slinking_model.save_weights(sing_path)
        self.plinking_model.save_weights(pl_path)

    def do_linking(self, states):
        ms, mreprs, creprs, cmmfts = [], [], [], []
        for s in states:
            m2aCs, m_mpairs = s.m2_aCs, s.mpairs

            for m in s:
                crepr, cmmft = self.get_cembds(m2aCs[m], m, m_mpairs)
                mreprs.append(m.feat_map['mrepr'])
                creprs.append(crepr)
                cmmfts.append(cmmft)
                ms.append(m)

        mreprs, creprs, cmmfts = map(np.array, [mreprs, creprs, cmmfts])
        spreds, ppreds = self.predict([mreprs, creprs, cmmfts])
        spreds = np.argmax(spreds, axis=1)
        for m, sp, pps in zip(ms, spreds, ppreds):
            if sp == self.nb_labels:
                argps = np.where(pps > 0.5)[0]
                m.auto_refs = [self.idx2label[argp] for argp in argps]
            else:
                m.auto_refs = [self.idx2label[sp]]

    def train_linking(self, Strn, Sdev, nb_epoch=20, batch_size=32, model_out=None):
        (Xtrn, Ystrn, Yptrn), (Xdev, Ysdev, Ypdev) = self.construct_batch(Strn), self.construct_batch(Sdev)
        sm, pm, Sall, timer = self.slinking_model, self.plinking_model, Strn + Sdev, Timer()
        best_strn, best_sdev, best_sepoch, best_weights_sing = 0, 0, 0, None
        best_ptrn, best_pdev, best_pepoch, best_weights_pl = 0, 0, 0, None

        timer.start('all_training')
        for e in range(nb_epoch):
            timer.start('epoch_training')
            sh = sm.fit(Xtrn, Ystrn, validation_data=(Xdev, Ysdev), batch_size=batch_size, epochs=1, verbose=0)
            ph = pm.fit(Xtrn, Yptrn, validation_data=(Xdev, Ypdev), batch_size=batch_size, epochs=1, verbose=0)

            stl, sta = sh.history['loss'][0], sh.history['sparse_categorical_accuracy'][0]
            sdl, sda = sh.history['val_loss'][0], sh.history['val_sparse_categorical_accuracy'][0]
            ptl, pta = ph.history['loss'][0], ph.history['acc'][0]
            pdl, pda = ph.history['val_loss'][0], ph.history['val_acc'][0]
            ttrn = timer.end('epoch_training')

            if sda > best_sdev:
                best_weights_sing = self.slinking_model.get_weights()
                best_weights_pl = self.plinking_model.get_weights()
                best_sepoch, best_strn, best_sdev = e, sta, sda
                best_pepoch, best_ptrn, best_pdev = e, pta, pda
                if model_out:
                    self.save_model_weights(model_out + ".sing", model_out + ".pl")

            self.logger.info('Epoch %3d - trn: %4.2fs, Trn - ls %.4f/%.4f ac %.4f/%.4f, Dev - ls %.4f/%.4f ac %.4f/%.4f'
                  % (e + 1, ttrn, stl, ptl, sta, pta, sdl, pdl, sda, pda))
        tall = timer.end('all_training')

        sm.set_weights(best_weights_sing)
        pm.set_weights(best_weights_pl)
        self.logger.info('Summary - S - Trn: %.4f, Dev: %.4f @ Epoch %d - %.2fs' % (best_strn, best_sdev, best_sepoch + 1, tall))
        self.logger.info('P - Trn: %.4f, Dev: %.4f @ Epoch %d - %.2fs' % (best_ptrn, best_pdev, best_pepoch + 1, tall))

    def get_cembds(self, cs, m, m_mpairs):
        # creprs, cmmfts = [], []
        creprs, cmmfts, cluster_sizes = [], [], []

        for c in cs:
            crepr, cmmft = self.get_creprs(c, m, m_mpairs)
            creprs.append(crepr)
            cmmfts.append(cmmft)
            cluster_sizes.append(len(c))

        # cluster_sizes = np.array(cluster_sizes).astype("float32") / np.sum(cluster_sizes)

        crepr_mx, crepr_av = np.amax(creprs, axis=0), np.mean(creprs, axis=0)
        cmmft_mx, cmmft_av = np.amax(cmmfts, axis=0), np.mean(cmmfts, axis=0)
        # crepr_mx, crepr_av = np.amax(creprs, axis=0), np.average(creprs, axis=0, weights=cluster_sizes)
        # cmmft_mx, cmmft_av = np.amax(cmmfts, axis=0), np.average(cmmfts, axis=0, weights=cluster_sizes)

        return crepr_av, cmmft_av

    def get_creprs(self, c, m, m_mpairs):
        cmreprs = [cm.feat_map['mrepr'] for cm in c]
        cmr_mx, cmr_av = np.amax(cmreprs, axis=0), np.mean(cmreprs, axis=0)

        cms = c[:c.index(m)]
        cmpairs = [m_mpairs[cm][m] for cm in cms] if cms else [np.zeros(self.mpair_dim)]
        cmp_mx, cmp_av = np.amax(cmpairs, axis=0), np.mean(cmpairs, axis=0)

        return [cmr_mx, cmr_av], [cmp_mx, cmp_av]

    def construct_batch(self, states):
        mreprs, creprs, cmmfts, slabels, plabels = [], [], [], [], []
        for s in states:
            m2aCs, m_mpairs = s.m2_aCs, s.mpairs

            for m in s:
                crepr, cmmft = self.get_cembds(m2aCs[m], m, m_mpairs)
                mreprs.append(m.feat_map['mrepr'])
                creprs.append(crepr)
                cmmfts.append(cmmft)

                slabels.append(self.label2idx[m.gold_refs[0]] if len(m.gold_refs) == 1 else self.nb_labels)

                g_label = np.zeros(shape=(self.nb_labels,)).astype("int32")
                idc = np.array([self.label2idx[gref] for gref in m.gold_refs])
                g_label[idc] = 1
                plabels.append(g_label)

        mreprs, creprs, cmmfts = map(np.array, [mreprs, creprs, cmmfts])
        slabels = np.array(slabels).astype('int32')
        plabels = np.array(plabels).astype('int32')

        return [mreprs, creprs, cmmfts], [slabels[:, np.newaxis]], plabels
