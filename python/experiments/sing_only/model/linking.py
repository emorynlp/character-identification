import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D

from util import *


class MentionClusterEntityLinker(object):
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
            probs = Dense(nb_labels, activation='softmax')(hidden2)

        self.linking_model = Model(inputs=[mrepr, crepr, cmmft], outputs=[probs])
        self.linking_model.compile(optimizer=RMSprop(),
                                   loss=['sparse_categorical_crossentropy'],
                                   metrics=['sparse_categorical_accuracy'])

    def predict(self, instances):
        return self.linking_model.predict(instances)

    def accuracy(self, states):
        X, Y = self.construct_batch(states)
        return self.linking_model.test_on_batch(X, Y)[1]

    def load_model_weights(self, path):
        self.linking_model.load_weights(path)

    def save_model_weights(self, path):
        self.linking_model.save_weights(path)

    def do_linking(self, states):
        ms, mreprs, creprs, cmmfts = [], [], [], []
        for s in states:
            m2aC, m_mpairs = s.m2_aC, s.mpairs

            for m in s:
                crepr, cmmft = self.get_creprs(m2aC[m], m, m_mpairs)
                mreprs.append(m.feat_map['mrepr'])
                creprs.append(crepr)
                cmmfts.append(cmmft)
                ms.append(m)

        mreprs, creprs, cmmfts = map(np.array, [mreprs, creprs, cmmfts])
        preds = np.argmax(self.predict([mreprs, creprs, cmmfts]), axis=1)
        for m, p in zip(ms, preds):
            m.auto_ref = self.idx2label[p]

    def train_linking(self, Strn, Sdev, nb_epoch=20, batch_size=32, model_out=None):
        (Xtrn, Ytrn), (Xdev, Ydev) = self.construct_batch(Strn), self.construct_batch(Sdev)
        m, Sall, timer = self.linking_model, Strn + Sdev, Timer()
        best_trn, best_dev, best_epoch, best_weights, = 0, 0, 0, None

        timer.start('all_training')
        for e in range(nb_epoch):
            timer.start('epoch_training')
            h = m.fit(Xtrn, Ytrn, validation_data=(Xdev, Ydev), batch_size=batch_size, epochs=1, verbose=0)
            tl, ta = h.history['loss'][0], h.history['sparse_categorical_accuracy'][0]
            dl, da = h.history['val_loss'][0], h.history['val_sparse_categorical_accuracy'][0]
            ttrn = timer.end('epoch_training')

            if da > best_dev:
                best_weights = self.linking_model.get_weights()
                best_epoch, best_trn, best_dev = e, ta, da
                if model_out:
                    self.save_model_weights(model_out)

            self.logger.info('Epoch %3d - trn: %4.2fs, Trn - ls %.4f ac %.4f, Dev - ls %.4f ac %.4f'
                             % (e + 1, ttrn, tl, ta, dl, da))
        tall = timer.end('all_training')

        m.set_weights(best_weights)
        self.logger.info('Summary - Trn: %.4f, Dev: %.4f @ Epoch %d - %.2fs' % (best_trn, best_dev, best_epoch + 1, tall))

    def get_creprs(self, c, m, m_mpairs):
        cmreprs = [cm.feat_map['mrepr'] for cm in c]
        cmr_mx, cmr_av = np.amax(cmreprs, axis=0), np.mean(cmreprs, axis=0)

        cms = c[:c.index(m)]
        cmpairs = [m_mpairs[cm][m] for cm in cms] if cms else [np.zeros(self.mpair_dim)]
        cmp_mx, cmp_av = np.amax(cmpairs, axis=0), np.mean(cmpairs, axis=0)

        return [cmr_mx, cmr_av], [cmp_mx, cmp_av]

    def construct_batch(self, states):
        mreprs, creprs, cmmfts, labels = [], [], [], []
        for s in states:
            m2aC, m_mpairs = s.m2_aC, s.mpairs

            for m in s:
                crepr, cmmft = self.get_creprs(m2aC[m], m, m_mpairs)
                mreprs.append(m.feat_map['mrepr'])
                creprs.append(crepr)
                cmmfts.append(cmmft)
                labels.append(self.label2idx[m.gold_ref])

        mreprs, creprs, cmmfts = map(np.array, [mreprs, creprs, cmmfts])
        labels = np.array(labels).astype('int32')
        return [mreprs, creprs, cmmfts], [labels[:, np.newaxis]]
