import os
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Dense, Flatten
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


K.tensorflow_backend.set_session(get_session())


class LearningRate(object):
    # e.g. (6, 2) -> 6e-2 -> 0.06
    def __init__(self, base_ex):
        self.base, self.ex = base_ex

    def __str__(self):
        return str(self.base) + str(self.ex)

    def __call__(self):
        return self.base * 10 ** (-self.ex)


class KerasTransfer(object):
    def __init__(self,
                 model=None,
                 dataset='coco',
                 subset='2017',
                 pp_method='zmuv',
                 base_pooling='avg',
                 transform_flags='',
                 img_width=224,
                 img_height=224,
                 batch_norm=False,  # use batch normalization instead of dropout.
                 dropout=5,
                 hidden_size_1=1024,
                 hidden_size_2=None,
                 learn_rate_pretrain=(1, 3),
                 batch_size_pretrain=32,
                 resume_pretraining=True,
                 learn_rate=(1, 3),
                 batch_size=32,
                 n_frozen=15,
                 epochs=500,
                 model_basefilename=None):

        self.base_model = None
        self.top_model = None

        self.model = model
        self.resume_pretraining = resume_pretraining

        if model_basefilename is not None:
            filename_fields = model_basefilename.split('_')
            dataset = filename_fields[0]
            subset = filename_fields[1]
            pp_method = filename_fields[2]
            base_pooling = filename_fields[3]

            assert filename_fields[4][1] == 'x'
            n_hidden, size_and_norm = filename_fields[4].split('x')
            hidden_size_1 = int(size_and_norm[:-2])
            hidden_size_2 = None if n_hidden == '1' else hidden_size_1
            batch_norm = size_and_norm[-2:] == 'bn'
            dropout = int(size_and_norm[-1]) if size_and_norm[-2] == 'd' else 0

            assert filename_fields[5].startswith('lr')
            assert len(filename_fields[5]) == 4
            learn_rate_pretrain = (int(filename_fields[5][2]), int(filename_fields[5][3]))

            assert filename_fields[6].startswith('b')
            batch_size_pretrain = int(filename_fields[6][1:])

            n_frozen = filename_fields[7] if filename_fields[7] == 'all' else int(filename_fields[7])

            assert filename_fields[8].startswith('lr')
            assert len(filename_fields[8]) == 4
            learn_rate = (int(filename_fields[8][2]), int(filename_fields[8][3]))

            assert filename_fields[9].startswith('b')
            batch_size = int(filename_fields[9][1:])

            transform_flags = filename_fields[10]

        if dataset == 'coco':
            from coco import CocoGenerator
            self.Generator = CocoGenerator
            self.n_classes = 80
        elif dataset == 'nuswide':
            from nuswide import NusWideGenerator, subset_n_classes
            self.Generator = NusWideGenerator
            self.n_classes = subset_n_classes[subset]

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, 3)

        self.dataset = dataset
        self.subset = subset
        self.pp_method = pp_method
        self.base_pooling = base_pooling

        self.batch_norm = batch_norm
        self.dropout = 0.1 * dropout
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.bsp = batch_size_pretrain
        self.n_frozen = n_frozen
        self.bs = batch_size
        self.transform_flags = transform_flags

        self.lrp = LearningRate(learn_rate_pretrain)
        self.lr = LearningRate(learn_rate)

        self.epochs = epochs

        self.dset_fstr = '_'.join([self.dataset, self.subset.lower()])
        self.base_fstr = '_'.join([self.dset_fstr, self.pp_method, self.base_pooling])

        normalizing_layer = 'bn' if self.batch_norm else 'd' + str(dropout)
        n_layers = '1x' if self.hidden_size_2 is None else '2x'
        self.fc_fstr = n_layers + str(self.hidden_size_1) + normalizing_layer

        if self.hidden_size_2 is not None:
            assert self.hidden_size_1 == self.hidden_size_2

        self.top_fstr = '_'.join([self.fc_fstr, 'lr' + str(self.lrp), 'b' + str(self.bsp)])
        self.final_fstr = '_'.join([str(self.n_frozen), 'lr' + str(self.lr), 'b' + str(self.bs), self.transform_flags])

        pretrain_dir = '../data/bottleneck_top_models/'
        train_dir = '../data/final_weights/'

        self.top_model_weights_file = pretrain_dir + self.base_fstr + '_' + self.top_fstr + '.hdf5'
        self.pretrain_log_file = self.top_model_weights_file.rsplit('.', maxsplit=1)[0] + '.csv'
        self.final_weights_file = train_dir + self.base_fstr + '_' + self.top_fstr + '_' + self.final_fstr + '.hdf5'
        self.train_log_file = self.final_weights_file.rsplit('.', maxsplit=1)[0] + '.csv'
        self.predictions_file = self.final_weights_file.rsplit('.', maxsplit=1)[0] + '.npy'
        # print(self.base_fstr + '_' + self.top_fstr + '_' + self.final_fstr)

        self.valid_augmenter = None
        self.train_augmenter = None
        self.create_augmenters()

        if model is None:
            self.create_base_and_top_models()

    def create_augmenters(self):
        kwargs = {}
        if self.pp_method == 'zmuv':
            kwargs['samplewise_center'] = True
            kwargs['samplewise_std_normalization'] = True
        else:
            kwargs['samplewise_center'] = False
            kwargs['samplewise_std_normalization'] = False

        self.valid_augmenter = ImageDataGenerator(**kwargs)

        for t in self.transform_flags:
            if t == 'h':
                kwargs['height_shift_range'] = 0.1
            elif t == 'w':
                kwargs['width_shift_range'] = 0.1
            elif t == 'r':
                kwargs['rotation_range'] = 10
            elif t == 's':
                kwargs['shear_range'] = 0.1
            elif t == 'z':
                kwargs['zoom_range'] = 0.1
            elif t == 'f':
                kwargs['horizontal_flip'] = True

        self.train_augmenter = ImageDataGenerator(**kwargs)

    def create_base_and_top_models(self):
        if self.base_pooling in ('avg', 'max'):
            base_model = vgg16.VGG16(
                input_shape=self.input_shape,
                weights='imagenet',
                include_top=False,
                pooling=self.base_pooling)

            # Why does Keras provide the option to use regularizers with individual layers?
            # I thought L1 and L2 regularization was something you added
            # to your loss function at the end of your network?

            top_model = Sequential()

            if self.batch_norm:
                top_model.add(Dense(self.hidden_size_1, input_shape=base_model.output_shape[1:], activation='relu'))
                top_model.add(BatchNormalization())
                top_model.add(Activation('relu'))
                if self.hidden_size_2 is not None:
                    top_model.add(Dense(self.hidden_size_2))
                    top_model.add(BatchNormalization())
                    top_model.add(Activation('relu'))
            else:
                top_model.add(Dense(self.hidden_size_1, input_shape=base_model.output_shape[1:], activation='relu'))
                top_model.add(Dropout(self.dropout))
                if self.hidden_size_2 is not None:
                    top_model.add(Dense(self.hidden_size_2, activation='relu'))
                    top_model.add(Dropout(self.dropout))

            top_model.add(Dense(self.n_classes, activation='sigmoid'))

        elif self.base_pooling in ('77',):
            base_model = vgg16.VGG16(
                input_shape=self.input_shape,
                weights='imagenet',
                include_top=False)

            top_model = Sequential()

            if self.batch_norm:
                top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
                top_model.add(Dense(self.hidden_size_1))
                top_model.add(BatchNormalization())
                top_model.add(Activation('relu'))
                if self.hidden_size_2 is not None:
                    top_model.add(Dense(self.hidden_size_2))
                    top_model.add(BatchNormalization())
                    top_model.add(Activation('relu'))
            else:
                top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
                top_model.add(Dense(self.hidden_size_1, activation='relu'))
                top_model.add(Dropout(self.dropout))
                if self.hidden_size_2 is not None:
                    top_model.add(Dense(self.hidden_size_2, activation='relu'))
                    top_model.add(Dropout(self.dropout))

            top_model.add(Dense(self.n_classes, activation='sigmoid'))
        else:
            raise ValueError

        self.base_model = base_model
        self.top_model = top_model

    def predict_bottleneck_features(self, split_name):

        features_dir = '../data/bottleneck_features/'
        features_file = features_dir + '_'.join([self.base_fstr, split_name]) + '.npy'
        labels_file = features_dir + '_'.join([self.dset_fstr, split_name, 'labels']) + '.npy'

        if os.path.exists(features_file):
            features = np.load(features_file)
            labels = np.load(labels_file)
        else:
            generator = self.Generator(
                self.valid_augmenter,
                self.subset,
                split_name,
                batch_size=32,
                shuffle=False,
                standardize_method=self.pp_method,
                store_labels=True
            )
            features = self.base_model.predict_generator(
                generator,
                steps=generator.steps,
                # workers=1, # make sure this is set to 1 or the features will be out of order with the labels.
                verbose=1
            )
            np.save(features_file, features)

            if os.path.exists(labels_file):
                labels = np.load(labels_file)
                assert np.all(generator.stored_labels == labels)
            else:
                labels = generator.stored_labels
                np.save(labels_file, labels)

        return features, labels

    def pretrain_top_model(self):

        initial_epoch = 0
        if self.resume_pretraining:
            if os.path.exists(self.pretrain_log_file):
                if os.path.exists(self.top_model_weights_file):
                    self.load_top_model()
                    with open(self.pretrain_log_file) as ifs:
                        lines = ifs.read().strip().split('\n')
                        initial_epoch = int(lines[-1].split(',')[0])+1

        x_valid, y_valid = self.predict_bottleneck_features('valid')
        x_train, y_train = self.predict_bottleneck_features('train')

        sgd = SGD(lr=self.lrp(),
                  decay=1e-6,
                  nesterov=True,
                  momentum=0.9)

        self.top_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        csvlogger = CSVLogger(self.pretrain_log_file, separator=',', append=True)
        checkpointer = ModelCheckpoint(self.top_model_weights_file, save_best_only=True, verbose=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=10, min_delta=5e-5, verbose=1)

        self.top_model.fit(
            x_train, y_train,
            batch_size=self.bsp,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            callbacks=[
                csvlogger,
                checkpointer,
                earlystopper],
            initial_epoch=initial_epoch,
            epochs=self.epochs)

    def create_final_model(self):
        self.model = Model(
            inputs=self.base_model.input,
            outputs=self.top_model(self.base_model.output))

        if self.n_frozen == 'all':
            for layer in self.model.layers:
                layer.trainable = False
        else:
            for layer in self.model.layers[:self.n_frozen]:
                layer.trainable = False

        # self.model.summary()

        sgd = SGD(lr=self.lr(),
                  decay=1e-6,
                  nesterov=True,
                  momentum=0.9)

        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    def train_final_model(self):
        train_generator = self.Generator(
            self.train_augmenter,
            self.subset,
            'train',
            batch_size=self.bs,
            shuffle=True,
            standardize_method=self.pp_method)

        valid_generator = self.Generator(
            self.valid_augmenter,
            self.subset,
            'valid',
            batch_size=self.bs,
            shuffle=False,
            standardize_method=self.pp_method)

        csvlogger = CSVLogger(self.train_log_file, separator=',', append=True)
        checkpointer = ModelCheckpoint(self.final_weights_file, save_best_only=True, verbose=1)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, min_lr=0.00001, verbose=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=10, min_delta=5e-5, verbose=1)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.steps,
            validation_data=valid_generator,
            validation_steps=valid_generator.steps,
            callbacks=[
                csvlogger,
                checkpointer,
                # reduce_lr,
                earlystopper
            ],
            epochs=self.epochs,
            max_queue_size=16,
            workers=16
        )

    def pretrain(self):
        self.pretrain_top_model()

    def load_top_model(self):
        self.top_model.load_weights(self.top_model_weights_file)

    def train(self):
        self.create_final_model()
        self.train_final_model()

    def load_final_model(self):
        self.create_final_model()
        self.model.load_weights(self.final_weights_file)

    def make_predictions(self, split_name, augment=False):
        self.load_final_model()

        augmenter = self.train_augmenter if augment else self.valid_augmenter
        generator = self.Generator(
            augmenter,
            self.subset,
            split_name,
            batch_size=self.bs,
            shuffle=False,
            standardize_method=self.pp_method)

        preds = self.model.predict_generator(
            generator,
            generator.steps,
            verbose=1)

        return preds

    def load_predictions(self, split_name):

        train_dir = '../data/final_weights/'
        preds_file = train_dir + '_'.join([self.base_fstr, self.top_fstr, self.final_fstr, split_name]) + '.npy'
        if os.path.exists(preds_file):
            preds = np.load(preds_file)
        else:
            preds = self.make_predictions(split_name)
            np.save(preds_file, preds)
        return preds

    def load_labels(self, split_name):
        features_dir = '../data/bottleneck_features/'
        labels_file = features_dir + '_'.join([self.dset_fstr, split_name, 'labels']) + '.npy'
        assert os.path.exists(labels_file)
        labels = np.load(labels_file)
        return labels

    def load_tensors(self, split_name):
        generator = self.Generator(
            self.valid_augmenter,
            self.subset,
            split_name,
            batch_size=self.bs,
            shuffle=False,
            standardize_method=self.pp_method)

        return generator.images

    def load(self, force_pretrain=False, force_train=False):

        if not os.path.exists(self.top_model_weights_file) or force_pretrain:
            self.pretrain_top_model()
        else:
            self.top_model.load_weights(self.top_model_weights_file)

        self.create_final_model()

        if not os.path.exists(self.final_weights_file) or force_train:
            self.train_final_model()
        else:
            self.model.load_weights(self.final_weights_file)


if __name__ == "__main__":

    kt = KerasTransfer(
        dataset='nuswide',
        subset='Object',
        base_pooling='avg',
        pp_method='zmuv',
        img_width=224,
        img_height=224,
        hidden_size_1=1024,
        hidden_size_2=1024,
        learn_rate_pretrain=(1, 2),
        batch_size_pretrain=16,
        learn_rate=(1, 3),
        batch_size=64,
        n_frozen=15,
        epochs=500)

    kt.pretrain()

    # kt.train()
