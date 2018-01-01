import os
from time import time
import random
import threading
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16

EPS = np.finfo(float).eps
subset_n_classes = {'All': 81, 'Lite': 81, 'Object': 31, 'Scene': 33}
data_source_dir = "/media/Borg_LS/DATA"


class NUSWIDE(object):
    def __init__(self,
                 source_dir=data_source_dir,
                 subset_name='Lite',
                 split_name='train',
                 test_size=50,  # in percent
                 seed=42,
                 llb=1,
                 lub=10):

        """Initialization

        :param
        subset_name: {All, Lite, Object, Scene}
        """

        self.source_dir = os.path.join(source_dir, 'NUS-WIDE')
        self.subset_name = subset_name
        self.split_name = split_name
        self.test_size = test_size
        self.seed = seed
        self.label_lower_bound = llb  # do not include any data with number of labels less than label_lower_bound
        self.label_upper_bound = lub  # do not include any data with number of labels greater than label_upper_bound.

        self.labels = None
        self.image_filename_list = None

        self.image_dir = os.path.join(self.source_dir, "Flickr")
        self.n_classes = subset_n_classes[subset_name]
        assert (self.n_classes in (31, 33, 81))

        index_valid_file = '_'.join([subset_name.lower(), str(seed), str(100 - test_size), 'valid']) + '.npy'
        index_test_file = '_'.join([subset_name.lower(), str(seed), str(test_size), 'test']) + '.npy'
        self.index_files = {
            'valid': '../data/input/' + index_valid_file,
            'test': '../data/input/' + index_test_file
        }
        self.create_dataset()
        self.filter_bad_data()
        self.num_samples, self.num_classes = self.labels.shape
        assert self.n_classes == self.num_classes

    def create_image_filename_list(self, train_or_test):

        if self.subset_name is "All":
            # ./ImageList/TestImagelist.txt
            imglist_file = self.source_dir + "/ImageList/" + train_or_test + "Imagelist.txt"
        elif self.subset_name is "Lite":
            # ./Lite/imagelist/Test_imageOutPutFileList.txt
            imglist_file = self.source_dir + "/Lite/imagelist/" + train_or_test + "_imageOutPutFileList.txt"
        elif self.subset_name is "Object":
            # ./OBJECT/imagelist/TestObject_image_name.txt
            imglist_file = self.source_dir + "/OBJECT/imagelist/" + train_or_test + "Object_image_name.txt"
        elif self.subset_name is "Scene":
            # ./SCENE/imagelist/Test_imageOutPutFileList.txt
            imglist_file = self.source_dir + "/SCENE/imagelist/" + train_or_test + "_imageOutPutFileList.txt"
        else:
            raise ValueError

        with open(imglist_file) as ifs:
            image_filename_list = ifs.read().strip().replace("\\", "/").split('\n')

        return np.array(image_filename_list)

        # self.image_filename_list = [os.path.join(self.image_dir, fpath) for fpath in fpaths]

    def read_nuswide_labels_from_file(self, train_or_test):

        if self.subset_name is "All":
            # ./TrainTestLabels/Labels_airport_Test.txt
            template = self.source_dir + "/TrainTestLabels/*" + train_or_test + ".txt"
        elif self.subset_name is "Lite":
            # ./Lite/groundtruth/Lite_Labels_airport_Test.txt
            template = self.source_dir + "/Lite/groundtruth/Lite_Labels*" + train_or_test + ".txt"
        elif self.subset_name is "Object":
            # ./OBJECT/groundtruth/bearTest.txt
            template = self.source_dir + "/OBJECT/groundtruth/*" + train_or_test + ".txt"
        elif self.subset_name is "Scene":
            # ./SCENE/groundtruth/Test_Labels_airport.txt
            template = self.source_dir + "/SCENE/groundtruth/" + train_or_test + "*.txt"
        else:
            raise ValueError

        return np.vstack([np.loadtxt(f, dtype=np.uint8) for f in sorted(glob(template))]).T

    def filter_outliers(self):
        sums = np.sum(self.labels, axis=1)
        lb = np.where(self.label_lower_bound <= sums)
        ub = np.where(sums <= self.label_upper_bound)
        return np.intersect1d(lb, ub)

    def remove_images_with_missing_labels(self):
        good_indices = self.filter_outliers()
        self.labels = self.labels[good_indices]
        self.image_filename_list = self.image_filename_list[good_indices]

    def filter_bad_data(self):
        self.remove_images_with_missing_labels()

    def create_dataset(self):

        if self.split_name in ("train",):
            labels = self.read_nuswide_labels_from_file("Train")
            image_filename_list = self.create_image_filename_list("Train")

        elif self.split_name in ("valid", "test"):
            y_testval = self.read_nuswide_labels_from_file("Test")
            x_testval = self.create_image_filename_list("Test")
            index_file = self.index_files[self.split_name]

            if os.path.exists(index_file):
                index = np.load(index_file)
                labels = y_testval[index]
                image_filename_list = x_testval[index]
            else:
                index_testval = np.arange(len(y_testval))
                index_valid, index_test, y_valid, y_test = train_test_split(
                    index_testval,
                    y_testval,
                    test_size=self.test_size/100.0,
                    random_state=self.seed)

                np.save(self.index_files['valid'], index_valid)
                np.save(self.index_files['test'], index_test)

                if self.split_name == 'valid':
                    labels = y_valid
                    image_filename_list = x_testval[index_valid]
                else:
                    labels = y_test
                    image_filename_list = x_testval[index_test]

        else:
            raise ValueError

        self.labels = labels
        self.image_filename_list = image_filename_list

    def create_concepts_names_file(self, concepts_file):
        # Comes with NUS-WIDE dataset.  Concepts81 should be in self.source_dir
        raise ValueError(concepts_file, "should already exist!!!")

    def create_object_names_file(self, concepts_file):
        # converts "OBJECT/groundtruth/bearTest.txt" to "bear"
        template = self.source_dir + "/OBJECT/groundtruth/*Test.txt"
        with open(concepts_file, 'w') as ofs:
            for f in sorted(glob(template)):
                ofs.write(f.rsplit(os.sep)[-1].split('T')[0] + '\n')

    def create_scene_names_file(self, concepts_file):
        # converts "SCENE/groundtruth/Test_Labels_airport.txt" to "airport"
        template = self.source_dir + "/SCENE/groundtruth/Test_Labels*.txt"
        with open(concepts_file, 'w') as ofs:
            for f in sorted(glob(template)):
                ofs.write(f.rsplit(os.sep)[-1].split('.')[0].split('_')[-1] + '\n')

    def load_label_names(self):

        label_names_file = os.path.join(self.source_dir, "Concepts" + str(self.n_classes) + ".txt")
        if not os.path.exists(label_names_file):
            if self.n_classes == 81:
                self.create_concepts_names_file(label_names_file)
            elif self.n_classes == 33:
                self.create_scene_names_file(label_names_file)
            elif self.n_classes == 31:
                self.create_object_names_file(label_names_file)
            else:
                raise ValueError("Incorrect number of classes. Should be one of (31, 33, 81)")

        with open(label_names_file, 'r') as ifs:
            label_names = ifs.read().strip().split('\n')
        return np.array(label_names)


class NusWideGenerator(object):
    def __init__(self,
                 image_data_generator=ImageDataGenerator(),
                 subset_name='Lite',
                 split_name='train',
                 source_dir=data_source_dir,
                 store_labels=False,
                 batch_size=1,
                 group_method='none',  # 'none' or 'random'
                 shuffle=True,
                 seed=None,
                 standardize_method='zmuv'
                 ):

        """Initialization"""
        self.subset_name = subset_name
        self.image_data_generator = image_data_generator
        self._nuswide = NUSWIDE(source_dir, subset_name, split_name)

        self._num_samples = None
        self._num_classes = None
        self._steps = None
        self._images = None
        self._labels = None
        self._label_names = None

        # self.class_ids = None
        # self.class_id_to_name = {}
        # self.class_id_to_index = {}
        # self.names = None
        # self.name_to_class_id = {}
        # self.name_to_index = {}
        # self.load_metadata()

        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle

        # self.store_labels = store_labels
        self.stored_labels = np.zeros((self.num_samples, self.num_classes)) if store_labels else None

        if seed is None:
            seed = np.uint32((time() % 1) * 1000)
        np.random.seed(seed)

        self.standardize_method = standardize_method
        self.groups = []
        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    # def load_metadata(self):
    #     cats = self._nuswide.loadCats(self._nuswide.getCatIds())
    #     cats.sort(key=lambda x: x['id'])
    #     self.class_ids = tuple([c['id'] for c in cats])
    #     self.class_id_to_name = {c['id']: c['name'] for c in cats}
    #     self.class_id_to_index = {cid: i for i, cid in enumerate(self.class_ids)}
    #     self.names = tuple([c['name'] for c in cats])
    #     self.name_to_class_id = {c['name']: c['id'] for c in cats}
    #     self.name_to_index = {cname: i for i, cname in enumerate(self.names)}

    @property
    def num_samples(self):
        if self._num_samples is None:
            self._num_samples = self._nuswide.num_samples
        return self._num_samples

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = self._nuswide.num_classes
        return self._num_classes

    @property
    def steps(self):
        if self._steps is None:
            self._steps = self.num_samples / self.batch_size
        return self._steps

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self._nuswide.labels
        return self._labels

    @property
    def label_names(self):
        if self._label_names is None:
            self._label_names = self._nuswide.load_label_names()
        return self._label_names

    @property
    def images(self):
        if self._images is None:
            self._images = self.load_image_group(np.arange(self.num_samples))
        return self._images

    def group_images(self):
        order = np.arange(self.num_samples)
        if self.group_method == 'random':
            np.random.shuffle(order)
        p = list(range(0, len(order), self.batch_size))[1:]
        self.groups = np.split(order, p)

    def load_image(self, image_index, dtype=np.uint8):
        image_filename = self._nuswide.image_filename_list[image_index]
        img_path = os.path.join(self._nuswide.image_dir, image_filename)
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img).astype(dtype)
        return np.expand_dims(x, axis=0)

    def load_image_group(self, group, dtype=np.uint8):
        return np.vstack([self.load_image(image_index, dtype=dtype) for image_index in group])

    def load_labels(self, image_index):
        return self.labels[image_index]

    def load_labels_group(self, group, dtype=np.uint8):
        return np.vstack([self.load_labels(image_index) for image_index in group])

    def preprocess_group(self, image_group):
        for index, image in enumerate(image_group):
            # image = vgg16.preprocess_input(image, mode='tf')
            if self.standardize_method == 'zmuv':
                image = self.image_data_generator.standardize(image)
            image = self.image_data_generator.random_transform(image)
            image_group[index] = image
        return image_group

    def compute_input_output(self, group):
        image_group = self.load_image_group(group, dtype=K.floatx())
        labels_group = self.load_labels_group(group)

        if self.standardize_method == 'inet':
            image_group = vgg16.preprocess_input(image_group, mode='tf')
        image_group = self.preprocess_group(image_group)

        if self.stored_labels is not None:
            for g, lg in zip(group, labels_group):
                self.stored_labels[g, :] = lg

        return image_group, labels_group

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nw_train_gen = NusWideGenerator(subset_name='Object', split_name='train')
    train_sums_0 = np.sum(nw_train_gen.labels, axis=0)
    train_sums_1 = np.sum(nw_train_gen.labels, axis=1)
    idx = np.random.randint(len(nw_train_gen.labels))
    img = nw_train_gen.load_image(idx)[0]
    plt.imshow(img)
    plt.show()
    print(nw_train_gen.labels.shape, idx, img.shape, img.min(), img.max())
    # print(label_names[np.where(nw_train_gen.labels[idx])[0]])
