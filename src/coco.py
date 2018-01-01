import os
import time
import random
import threading
import numpy as np

from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16

from pycocotools.coco import COCO

EPS = np.finfo(float).eps
split_name_dict = {'valid': 'val', 'train': 'train', 'test': 'test'}
data_source_dir = "/media/Borg_LS/DATA"



class CocoGenerator(object):
    def __init__(self,
                 image_data_generator=ImageDataGenerator(),
                 subset_name='2017',
                 split_name='train',
                 source_dir=data_source_dir,
                 store_labels=False,
                 batch_size=1,
                 group_method='none',  # one of 'none', 'random', 'ratio'
                 shuffle=True,
                 seed=None,
                 standardize_method='zmuv',
                 llb=None,
                 lub=None,
                 ):

        """Initialization"""
        self.set_name = split_name_dict[split_name] + subset_name
        self.image_data_generator = image_data_generator
        self.source_dir = os.path.join(source_dir, 'coco')
        self._coco = COCO(os.path.join(self.source_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self._coco.getImgIds()

        if llb is not None or lub is not None:
            self.remove_outliers = True
        else:
            self.remove_outliers = False

        self.label_lower_bound = llb
        self.label_upper_bound = lub
        self._num_samples = None
        self._num_classes = None
        self._steps = None
        self._good_indices = None
        self._images = None
        self._labels = None
        self._label_names = None

        self.class_ids = None
        self.class_id_to_name = {}
        self.class_id_to_index = {}
        self.names = None
        self.name_to_class_id = {}
        self.name_to_index = {}

        self.load_metadata()

        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle

        # self.store_labels = store_labels
        self.stored_labels = np.zeros((self.num_samples, self.num_classes)) if store_labels else None

        if seed is None:
            seed = np.uint32((time.time() % 1) * 1000)
        np.random.seed(seed)

        self.standardize_method = standardize_method
        self.groups = []
        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    def load_metadata(self):
        cats = self._coco.loadCats(self._coco.getCatIds())
        cats.sort(key=lambda x: x['id'])

        self.class_ids = tuple([c['id'] for c in cats])
        self.class_id_to_name = {c['id']: c['name'] for c in cats}
        self.class_id_to_index = {cid: i for i, cid in enumerate(self.class_ids)}
        self.names = tuple([c['name'] for c in cats])
        self.name_to_class_id = {c['name']: c['id'] for c in cats}
        self.name_to_index = {cname: i for i, cname in enumerate(self.names)}

    def filter_outliers(self):
        labels = self.load_labels_group(np.arange(self.num_samples))
        sums = np.sum(labels, axis=1)
        lb = np.where(self.label_lower_bound <= sums)
        ub = np.where(sums <= self.label_upper_bound)
        return np.intersect1d(lb, ub)

    @property
    def num_samples(self):
        if self._num_samples is None:
            self._num_samples = len(self.image_ids)
        return self._num_samples

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = len(self.class_ids)
        return self._num_classes

    @property
    def steps(self):
        if self._steps is None:
            self._steps = self.num_samples / self.batch_size
        return self._steps

    @property
    def good_indices(self):
        if self._good_indices is None:
            self._good_indices = self.filter_outliers()
        return self._good_indices

    @property
    def images(self):
        if self._images is None:
            indices = self.good_indices if self.remove_outliers else np.arange(self.num_samples)
            self._images = self.load_image_group(indices)
        return self._images

    @property
    def labels(self):
        if self._labels is None:
            indices = self.good_indices if self.remove_outliers else np.arange(self.num_samples)
            self._labels = self.load_labels_group(indices)
        return self._labels

    @property
    def label_names(self):
        if self._label_names is None:
            self._label_names = np.array(self.names)
        return self._label_names

    def group_images(self):
        # determine the order of the images
        order = np.arange(self.num_samples)
        if self.group_method == 'random':
            np.random.shuffle(order)

        p = list(range(0, len(order), self.batch_size))[1:]
        self.groups = np.split(order, p)

    def load_image(self, image_index, dtype=np.uint8):
        image = self._coco.loadImgs(self.image_ids[image_index])[0]
        img_path = os.path.join(self.source_dir, 'images', self.set_name, image['file_name'])
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img).astype(dtype)
        return np.expand_dims(x, axis=0)

    def load_image_group(self, group, dtype=np.uint8):
        return np.vstack([self.load_image(image_index, dtype=dtype) for image_index in group])

    def load_labels(self, image_index, dtype=np.uint8):
        label_ids = self._coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        labels = np.zeros((self.num_classes,), dtype=dtype)

        if len(label_ids) == 0:
            return labels

        indexes = np.array([self.class_id_to_index[a['category_id']] for a in self._coco.loadAnns(label_ids)])
        labels[indexes] = 1
        return labels

    def load_labels_group(self, group, dtype=np.uint8):
        return np.vstack([self.load_labels(image_index, dtype=dtype) for image_index in group])

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
        labels_group = self.load_labels_group(group, dtype=K.floatx())

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

    coco_valid_gen = CocoGenerator(subset_name='2017', split_name='valid')
    valid_sums_0 = np.sum(coco_valid_gen.labels, axis=0)
    valid_sums_1 = np.sum(coco_valid_gen.labels, axis=1)
    index = np.random.randint(len(coco_valid_gen.labels))
    img = coco_valid_gen.load_image(index)[0]
    plt.imshow(img)
    plt.show()
    print(coco_valid_gen.labels.shape, index, img.shape, img.min(), img.max())
    print(coco_valid_gen.label_names[np.where(coco_valid_gen.labels[index])[0]])

