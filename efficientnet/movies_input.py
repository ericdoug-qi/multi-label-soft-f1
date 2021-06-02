# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: movies_input.py
   Description : 
   Author : ericdoug
   date：2021/6/2
-------------------------------------------------
   Change Activity:
         2021/6/2: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os

# third packages
from absl import logging
import six
import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import preprocessing

# my packages
from imagenet_input import ImageNetTFExampleInput


class MoviesInput(ImageNetTFExampleInput):

    def __init__(self,
                 is_training,
                 use_bfloat16,
                 transpose_input,
                 data_dir,
                 image_size=224,
                 num_parallel_calls=64,
                 cache=False,
                 num_label_classes=1000,
                 include_background_label=False,
                 augment_name=None,
                 mixup_alpha=0.0,
                 randaug_num_layers=None,
                 randaug_magnitude=None,
                 resize_method=None,
                 holdout_shards=None):
        """Create an input from TFRecord files.

        Args:
          is_training: `bool` for whether the input is for training
          use_bfloat16: If True, use bfloat16 precision; else use float32.
          transpose_input: 'bool' for whether to use the double transpose trick
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
          image_size: `int` for image size (both width and height).
          num_parallel_calls: concurrency level to use when reading data from disk.
          cache: if true, fill the dataset by repeating from its cache.
          num_label_classes: number of label classes. Default to 1000 for ImageNet.
          include_background_label: if true, label #0 is reserved for background.
          augment_name: `string` that is the name of the augmentation method
              to apply to the image. `autoaugment` if AutoAugment is to be used or
              `randaugment` if RandAugment is to be used. If the value is `None` no
              no augmentation method will be applied applied. See autoaugment.py
              for more details.
          mixup_alpha: float to control the strength of Mixup regularization, set
              to 0.0 to disable.
          randaug_num_layers: 'int', if RandAug is used, what should the number of
            layers be. See autoaugment.py for detailed description.
          randaug_magnitude: 'int', if RandAug is used, what should the magnitude
            be. See autoaugment.py for detailed description.
          resize_method: If None, use bicubic in default.
          holdout_shards: number of holdout training shards for validation.
        """
        super(MoviesInput, self).__init__(
            is_training=is_training,
            image_size=image_size,
            use_bfloat16=use_bfloat16,
            transpose_input=transpose_input,
            num_label_classes=num_label_classes,
            include_background_label=include_background_label,
            augment_name=augment_name,
            mixup_alpha=mixup_alpha,
            randaug_num_layers=randaug_num_layers,
            randaug_magnitude=randaug_magnitude)
        self.data_dir = data_dir
        if self.data_dir == 'null' or not self.data_dir:
            self.data_dir = None
        self.num_parallel_calls = num_parallel_calls
        self.cache = cache
        self.holdout_shards = holdout_shards

    def _get_null_input(self, data):
        """Returns a null image (all black pixels).

        Args:
          data: element of a dataset, ignored in this method, since it produces
              the same null image regardless of the element.

        Returns:
          a tensor representing a null image.
        """
        del data  # Unused since output is constant regardless of input
        return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
        if self.use_bfloat16 else tf.float32)

    def dataset_parser(self, value):
        """See base class."""
        if not self.data_dir:
            return value, tf.constant(0., tf.float32, (1000,))
        return super(MoviesInput, self).dataset_parser(value)

    def parse_function(self, filename, label):
        """Function that returns a tuple of normalized image array and labels array.
        Args:
            filename: string representing path to image
            label: 0/1 one-dimensional array of size N_LABELS
        """
        # Read an image from a file
        image_string = tf.io.read_file(filename)
        # Decode it into a dense vector
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, [self.image_size, self.image_size])
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0
        # return image_normalized, label
        new_features = {'feature': image_normalized}
        new_label = {'label': label}
        return new_features, new_label

    def create_dataset(self, filenames, labels, batch_size, current_host, num_hosts, is_training=True):
        """Load and parse dataset.
        Args:
            filenames: list of image paths
            labels: numpy array of shape (BATCH_SIZE, N_LABELS)
            is_training: boolean to indicate training mode
        """

        # Create a first dataset of file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        dataset = dataset.shard(num_hosts, current_host)

        # Parse and preprocess observations in parallel
        dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_training:
            if self.cache:
                # This is a small dataset, only load it once, and keep it in memory.
                dataset = dataset.cache()
                # Shuffle the data each buffer size
                dataset = dataset.shuffle(buffer_size=16 * 1024, seed=44)
            else:
                dataset = dataset.shuffle(16 * 1024, seed=44)

        # Batch the data for multiple steps
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Fetch batches in the background while the model is training.
        dataset = dataset.prefetch(1)

        options = tf.data.Options()
        options.experimental_optimization.autotune = True

        return dataset.with_options(options)

    def make_source_dataset(self, index, num_hosts):
        """See base class."""
        if not self.data_dir:
            logging.info('Undefined data_dir implies null input')
            return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

        if self.holdout_shards:
            if self.is_training:
                filenames = [
                    os.path.join(self.data_dir, 'train-%05d-of-01024' % i)
                    for i in range(self.holdout_shards, 1024)
                ]
            else:
                filenames = [
                    os.path.join(self.data_dir, 'train-%05d-of-01024' % i)
                    for i in range(0, self.holdout_shards)
                ]
            for f in filenames[:10]:
                logging.info('datafiles: %s', f)
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
        else:
            file_pattern = os.path.join(
                self.data_dir, 'train-*' if self.is_training else 'validation-*')
            logging.info('datafiles: %s', file_pattern)
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

        # For multi-host training, we want each hosts to always process the same
        # subset of files.  Each host only sees a subset of the entire dataset,
        # allowing us to cache larger datasets in memory.
        dataset = dataset.shard(num_hosts, index)

        if self.is_training and not self.cache:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls,
            num_parallel_calls=self.num_parallel_calls, deterministic=False)

        if self.cache:
            dataset = dataset.cache().shuffle(1024 * 16).repeat()
        else:
            dataset = dataset.shuffle(1024)
        return dataset

    def input_fn(self, params):

        batch_size = params['batch_size']

        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1

        logging.info("load movie datas")
        movies = pd.read_csv(os.path.join(self.data_dir, 'movies.csv'))
        label_freq = movies['Genre'].apply(lambda s: str(s).split('|')).explode().value_counts().sort_values(
            ascending=False)
        rare = list(label_freq[label_freq < 1000].index)
        movies['Genre'] = movies['Genre'].apply(lambda s: [l for l in str(s).split('|') if l not in rare])
        X_train, X_val, y_train, y_val = train_test_split(movies['imdbId'], movies['Genre'], test_size=0.2,
                                                          random_state=44)
        X_train = [os.path.join(self.data_dir, 'movie_poster/images', str(f) + '.jpg') for f in X_train]
        X_val = [os.path.join(self.data_dir, 'movie_poster/images', str(f) + '.jpg') for f in X_val]

        y_train = list(y_train)
        y_val = list(y_val)

        # X = [os.path.join(self.data_dir, 'movie_poster/images', str(f) + '.jpg') for f in movies['imdbId']]
        # y = movies['Genre']

        # y = list(y)

        mlb = MultiLabelBinarizer()
        # mlb.fit(y)
        # y_bin = mlb.transform(y)
        mlb.fit(y_train)
        y_train_bin = mlb.transform(y_train)
        y_val_bin = mlb.transform(y_val)

        if self.is_training:
            return self.create_dataset(X_train, y_train_bin, batch_size, current_host, num_hosts, self.is_training)
        else:
            return self.create_dataset(X_val, y_val_bin, batch_size, current_host, num_hosts, self.is_training)
