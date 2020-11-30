import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_FEATURE_DESCR = {
    "image/channels": tf.io.FixedLenFeature([], tf.int64),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    "image/class/raw": tf.io.FixedLenFeature([], tf.int64),
    "image/class/source": tf.io.FixedLenFeature([], tf.int64),
    "image/class/text": tf.io.FixedLenFeature([], tf.string),
    "image/colorspace": tf.io.FixedLenFeature([], tf.string),
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/format": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/id": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
}

def load(data_dir,
         shuffle_files=False,
         as_supervised=True,):
    list_ds = tf.data.Dataset.list_files(data_dir + "/*",
                                         shuffle=shuffle_files,
                                         seed=78165)
    raw_ds = tf.data.TFRecordDataset(list_ds)
    parsed_ds = raw_ds.map(_parse_image_function)
    labelled_ds = parsed_ds.map(lambda x: _process_record(x, as_supervised),
                                num_parallel_calls=AUTOTUNE)

    return labelled_ds

def prepare_for_training(ds,
                         cache=True,
                         batch_size=BATCH_SIZE,
                         shuffle_buffer_size=None):
    # If this is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            print("Creating temp file - %s" % cache)
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size,
                        reshuffle_each_iteration=True)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def _process_record(record, as_supervised=True):
    img = record["image/encoded"]
    img = _decode_img(img)

    if as_supervised:
        index = record["image/class/label"]
        # label = tf.one_hot(index, depth=4, on_value=True, off_value=False)
        label = index
        return img, label
    else:
        return img

def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.

    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    return img

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto,
                                      IMAGE_FEATURE_DESCR)