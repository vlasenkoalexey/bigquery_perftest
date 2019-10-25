from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import sys
import time
import os
import numpy

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession

if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None,
                    "GCP project id.")
flags.mark_flag_as_required("project_id")
flags.DEFINE_integer("num_iterations", 1000, "Number of batchs to load.")
flags.DEFINE_integer("requested_streams", 1, "Number of streams.")
flags.DEFINE_integer("batch_size", 2048, "Batch size.")
flags.DEFINE_bool("sloppy", False,
  "If True the implementation is allowed, for the sake of expediency, to produce"
  "elements in a non-deterministic order")
flags.DEFINE_bool("get_size_bytes", False,
  "Gets the data size, can slow down the test")
flags.DEFINE_enum("data_source", None, ["BQ", "GCS"], "Data source, BQ or GCS.")
flags.mark_flag_as_required("data_source")

DATASET_GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "samples"
TABLE_ID = "wikipedia"
GCS_DATASET_FILE_PATTERN='gs://bigquery-public-data/samples/wikipedia/samples_wikipedia_*'
FEATURE_DESCRIPTION = {
    'title': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'num_characters': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'language': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'timestamp': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'wp_namespace': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'contributor_username': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def get_row_size_bytes(row):
  size_bytes = 0
  for key, value in row.items():
    lst = value.numpy().tolist()
    for elem in lst:
      size_bytes += sys.getsizeof(elem)
  return size_bytes

def get_dataset_from_gcs():
  filenames = tf.io.gfile.glob(GCS_DATASET_FILE_PATTERN)
  dataset = tf.data.Dataset.from_tensor_slices(filenames) \
    .repeat() \
    .apply(tf.data.experimental.parallel_interleave(
      map_func=tf.data.TFRecordDataset,
      cycle_length=FLAGS.requested_streams,
      sloppy=FLAGS.sloppy)) \
      .batch(FLAGS.batch_size) \
      .map (lambda tf_records_batch:
        tf.io.parse_example(tf_records_batch, FEATURE_DESCRIPTION))
  return dataset

def get_dataset_from_bigquery():
  print('Batch size: %d, Sloppy: %s' % (FLAGS.batch_size, FLAGS.sloppy))
  client = BigQueryClient()
  read_session = client.read_session(
      "projects/" + FLAGS.project_id,
      DATASET_GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      ["title",
       "id",
       "num_characters",
       "language",
       "timestamp",
       "wp_namespace",
       "contributor_username"],
      [dtypes.string,
       dtypes.int64,
       dtypes.int64,
       dtypes.string,
       dtypes.int64,
       dtypes.int64,
       dtypes.string],
      requested_streams=FLAGS.requested_streams
      )

  streams = read_session.get_streams()
  print('Requested %d streams, BigQuery returned %d streams' % (
    len(streams),
    FLAGS.requested_streams))
  dataset = read_session.parallel_read_rows(sloppy=FLAGS.sloppy).batch(FLAGS.batch_size)
  return dataset

def run_benchmark(_):
  data_source = FLAGS.data_source
  batch_size = FLAGS.batch_size
  dataset = None
  if data_source == 'BQ':
    print('Reading from BigQuery')
    dataset = get_dataset_from_bigquery()
  elif data_source == 'GCS':
    print('Reading from GCS')
    dataset = get_dataset_from_gcs()
  else:
    print('Invalid data_source:' + data_source)
    exit(1)

  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  num_iterations = FLAGS.num_iterations
  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
  start = time.time()
  n = 0
  mini_batch = 100
  size_bytes = 0
  for i in range(num_iterations // mini_batch):
    local_start = time.time()
    start_n = n
    local_size_bytes = 0
    for j in range(mini_batch):
      n += batch_size
      row = itr.get_next()
      if FLAGS.get_size_bytes:
        local_size_bytes += get_row_size_bytes(row)
    size_bytes += local_size_bytes

    local_end = time.time()
    print('Processed %d entries in %f seconds. [%f] examples/s' % (
        n - start_n, local_end - local_start,
        (mini_batch * batch_size) / (local_end - local_start)))
    if FLAGS.get_size_bytes:
      print('%d bytes. [%f] bytes/s' % (
          local_size_bytes,
          (local_size_bytes) / (local_end - local_start)))

  end = time.time()
  print('Processed %d entries in %f seconds. [%f] examples/s' % (
      n, end - start,
      n / (end - start)))
  if FLAGS.get_size_bytes:
    print('%d bytes. [%f] MB/s' % (
        size_bytes,
        (size_bytes) / (end - start) / 1024 / 1024))


if __name__ == '__main__':
  app.run(run_benchmark)
