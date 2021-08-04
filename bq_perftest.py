from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import sys
import time
import os
import numpy

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession

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


DATASET_GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "samples"
TABLE_ID = "wikipedia"

def get_row_size_bytes(row):
  size_bytes = 0
  for key, value in row.items():
    lst = value.numpy().tolist()
    for elem in lst:
      size_bytes += sys.getsizeof(elem)
  return size_bytes

def run_benchmark(_):
  num_iterations = FLAGS.num_iterations
  batch_size = FLAGS.batch_size
  print('Batch size: %d, Sloppy: %s' % (batch_size, FLAGS.sloppy))
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
  # print('Requested %d streams, BigQuery returned %d streams' % (
  #   len(streams), 
  #   FLAGS.requested_streams))
  dataset = read_session.parallel_read_rows(sloppy=FLAGS.sloppy).batch(batch_size)
  itr = dataset.make_one_shot_iterator()

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
