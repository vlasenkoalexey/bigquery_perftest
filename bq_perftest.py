from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import time
import os

import tensorflow as tf
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

DATASET_GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "samples"
TABLE_ID = "wikipedia"


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
  print('Requested %d streams, BigQuery returned %d streams' % (
    len(streams), 
    FLAGS.requested_streams))
  dataset = read_session.parallel_read_rows(sloppy=FLAGS.sloppy).batch(batch_size)
  itr = dataset.make_one_shot_iterator()

  start = time.time()
  n = 0
  mini_batch = 100
  for i in range(num_iterations // mini_batch):
    local_start = time.time()
    start_n = n
    for j in range(mini_batch):
      n += batch_size
      itr.get_next()
    local_end = time.time()
    print('Processed %d entries in %f seconds. [%f] examples/s' % (
        n - start_n, local_end - local_start,
        (mini_batch * batch_size) / (local_end - local_start)))
  end = time.time()
  print('Processed %d entries in %f seconds. [%f] examples/s' % (
      n, end - start,
      n / (end - start)))


if __name__ == '__main__':
  app.run(run_benchmark)
