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

# Sample command run:
# python3 bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=1 --num_iterations=20480 --mini_batch_size=2048 --num_columns=120

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None,
                    "GCP project id.")
flags.mark_flag_as_required("project_id")
flags.DEFINE_integer("num_iterations", 1000, "Number of batchs to load.")
flags.DEFINE_integer("requested_streams", 1, "Number of streams.")
flags.DEFINE_integer("batch_size", 2048, "Batch size.")
flags.DEFINE_integer("mini_batch_size", 100, "Mini batch size - to divide num_iterations.")
flags.DEFINE_integer("num_columns", 120, "Number of columns to read.")
flags.DEFINE_bool("sloppy", False,
  "If True the implementation is allowed, for the sake of expediency, to produce"
  "elements in a non-deterministic order")
flags.DEFINE_bool("get_size_bytes", False,
  "Gets the data size, can slow down the test")


DATASET_GCP_PROJECT_ID = "alekseyv-scalableai-dev"
DATASET_ID = "criteo_kaggle_2"
TABLE_ID = "days_duplicated_columns"

COLUMN_NAMES_STRING = 'label,int1,int2,int3,int4,int5,int6,int7,int8,int9,int10,int11,int12,int13,cat1,cat2,cat3,cat4,cat5,cat6,cat7,cat8,cat9,cat10,cat11,cat12,cat13,cat14,cat15,cat16,cat17,cat18,cat19,cat20,cat21,cat22,cat23,cat24,cat25,cat26,int_11,int_12,int_13,int_14,int_15,int_16,int_17,int_18,int_19,int_110,int_111,int_112,int_113,cat_11,cat_12,cat_13,cat_14,cat_15,cat_16,cat_17,cat_18,cat_19,cat_110,cat_111,cat_112,cat_113,cat_114,cat_115,cat_116,cat_117,cat_118,cat_119,cat_120,cat_121,cat_122,cat_123,cat_124,cat_125,cat_126,int_21,int_22,int_23,int_24,int_25,int_26,int_27,int_28,int_29,int_210,int_211,int_212,int_213,cat_21,cat_22,cat_23,cat_24,cat_25,cat_26,cat_27,cat_28,cat_29,cat_210,cat_211,cat_212,cat_213,cat_214,cat_215,cat_216,cat_217,cat_218,cat_219,cat_220,cat_221,cat_222,cat_223,cat_224,cat_225,cat_226'

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
  column_names = COLUMN_NAMES_STRING.split(',')[:FLAGS.num_columns]
  selected_fields = { name:{ 'mode': BigQueryClient.FieldMode.NULLABLE, 'output_type':(dtypes.string if name.startswith('cat') else dtypes.int64) }  for name in column_names }

  print('Batch size: %d, Sloppy: %s' % (batch_size, FLAGS.sloppy))
  client = BigQueryClient()
  read_session = client.read_session(
      "projects/" + FLAGS.project_id,
      DATASET_GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      selected_fields=selected_fields,
      requested_streams=FLAGS.requested_streams)

  streams = read_session.get_streams()
  print('Requested %d streams, BigQuery returned %d streams' % (
    len(streams),
    FLAGS.requested_streams))
  dataset = read_session.parallel_read_rows(sloppy=FLAGS.sloppy)
  if batch_size != 1:
    dataset = dataset.batch(batch_size)
  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)

  start = time.time()
  n = 0
  mini_batch = FLAGS.mini_batch_size
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
