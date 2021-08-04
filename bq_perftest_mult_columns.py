from absl import app
from absl import flags
import contextlib
import sys
import time
import os
import numpy
import inspect

import tensorflow as tf
import tensorflow_io as tf_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
a
# Sample command run:
# python3 bq_perftest_mult_columns.py --project_id=alekseyv-gke --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=20

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project id.")
flags.mark_flag_as_required("project_id")
flags.DEFINE_integer("num_iterations", 1000, "Number of batches to load.")
flags.DEFINE_integer("num_warmup_iterations", 10,
                     "Number of warmup batches to load that doesn'ta count towards benchmark results.")
flags.DEFINE_integer("requested_streams", 1, "Number of streams.")
flags.DEFINE_integer("batch_size", 2048, "Batch size.")
flags.DEFINE_integer("prefetch_size", None, "Prefetch size.")
flags.DEFINE_integer(
    "mini_batch_size", 100, "Mini batch size - to divide num_iterations."
)
flags.DEFINE_integer("num_columns", 120, "Number of columns to read.")
flags.DEFINE_bool(
    "sloppy",
    False,
    "If True the implementation is allowed, for the sake of expediency, to produce"
    "elements in a non-deterministic order",
)
flags.DEFINE_bool("get_size_bytes", False,
                  "Gets the data size, can slow down the test")
flags.DEFINE_enum("format", "AVRO", ["AVRO", "ARROW"],
                  "Serialization format - AVRO or ARROW")
flags.DEFINE_enum("compression_type", "", [
                  "", "GZIP"], "Data compression algorithm, only GCS compression is supported")
flags.DEFINE_enum("data_source", "BQ", [
                  "BQ", "GCS"], "Data source, BQ or GCS.")
flags.DEFINE_string("profile_log_path", "", "If this argument is specified, benchmark is going to be profiled and results dumped to the specified folder. Later you can run tensorboard --logidr=<profile_log_path> to inspect profile results. GCS file format is supported.")

# Dataset has ~45M rows
DATASET_GCP_PROJECT_ID = "alekseyv-gke"
DATASET_ID = "criteo_kaggle_2"
TABLE_ID = "days_duplicated_columns"

COLUMN_NAMES_STRING = "label,int1,int2,int3,int4,int5,int6,int7,int8,int9,int10,int11,int12,int13,cat1,cat2,cat3,cat4,cat5,cat6,cat7,cat8,cat9,cat10,cat11,cat12,cat13,cat14,cat15,cat16,cat17,cat18,cat19,cat20,cat21,cat22,cat23,cat24,cat25,cat26,int_11,int_12,int_13,int_14,int_15,int_16,int_17,int_18,int_19,int_110,int_111,int_112,int_113,cat_11,cat_12,cat_13,cat_14,cat_15,cat_16,cat_17,cat_18,cat_19,cat_110,cat_111,cat_112,cat_113,cat_114,cat_115,cat_116,cat_117,cat_118,cat_119,cat_120,cat_121,cat_122,cat_123,cat_124,cat_125,cat_126,int_21,int_22,int_23,int_24,int_25,int_26,int_27,int_28,int_29,int_210,int_211,int_212,int_213,cat_21,cat_22,cat_23,cat_24,cat_25,cat_26,cat_27,cat_28,cat_29,cat_210,cat_211,cat_212,cat_213,cat_214,cat_215,cat_216,cat_217,cat_218,cat_219,cat_220,cat_221,cat_222,cat_223,cat_224,cat_225,cat_226"
# Dataset has 51 files, ~430K rows total
GCS_DATASET_FILE_PATTERN = 'gs://alekseyv-scalableai-dev/criteo_kaggle_2_days_duplicated_columns/data_*'


def get_dataset_from_gcs():
  column_names = COLUMN_NAMES_STRING.split(",")[: FLAGS.num_columns]
  feature_description = {name: tf.io.FixedLenFeature([], tf.string, default_value='') if name.startswith(
      'cat') else tf.io.FixedLenFeature([], tf.int64, default_value=0) for name in column_names}
  filenames = tf.io.gfile.glob(GCS_DATASET_FILE_PATTERN)

  def parse_batch_func(tf_records_batch): return tf.io.parse_example(
      tf_records_batch, feature_description)
  compression_type = None if FLAGS.compression_type == "" else FLAGS.compression_type

  def map_func(filename):
    dataset = tf.data.TFRecordDataset(
        filename, compression_type=compression_type)
    if FLAGS.batch_size != 1:
      dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(parse_batch_func)
    return dataset

  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset \
      .interleave(
          map_func=map_func,
          cycle_length=FLAGS.requested_streams,
          num_parallel_calls=FLAGS.requested_streams,
          deterministic=not(FLAGS.sloppy))

  if FLAGS.prefetch_size is not None:
    dataset = dataset.prefetch(FLAGS.prefetch_size)

  return dataset.repeat()


def get_dataset_from_bigquery():
  column_names = COLUMN_NAMES_STRING.split(",")[: FLAGS.num_columns]
  selected_fields = column_names
  output_types = list((dtypes.string if name.startswith(
      "cat") else dtypes.int64) for name in column_names)

  client = BigQueryClient()

  read_session = None
  if 'data_format' in inspect.getfullargspec(client.read_session).args:
    read_session = client.read_session(
        "projects/" + FLAGS.project_id,
        DATASET_GCP_PROJECT_ID,
        TABLE_ID,
        DATASET_ID,
        selected_fields=selected_fields,
        output_types=output_types,
        requested_streams=FLAGS.requested_streams,
        data_format=BigQueryClient.DataFormat.AVRO if FLAGS.format == 'AVRO' else BigQueryClient.DataFormat.ARROW
    )
  else:
    # ARROW only supported starting TF.IO 0.14
    if FLAGS.format == 'ARROW':
      print('ARROW is not supported in this version of tensorflow.io')
      exit(1)
    read_session = client.read_session(
        "projects/" + FLAGS.project_id,
        DATASET_GCP_PROJECT_ID,
        TABLE_ID,
        DATASET_ID,
        selected_fields=selected_fields,
        output_types=output_types,
        requested_streams=FLAGS.requested_streams)

  streams = read_session.get_streams()
  print(
      "Requested %d streams, BigQuery returned %d streams"
      % (FLAGS.requested_streams, len(streams))
  )

  def read_rows(stream):
    dataset = read_session.read_rows(stream)
    if FLAGS.batch_size != 1:
      dataset = dataset.batch(FLAGS.batch_size)
    return dataset

  streams_count = tf.size(streams)
  streams_count64 = tf.cast(streams_count, dtype=tf.int64)
  streams_ds = tf.data.Dataset.from_tensor_slices(streams)
  dataset = streams_ds.interleave(
      read_rows,
      cycle_length=streams_count64,
      num_parallel_calls=streams_count64,
      deterministic=not(FLAGS.sloppy))

  if FLAGS.prefetch_size is not None:
    dataset = dataset.prefetch(FLAGS.prefetch_size)

  return dataset.repeat()


def get_row_size_bytes(row):
  size_bytes = 0
  for key, value in row.items():
    lst = value.numpy().tolist()
    for elem in lst:
      size_bytes += sys.getsizeof(elem)
  return size_bytes


def run_benchmark(_):
  print('tf version: ' + tf.version.VERSION)
  if hasattr(tf_io, 'version'):
    print('tf.io version: ' + tf_io.version.VERSION)

  dataset = None
  if FLAGS.data_source == 'BQ':
    print('Reading from BigQuery')
    dataset = get_dataset_from_bigquery()
  elif FLAGS.data_source == 'GCS':
    print('Reading from GCS')
    dataset = get_dataset_from_gcs()
  else:
    print('Invalid data_source:' + FLAGS.data_source)
    exit(1)

  num_iterations = FLAGS.num_iterations
  batch_size = FLAGS.batch_size
  print("Reading from: %s, Batch size: %d, Sloppy: %s" %
        (FLAGS.data_source, FLAGS.batch_size, FLAGS.sloppy))

  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
  mini_batch = FLAGS.mini_batch_size

  print("Warming up for: %d steps" % FLAGS.num_warmup_iterations)
  for j in range(FLAGS.num_warmup_iterations):
    _ = itr.get_next()

  print("Running benchmark")
  n = 0
  size_bytes = 0
  def profiler_ctx_manager(_): return contextlib.suppress()
  if FLAGS.profile_log_path != '':
    tf.profiler.experimental.start(FLAGS.profile_log_path)

    def profiler_ctx_manager(step): return tf.profiler.experimental.Trace(
        'step', step_num=step, _r=1)
  start = time.time()
  for i in range(num_iterations // mini_batch):
    local_start = time.time()
    start_n = n
    local_size_bytes = 0
    for j in range(mini_batch):
      n += batch_size
      with profiler_ctx_manager(i * mini_batch + j):
        row = itr.get_next()
      if FLAGS.get_size_bytes:
        local_size_bytes += get_row_size_bytes(row)

    size_bytes += local_size_bytes

    local_end = time.time()
    print(
        "Processed %d entries in %f seconds. [%f] examples/s"
        % (
            n - start_n,
            local_end - local_start,
            (mini_batch * batch_size) / (local_end - local_start),
        )
    )
    if FLAGS.get_size_bytes:
      print(
          "%d bytes. [%f] bytes/s"
          % (local_size_bytes, (local_size_bytes) / (local_end - local_start))
      )

  end = time.time()
  if FLAGS.profile_log_path != '':
    tf.profiler.experimental.stop()
  print(
      "Processed %d entries in %f seconds. [%f] examples/s"
      % (n, end - start, n / (end - start))
  )
  if FLAGS.get_size_bytes:
    print(
        "%d bytes. [%f] MB/s"
        % (size_bytes, (size_bytes) / (end - start) / 1024 / 1024)
    )
  return n / (end - start)


if __name__ == "__main__":
  app.run(run_benchmark)
