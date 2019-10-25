from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import time
import os

import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string("data_files_pattern", "",
                    "TFRecords input pattern.")
flags.mark_flag_as_required("data_files_pattern")
flags.DEFINE_integer("num_iterations", 1000, "Number of batchs to load.")


def input_fn(data_files_pattern,
             batch_size,
             num_iterations=1):
  filenames = tf.gfile.Glob(data_files_pattern)
  print (filenames)
  dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()

  def _read_fn(f):
    return tf.data.TFRecordDataset(f)

  dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      map_func=_read_fn,
      cycle_length=20,
      block_length=1,
      sloppy=True,
      buffer_output_elements=50000,
      prefetch_input_elements=40))
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.prefetch(5)
  return dataset


def run_benchmark(_):
  num_iterations = FLAGS.num_iterations
  batch_size = 2048
  print('started')
  dataset = input_fn(
        data_files_pattern=FLAGS.data_files_pattern,
        batch_size=batch_size)
  itr = dataset.make_one_shot_iterator()
  size = tf.shape(itr.get_next())[0]
  with tf.Session() as sess:
    size_callable = sess.make_callable(size)
    start = time.time()
    n = 0
    mini_batch = 100
    for i in range(num_iterations // mini_batch):
      local_start = time.time()
      start_n = n
      for j in range(mini_batch):
        n += size_callable()
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
