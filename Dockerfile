ARG BASE_IMAGE=''
FROM ${BASE_IMAGE}

ARG TF_IO='tensorflow-io'
RUN echo "base image: ${BASE_IMAGE}"
RUN echo "tf-io: ${TF_IO}"
RUN pip install -U --no-deps ${TF_IO}

RUN python --version
RUN python -c "import tensorflow as tf ; print ('tf version: ' + tf.version.VERSION)"
RUN python -c "import tensorflow_io ; print ('tf.io version: ' + tensorflow_io.version.VERSION)"
