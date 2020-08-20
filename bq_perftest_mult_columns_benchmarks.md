# Setup
```sh
export PROJECT_ID="alekseyv-scalableai-dev"
gcloud iam service-accounts keys create ./service_account_key.json --iam-account=${USER}-service@${PROJECT_ID}.iam.gserviceaccount.com
```

#Benchmarks
##1 stream

###Local TF2.3 + custom TF.IO build
```sh
pip install tensorflow==2.3
bazel build -s --verbose_failures --copt=-msse4.2 --copt=-mavx --compilation_mode=opt //tensorflow_io/...
TFIO_DATAPATH=bazel-bin python3 -m tensorflow_io.bigquery_experiments.bq_perftest_mult_columns --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 21.747404 seconds. [9417.216019] examples/s
```

###TFE 2.3
```sh
docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json gcr.io/deeplearning-platform-release/tf2-cpu.2-3 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#undefined symbol: _ZNK10tensorflow10FileSystem8BasenameEN4absl11string_viewE']

###TFE 2.3 + fixed tf.io dependency
```sh
docker build --build-arg BASE_IMAGE=gcr.io/deeplearning-platform-release/tf2-cpu.2-3 --build-arg TF_IO=tensorflow-io==0.15.0 -f Dockerfile -t tfent2.3 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tfent2.3 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 25.733577 seconds. [7958.474008] examples/s

###TFE 2.2
```sh
docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json gcr.io/deeplearning-platform-release/tf2-cpu.2-2 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 23.450918 seconds. [8733.133529] examples/s
```

###TFE 2.1
```sh
docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json gcr.io/deeplearning-platform-release/tf2-cpu.2-1 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 38.842539 seconds. [5272.569904] examples/s
```

###TFE 2.0
```sh
docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json gcr.io/deeplearning-platform-release/tf2-cpu.2-0 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 56.543637 seconds. [3621.981402] examples/s
```

###TF 2.3 + tf.io nightly
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.3.0 --build-arg TF_IO=tensorflow-io-nightly==0.15.0.dev20200814050520 -f Dockerfile -t tf2.3_tfio_nightly . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.3_tfio_nightly python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 18.046976 seconds. [11348.161691] examples/s
```

###TF 2.3
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.3.0 --build-arg TF_IO=tensorflow-io==0.15.0 -f Dockerfile -t tf2.3 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.3 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 19.775841 seconds. [10356.070320] examples/s
```

###TF 2.2
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.2.0 --build-arg TF_IO=tensorflow-io==0.14.0 -f Dockerfile -t tf2.2 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.2 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 20.083804 seconds. [10197.271327] examples/s
```

###TF 2.1
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.1.0-py3 --build-arg TF_IO=tensorflow-io==0.12.0 -f Dockerfile -t tf2.1 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.1 python /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 32.097766 seconds. [6380.506277] examples/s
```

###TF 2.0
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.0.0-py3 --build-arg TF_IO=tensorflow-io==0.10.0 -f Dockerfile -t tf2.0 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.0 python /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 68.433091 seconds. [2992.704230] examples/s
```

##10 streams

###TFE 2.2 - 10 streams
```sh
docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json gcr.io/deeplearning-platform-release/tf2-cpu.2-2 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 11.804269 seconds. [17349.655691] examples/s
```

###TFE 2.3 + fixed tf.io dependency - 10 streams
```sh
docker build --build-arg BASE_IMAGE=gcr.io/deeplearning-platform-release/tf2-cpu.2-3 --build-arg TF_IO=tensorflow-io==0.15.0 -f Dockerfile -t tfent2.3 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tfent2.3 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 10.790630 seconds. [18979.430034] examples/s
```

###TF 2.2 - 10 streams
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.2.0 --build-arg TF_IO=tensorflow-io==0.14.0 -f Dockerfile -t tf2.2 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.2 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 11.046746 seconds. [18539.395300] examples/s
```

###TF 2.3 - 10 streams
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.3.0 --build-arg TF_IO=tensorflow-io==0.15.0 -f Dockerfile -t tf2.3 . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.3 python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 10.921412 seconds. [18752.154859] examples/s
```

###TF 2.3 + tf.io nightly - 10 streams
```sh
docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.3.0 --build-arg TF_IO=tensorflow-io-nightly==0.15.0.dev20200814050520 -f Dockerfile -t tf2.3_tfio_nightly . && docker run -v ${PWD}:/v -e GOOGLE_APPLICATION_CREDENTIALS=/v/service_account_key.json tf2.3_tfio_nightly python3 /v/bq_perftest_mult_columns.py --project_id=alekseyv-scalableai-dev --batch_size=2048 --num_iterations=100 --mini_batch_size=10 --num_columns=120 --requested_streams=1 --sloppy=true --format=AVRO
#Processed 204800 entries in 9.544082 seconds. [21458.323227] examples/s
```