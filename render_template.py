#!/usr/bin/env python

from __future__ import print_function

import datetime
import jinja2
import os
import sys

import argparse

parser = argparse.ArgumentParser(description="Creates GKE Job specs for training.", allow_abbrev=False)


parser.add_argument(
    "--num_workers",
    dest="num_workers",
    type=int,
    default=1,
    help="Number of workers")
parser.add_argument(
    "--num_gpus_per_worker",
    help="Number of gpus per worker",
    required=False,
    type=int,
    default=4)
parser.add_argument(
    "--num_ps",
    dest="num_ps",
    type=int,
    default=0,
    help="Number of workers")
parser.add_argument(
    "--name", help="Name of the run", required=False, type=str, default="")
parser.add_argument(
    "--image", help="Docker Image to run.", required=True, type=str)
parser.add_argument(
    "--gcs_input_path",
    help="GCS path for the input Data.",
    required=False,
    type=str)
parser.add_argument(
    "--template_args",
    action="append",
    type=str)
parser.add_argument(
    "--gcs_output_path",
    help="GCS path for the output Data.",
    required=False,
    type=str)
parser.add_argument(
    "--has_eval",
    help="Whether to start eval job.",
    required=False,
    type=bool,
    default=False)
parser.add_argument(
    "--has_tensorboard",
    help="Whether to start tensorboard.",
    required=False,
    type=bool,
    default=False)


def main():
  args, extras = parser.parse_known_args()
  args = (vars(args))
  if not args["name"]:
    args["name"] = "run-" + datetime.datetime.now().strftime("%Y-%M-%d-%H-%M-%S")
  if args["gcs_output_path"]:
    args["gcs_output_path"] = os.path.join(args["gcs_output_path"], args["name"])
  args["cmdline_arg_list"] = extras[1:]
  with open(sys.argv[1], "r") as f:
    print(jinja2.Template(f.read()).render(args))


if __name__ == "__main__":
  main()
