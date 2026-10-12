import argparse
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ydf

parser = argparse.ArgumentParser()
# Path to training dataset. Should be prefixed with the dataset type e.g. 'csv':.
# See the supported formats at https://ydf.readthedocs.io/en/latest/dataset_formats/
parser.add_argument("--train_ds", type=str, required=True)
# Path to validation dataset. If empty, the model is trained without validation.
parser.add_argument("--valid_ds", type=str)
# Path to test dataset. If empty, the model is not evaluated.
parser.add_argument("--test_ds", type=str)
# Path to save the model.
parser.add_argument("--model", type=str, required=True)
# Work directory containing the temporary working data. Only used for distributed training.
parser.add_argument("--work_dir", default="", type=str)
# Label column to predict.
parser.add_argument("--label", type=str, required=True)
# Is the training distributed, or on a single machine?
parser.add_argument("--distributed", action="store_true")
args = parser.parse_args()


def main():
  print("Arguments:\n", args)

  if args.distributed:
    main_distributed()
  else:
    main_in_process()


def main_in_process():
  ydf.verbose(2)

  print("Train model in process on", args.train_ds)
  learner = ydf.GradientBoostedTreesLearner(label=args.label)
  model = learner.train(args.train_ds, valid=args.valid_ds)

  print("Save model in", args.model)
  model.save(args.model)

  if args.test_ds:
    print("Evaluate model on", args.test_ds)
    evaluation = model.evaluate(args.test_ds)
    print(evaluation)


def main_distributed():
  ydf.verbose(2)

  # Gather the manager and workers configuration.
  cluster_config = ydf.util.get_vertex_ai_cluster_spec()
  print("cluster_config:\n", cluster_config)

  if cluster_config.is_worker:
    # This machine is running a worker.
    ydf.start_worker(cluster_config.port)
    return

  print("Train model with distribution on", args.train_ds)
  learner = ydf.DistributedGradientBoostedTreesLearner(
      label=args.label,
      workers=cluster_config.workers,
      working_dir=args.work_dir,
      resume_training=True,
  )
  model = learner.train(args.train_ds, valid=args.valid_ds)

  print("Save model in", args.model)
  model.save(args.model)

  if args.test_ds:
    print("Evaluate model on", args.test_ds)
    evaluation = model.evaluate(args.test_ds)
    print(evaluation)


if __name__ == "__main__":
  main()
