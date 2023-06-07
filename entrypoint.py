import argparse
import json
import os
import sys
import more_itertools
import torch
from typing import List
import transformers
from datasets import Dataset
import time
from subprocess import SubprocessError
import datasets
from raft_baselines import classifiers
from sacred import Experiment

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# We provide this
def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    try:
        for line in sys.stdin:
            line = line.rstrip()
            inputs = json.loads(line)
            assert isinstance(inputs, list)

            # Participants need to connect their inference code 
            # to our wrapper through the following line.
            outputs = predictor.predict(inputs=inputs)
            outputs = list(outputs)


            # Writes are \n deliminated, so adding \n is essential 
            # to separate this write from the next loop iteration.
            sys.stdout.write(f"{json.dumps(outputs)}\n")
            # Writes to stdout are buffered. 
            # The flush ensures the output is immediately sent through 
            # the pipe instead of buffered.
            sys.stdout.flush()
    except:
        sys.stdout.write("Efficiency benchmark exception: SubprocessError\n")
        sys.stdout.flush()
        raise SubprocessError


def offline_predictor_wrapper(predictor):
    try:
        configs = sys.stdin.readline().rstrip()
        configs = json.loads(configs)
        assert isinstance(configs, dict)

        offline_dataset = Dataset.from_json(configs["offline_data_path"])
        offline_dataset_inputs = [instance["input"] for instance in offline_dataset]
        predictor.prepare()
        sys.stdout.write("Model and data loaded. Start the timer.\n")
        sys.stdout.flush()
        
        limit = configs.get("limit", None)
        if limit is not None and limit > 0:
            offline_dataset_inputs = offline_dataset_inputs[:limit]
        outputs = predictor.predict_offline(offline_dataset_inputs)
        outputs = list(outputs)
        sys.stdout.write("Offiline prediction done. Stop the timer.\n")
        sys.stdout.flush()

        outputs = Dataset.from_list([{"output": o} for o in outputs])
        outputs.to_json(configs["offline_output_path"])
        sys.stdout.write("Offiline outputs written. Exit.\n")
        sys.stdout.flush()
    except:
        sys.exit("Efficiency benchmark exception: SubprocessError")

# Submission
class Raft():
    def __init__(self, classifier_name: str, task: str, classifier_kwargs: str = None):
        raft_dataset = datasets.load_dataset("ought/raft", name=task)
        train_dataset = raft_dataset["train"]
        if classifier_name != "test":
            classifier_cls = getattr(classifiers, classifier_name)
            self.classifier = classifier_cls(train_dataset, **classifier_kwargs)
        self.classifier_name = classifier_name
        if not classifier_kwargs:
            classifier_kwargs = {}
        self.test_dataset = raft_dataset["test"]

    def predict(self, inputs: List[str]):
        if self.classifier_name == "test":
            for one_input in inputs:
                time.sleep(0.01)
                yield "1"
        else:
            for one_input in inputs:
                output_probs = self.classifier.classify(one_input)
                output, _ = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])
                # print(output_probs)
                yield output

    def predict_offline(self, inputs: List[str]):
        return self.predict(inputs)

experiment_name = "raft_efficiency_benchmark"
raft_experiment = Experiment(experiment_name, save_git_info=False)

@raft_experiment.automain
def main(classifier_name,
         classifier_kwargs: str = "",
         task="ade_corpus_v2",
         offline=False,
         debug=False,
         ):
    # We read outputs from stdout, and it is crucial to surpress unnecessary logging to stdout
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    transformers.logging.disable_progress_bar()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--classifier_name", type=str, default="AdaBoostClassifier", choices=["AdaBoostClassifier", "TransformersCausalLMClassifier", "TransformersZeroShotPipelineClassifier"])
    # parser.add_argument("--classifier_kwargs", type=str, default="{'':''}")
    # parser.add_argument("--task", type=str, default="ade_corpus_v2")
    # parser.add_argument("--offline", action="store_true")
    # args = parser.parse_args()
    # import ast
    # classifier_kwargs = ast.literal_eval(args.classifier_kwargs)
    if classifier_kwargs is not None and classifier_kwargs != "":
        classifier_kwargs["config"] = task
    predictor = Raft(classifier_name, task, classifier_kwargs)
    if debug:
        test_examples = predictor.test_dataset[0]
        print(test_examples)
        for output in predictor.predict([test_examples]):
            print(output)
    else:
        if offline:
            offline_predictor_wrapper(predictor)
        else:   
            stdio_predictor_wrapper(predictor)
