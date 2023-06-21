# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from abc import ABC
import numpy as np
import pandas as pd

# Modules
from kbqa.src.ftech.experiments.src.model.utils.evaluation import Evaluation
from kbqa.src.ftech.experiments.src.model.training_args import TrainingArgs
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.data.data_processing import DataProcessing


class EnsempleLearning(ABC):
    args = TrainingArgs()
    evaluation = Evaluation()
    data = DataProcessing(path=constants.train_path)
    training_arguments = TrainingArguments(**args.to_dict())
    classifier_config = AutoConfig.from_pretrained(constants.model_name,
                                                   from_tf=False,
                                                   num_labels=len(
                                                       constants.label_mapping),
                                                   output_hidden_states=False,
                                                   id2label=constants.idx_mapping,
                                                   label2id=constants.label_mapping)

    def __init__(self, args=args, data=data, evaluation=evaluation,
                 training_arguments=training_arguments, classifier_config=classifier_config,
                 train_path=constants.train_path, test_path=constants.test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.args = args
        self.data = data
        self.train_dataset, self.test_dataset = data.handle_data(segment=False)
        self.evaluation = evaluation
        self.training_arguments = training_arguments
        self.classifier_config = classifier_config

    def compute_metrics(self, p):
        """
        Phương thức đánh giá cho trainer huggingface
        """
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

        return {
            "accuracy": accuracy,
            "f1": f1
        }

    def trainer_init(self, model_init, train_dataset, eval_dataset):
        """
        Khởi tạo đối tượng trainer với model_init
        """
        return Trainer(
            model_init=model_init,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

    def trainer(self, model, train_dataset, eval_dataset):
        """
        Khởi tạo đối tượng trainer với model
        """
        return Trainer(
            model=model,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(constants.model_name,
                                                                  config=self.classifier_config)

    def model_init_with_name(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                  config=self.classifier_config,
                                                                  ignore_mismatched_sizes=True
                                                                  )

    def partial_model_init(self, model_name):
        partial_model_init = partial(self.model_init_with_name, model_name=model_name)
        return partial_model_init
