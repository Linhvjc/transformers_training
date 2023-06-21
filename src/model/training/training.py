# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification,
                          AutoConfig)
from abc import ABC, abstractmethod
import numpy as np

# Modules
from kbqa.src.ftech.experiments.src.model.training_args import TrainingArgs
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.model.utils.evaluation import Evaluation
from kbqa.src.ftech.experiments.src.data.data_processing import DataProcessing


class Training(ABC):
    # Initial object
    args = TrainingArgs()
    evaluation = Evaluation()
    data = DataProcessing(path=constants.train_path)
    training_arguments = TrainingArguments(**args.to_dict())
    classifier_config = AutoConfig.from_pretrained(constants.model_name,
                                                   from_tf=False,
                                                   num_labels=len(constants.label_mapping),
                                                   output_hidden_states=False,
                                                   id2label=constants.idx_mapping,
                                                   label2id=constants.label_mapping)

    def __init__(self, args=args, data=data, evaluation=evaluation,
                 training_arguments=training_arguments, classifier_config=classifier_config):
        self.args = args
        self.data = data
        self.train_dataset, self.test_dataset = data.handle_data(segment=False)
        self.evaluation = evaluation
        self.training_arguments = training_arguments
        self.classifier_config = classifier_config

    @abstractmethod
    def train(self):
        pass

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(constants.model_name,
                                                                  config=self.classifier_config)

    def model_init_with_name(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                  config=self.classifier_config)

    def partial_model_init(self, model_name):
        partial_model_init = partial(self.model_init_with_name, model_name=model_name)
        return partial_model_init

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

        return {
            "accuracy": accuracy,
            "f1": f1
        }

    def handing_data(self, train_index, test_index, df):
        """
        Xử lý dữ liệu text thành token
        """
        df_train = df.iloc[train_index, :]
        df_test = df.iloc[test_index, :]

        train_texts, train_labels = self.data.mapping_data(df_train)
        test_texts, test_labels = self.data.mapping_data(df_test)

        train_dataset, test_dataset = self.data.encoding_data(
            train_texts=list(train_texts),
            test_texts=list(test_texts),
            train_labels=list(train_labels),
            test_labels=list(test_labels)
        )
        return train_dataset, test_dataset

    def trainer_init(self, model_init, train_dataset, test_dataset):
        return Trainer(
            model_init=model_init,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

    def trainer(self, model, train_dataset, test_dataset):
        return Trainer(
            model=model,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

    def show_evaluation(self, cfm_path, csv_path):
        """
        Hiện kết quả classification
        """
        self.evaluation.show_classfication_report()
        self.evaluation.export_pred_csv(csv_path = csv_path)
        self.evaluation.export_confusion_matrix(cfm_path=cfm_path)


if __name__ == '__main__':
    print('Hello world')
