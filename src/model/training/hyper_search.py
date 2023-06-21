# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
import optuna
from transformers import (TrainingArguments, Trainer,
                          AutoModelForSequenceClassification)

# Modules
from kbqa.src.ftech.experiments.src.model.training_args import TrainingArgs
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.model.training.training import Training


class HyperparameterSearch(Training):
    def __init__(self):
        super().__init__()

    def objective(self, trial: optuna.Trial):
        """
        Định nghĩa những giá trị mà chúng ta cần tìm kiếm hyperparameter tối ưu
        """

        model = AutoModelForSequenceClassification.from_pretrained(constants.model_name,
                                                                   config=self.classifier_config)

        search_args = {
            "output_dir": constants.output_dir,
            'learning_rate': trial.suggest_loguniform('learning_rate', low=1e-7, high=1e-6),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-7, 0.01),
            'num_train_epochs': trial.suggest_int('num_train_epochs', low=10, high=200, step=5),
            'warmup_steps': trial.suggest_int('warmup_steps', low=100, high=1000, step=10),
            'per_device_train_batch_size': 640,
            'per_device_eval_batch_size': 640
        }

        args_search = TrainingArgs(**search_args)
        training_args = TrainingArguments(**args_search.to_dict())

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics
        )

        eval_result = trainer.evaluate()
        return eval_result['eval_f1']

    def searching(self):
        """
        Tiến hành tìm kiếm hyperparameter tốt nhất
        """
        print('Triggering Optuna study')
        study = optuna.create_study(study_name='hp-search-electra', 
                                    direction='minimize')
        study.optimize(func=self.objective, n_trials=10)
        return study

    def train(self):
        """
        Tiến hành training với best model đã được tìm thấy
        """
        study = self.searching()
        best_lr = float(study.best_params['learning_rate'])
        best_weight_decay = float(study.best_params['weight_decay'])
        best_epoch = int(study.best_params['num_train_epochs'])
        best_warmup_steps = float(study.best_params['warmup_steps'])

        best_dict = {
            'learning_rate': best_lr,
            'weight_decay': best_weight_decay,
            'num_train_epochs': best_epoch,
            'warmup_steps': best_warmup_steps
        }
        self.args = TrainingArgs(**best_dict)
        self.training_arguments = TrainingArguments(**self.args.to_dict())

        trainer = self.trainer_init(model_init=self.partial_model_init(model_name=constants.model_name), 
                                    train_dataset=self.train_dataset, 
                                    test_dataset=self.test_dataset)

        trainer.train()
        trainer.save_model(constants.model_path)
        constants.tokenizer.save_pretrained(constants.model_path)


if __name__ == '__main__':
    training = HyperparameterSearch()
    training.train()
