import os
# Modules
from kbqa.src.ftech.experiments.src.model import constants

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainingArgs:

    def __init__(self,
                 output_dir=constants.output_dir,
                 num_train_epochs=0.3,
                 per_device_train_batch_size=32,
                 per_device_eval_batch_size=32,
                 warmup_steps=200,
                 learning_rate=3e-5,
                 weight_decay=7e-5,
                 logging_dir='./logs',
                 logging_steps=100,
                 evaluation_strategy='steps',
                 save_strategy='steps',
                 load_best_model_at_end=True,
                 gradient_accumulation_steps=1,
                 seed=42,
                 metric_for_best_model='eval_f1',
                 save_total_limit=1):

        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.metric_for_best_model = metric_for_best_model
        self.save_total_limit = save_total_limit

    def to_dict(self):
        return self.__dict__


if __name__ == '__main__':
    args = TrainingArgs()
    # print(args.to_dict(*x, **y))
