# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
from transformers import AutoModelForSequenceClassification
import pandas as pd

# Modules
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.model.training.training import Training


class TrainingStratifiedKfold(Training):
    def __init__(self):
        super().__init__()

    def train(self):
        """
        Sử dụng stratified kfold để training
        """
        df = pd.read_csv(constants.train_path)
        df = df.dropna()
        df = df.drop_duplicates()
        texts = df[constants.x_label].tolist()
        labels = df[constants.y_label].tolist()
        i = 1

        trainer = self.trainer_init(model_init=self.model_init, 
                                    train_dataset=None, 
                                    test_dataset=None)
        trainer.save_model(constants.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(constants.model_path, 
                                                                   config=self.classifier_config)

        for train, test in constants.skf.split(texts, labels):
            print(f"------------------- Loop {i} ----------------")
            i += 1
            train_dataset, test_dataset = self.handing_data(train, test, df)
            trainer = self.trainer(model=model, 
                                   train_dataset=train_dataset, 
                                   test_dataset=test_dataset)
            trainer.train()

        trainer.save_model(constants.model_path)
        constants.tokenizer.save_pretrained(constants.model_path)


if __name__ == '__main__':
    training_obj = TrainingStratifiedKfold()
    training_obj.train()
