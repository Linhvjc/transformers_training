# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
import pandas as pd

# Modules
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.model.training.training import Training
from kbqa.src.ftech.experiments.src.data.custom_dataset import CustomDataset


class TrainingBase(Training):

    def __init__(self):
        super().__init__()

    def train(self):
        """
        Thực hiện training mô hình huggingface
        """
        trainer = self.trainer_init(self.partial_model_init(constants.model_name),
                                    train_dataset=self.train_dataset,
                                    test_dataset=self.test_dataset)

        trainer.train()
        trainer.save_model(constants.model_path)
        constants.tokenizer.save_pretrained(constants.model_path)
        
    def train_with_val(self, val_path):
        df_val = pd.read_csv(val_path)
        df_val = df_val.dropna()
        df_val = df_val.drop_duplicates()
        val_texts = df_val['Text'].tolist()
        val_labels = df_val['Label'].tolist()
        val_labels = [constants.label_mapping[label] for label in val_labels]
        
        df_train = pd.read_csv(constants.train_path)
        df_train = df_train.dropna()
        df_train = df_train.drop_duplicates()
        train_texts = df_train['Text'].tolist()
        train_labels = df_train['Label'].tolist()
        train_labels = [constants.label_mapping[label] for label in train_labels]
        
        train_dataset, val_dataset = self.data.encoding_data(train_texts, 
                                                            val_texts, 
                                                            train_labels, 
                                                            val_labels)
        
        trainer = self.trainer_init(self.partial_model_init(constants.model_name),
                                    train_dataset=train_dataset,
                                    test_dataset=val_dataset)
        trainer.train()
        trainer.save_model(constants.model_path)
        constants.tokenizer.save_pretrained(constants.model_path)


if __name__ == '__main__':
    training_obj = TrainingBase()
    training_obj.train()
    # training_obj.train_with_val(val_path = '/home/annt/linh/timi/data/test/review/concat_v1_v9.csv')
