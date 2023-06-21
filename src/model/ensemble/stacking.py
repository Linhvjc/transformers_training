# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Modules
from kbqa.src.ftech.experiments.src.data.custom_dataset import CustomDataset
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.model.ensemble.ensemble_learning import EnsempleLearning
from kbqa.src.ftech.experiments.src.model.utils.evaluation import Evaluation


class Stacking(EnsempleLearning):

    def __init__(self, 
                 base_models_folder=constants.base_models_folder, 
                 meta_model_folder=constants.meta_model_folder):
        super().__init__()
        self.base_models_folder = base_models_folder
        self.meta_model_folder = meta_model_folder

    def _handing_data(self, 
                      train_index: list, 
                      test_index: list, 
                      df: pd.DataFrame, 
                      tokenizer_name: str):
        """
        Xử lý dữ liệu train và test khi sử sử dụng kfold
        """
        df_train = df.iloc[train_index, :]
        df_test = df.iloc[test_index, :]

        train_texts, train_labels = self.data.mapping_data(df_train)
        test_texts, test_labels = self.data.mapping_data(df_test)

        train_texts, train_labels = list(train_texts), list(train_labels)
        test_texts, test_labels = list(test_texts), list(test_labels)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        train_encodings = tokenizer.batch_encode_plus(train_texts, truncation=True, padding=True)
        test_encodings = tokenizer.batch_encode_plus(test_texts, truncation=True, padding=True)

        train_dataset = CustomDataset(train_encodings, train_labels)
        test_dataset = CustomDataset(test_encodings, test_labels)

        return train_dataset, test_dataset

    def prediction_pretrain_model(self, test_texts: list, model_path: str):
        """
        Sử dụng các model pre-train để dự đoán và trả ra kết quả 
        """
        label_mapping = constants.label_mapping
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        classifier = pipeline(constants.task, 
                              model=model,
                              tokenizer=tokenizer, 
                              device="cuda:0")
        results = classifier(test_texts, **constants.tokenizer_kwargs)
        results = [result["label"] for result in results]
        results = [label_mapping[result] for result in results]

        return np.array(results, dtype=np.int32)

    def base_model_train(self, model_name: str, save_name: str, n_fold: int = 5):
        """
        Tiến hành training và dự đoán kết quả cho model base
        """
        print('********************************************************')
        print(f"{model_name}")
        print('********************************************************')

        folds = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
        df = pd.read_csv(constants.train_path)
        df = df.dropna()
        df = df.drop_duplicates()
        df[constants.x_label] = df[constants.x_label].apply(lambda x: x.lower())
        texts = df[constants.x_label].tolist()
        labels = df[constants.y_label].tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        trainer = self.trainer_init(model_init=self.model_init, 
                                    train_dataset=None, 
                                    eval_dataset=None)
        trainer.save_model(constants.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(constants.model_path, 
                                                                   config=self.classifier_config)

        for train, test in folds.split(texts, labels):
            train_dataset, test_dataset = self._handing_data(train, test, df, tokenizer_name=model_name)

            trainer = self.trainer(model=model, 
                                   train_dataset=train_dataset, 
                                   eval_dataset=test_dataset)
            trainer.train()

        trainer.save_model(f"{self.base_models_folder}/{save_name}")
        tokenizer.save_pretrained(f"{self.base_models_folder}/{save_name}")
        single_model_predict = self.prediction_pretrain_model(test_texts=texts,
                                                              model_path=f"{self.base_models_folder}/{save_name}")
        return single_model_predict

    def train_meta_model(self, meta_model, new_train: pd.DataFrame):
        """
        Tiến hành training meta-model dựa vào tập train mới đã tạo
        """
        y = new_train['Label'].to_numpy()
        X = new_train.drop(['Label'], axis=1).to_numpy()
        meta_model.fit(X, y)

    def predict_meta_model(self, base_models: list, meta_model, test_path: str = constants.test_path):
        """
        Dự đoán sử dụng meta-model
        """

        df = pd.read_csv(test_path)
        df = df.dropna()
        df = df.drop_duplicates()
        df[constants.x_label] = df[constants.x_label].map(lambda x: x.lower())
        
        X = df[constants.x_label].tolist()
        y_true = df[constants.y_label].tolist()
        features = []

        for model in base_models:
            feature = self.prediction_pretrain_model(test_texts=X, model_path=model)
            features.append(feature)

        features = np.array(features, dtype=np.int32).transpose()
        y_pred = meta_model.predict(features)

        y_pred = [constants.idx_mapping[item] for item in y_pred]
        return y_pred, y_true

    def predict_by_voting(self, base_models: list, test_path: str = constants.test_path):
        """
        Dự đoán dựa trên voting
        """
        df = pd.read_csv(test_path)
        df = df.dropna()
        df = df.drop_duplicates()
        df[constants.x_label] = df[constants.x_label].map(lambda x: x.lower())
        
        X = df[constants.x_label]
        y_true = df[constants.y_label]
        features = []

        model_name = []
        for model in base_models:
            model_name.append('model' + model[-1])
            feature = self.prediction_pretrain_model(test_texts=X, model_path=model)
            features.append(feature)

        features = np.array(features, dtype=np.int32).transpose()
        y_pred = mode(features, axis=1)[0].reshape(-1)

        column_predict = np.array(y_pred)
        column_label = np.array(y_true)
        matrix = np.append(features, column_label.reshape(-1, 1), axis=1)
        matrix = np.append(matrix, column_predict.reshape(-1, 1), axis=1)

        y_pred = [constants.idx_mapping[item] for item in y_pred]

        model_name.extend(['Label', 'Predict'])
        new_df = pd.DataFrame(matrix, columns=model_name)
        new_df.to_csv(constants.out_stacking_voting)
        return y_pred, y_true.tolist()

    def stacking(self, base_models: list, voting: bool = False):
        """
        Thực hiện quá trình stacking trên tập train và đưa ra kết quả dự đoán trên tập test
        """
        dict_df = {}
        base_models_path = []

        for i, model in enumerate(base_models):
            feature = self.base_model_train(model_name=model, save_name=f"model{i+1}")
            dict_df[f"model{i+1}"] = feature
            base_models_path.append(self.base_models_folder+f"/model{i+1}")

        df_train = pd.read_csv(self.train_path)
        df_train = df_train.dropna()
        df_train = df_train.drop_duplicates()
        train_labels = df_train[constants.y_label].tolist()
        train_labels = [constants.label_mapping[train_label] for train_label in train_labels]
        y_true = np.array(train_labels, dtype=np.int32)
        dict_df['Label'] = y_true
        df = pd.DataFrame(dict_df)
        df.to_csv('/home/annt/linh/timi/output/stacking/base_models/new_df.csv', index= False)

        if voting:
            y_test_pred, y_test_true = self.predict_by_voting(base_models=base_models_path)
        else:
            meta_model = DecisionTreeClassifier()
            self.train_meta_model(meta_model=meta_model, new_train=df)
            pickle.dump(meta_model, open(f"{self.meta_model_folder}/meta_model.pickle", "wb"))
            y_test_pred, y_test_true = self.predict_meta_model(base_models=base_models_path, 
                                                                   meta_model=meta_model)
        return y_test_pred, y_test_true

    def infer(self, base_models_path:list, meta_model_path,  test_path = constants.test_path):
        
        meta_model = pickle.load(open(meta_model_path, 'rb'))
        y_test_pred, y_test_true = self.predict_meta_model(base_models=base_models_path, 
                                                            meta_model=meta_model,
                                                            test_path = test_path)
        return y_test_pred, y_test_true


if __name__ == '__main__':
    stacking = Stacking()
    bases_models = ['bert-base-multilingual-cased',
                    'xlm-roberta-base',  
                    "microsoft/infoxlm-base"]
    y_pred, y_true = stacking.stacking(bases_models, voting=False)
    
    base_models_path = [
            '/home/annt/linh/timi/output/stacking/base_models/model1',
            '/home/annt/linh/timi/output/stacking/base_models/model2',
            '/home/annt/linh/timi/output/stacking/base_models/model3'
    ]
    
    test_path = '/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv'
    
    # y_pred, y_true = stacking.infer(
    #     test_path = test_path,
    #     base_models_path=base_models_path,
    #     meta_model_path = '/home/annt/linh/timi/output/stacking/meta_model/meta_model.pickle'
    #     )
    df = pd.read_csv(test_path)
    df = df.dropna()
    texts = df['Text'].tolist()
    
    evaluation = Evaluation()
    evaluation.show_classfication_report(y_true=y_true,y_pred=y_pred)
    evaluation.export_output(y_true=y_true, y_pred = y_pred, texts=texts, folder_path=constants.output_stacking_path)
    
    