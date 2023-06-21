import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from kbqa.src.ftech.experiments.src.model import constants 
from kbqa.src.ftech.experiments.src.model.utils.evaluation import Evaluation

class Boosting:
    def __init__(self):
        pass
    
    def adaboost_fn(self, train_path = constants.train_path, test_path = constants.test_path):
        df_train = pd.read_csv(train_path)
        df_train = df_train.dropna()
        texts_train = df_train[constants.x_label].tolist()
        labels_train = df_train[constants.y_label].tolist()
        
        texts_train_encode = constants.tokenizer(texts_train, padding=True, truncation=True, max_length=512)['input_ids']
        labels_train_encode = [constants.label_mapping[label] for label in labels_train]
        
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(np.array([texts_train_encode]).reshape(-1,1), labels_train_encode)
        
        df_test = pd.read_csv(test_path)
        texts_test = df_test[constants.x_label].tolist()
        texts_test_encode = constants.tokenizer(texts_test, padding=True, truncation=True, max_length=512)['input_ids']
        
        y_preds = clf.predict(texts_test_encode)
        y_preds = [constants.idx_mapping[y_pred] for y_pred in y_preds] 
        
        return labels_train, y_preds
    

if __name__ == '__main__':
    boosting = Boosting()
    evaluation = Evaluation()
    
    y_true, y_pred = boosting.adaboost_fn(train_path='/home/annt/linh/timi/data/train/review/train_review_remove_entity.csv',
                                          test_path='/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv')
    
    evaluation.show_classfication_report(y_true=y_true, y_pred=y_pred)