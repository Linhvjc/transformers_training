from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
import pandas as pd
import numpy as np
from scipy.stats import mode

from kbqa.src.ftech.experiments.src.model import constants 
from kbqa.src.ftech.experiments.src.model.utils.evaluation import Evaluation

class Voting:
    
    def __init__(self):
        pass
    
    def inference(self, models: list, test_path = constants.test_path):
        df = pd.read_csv(test_path)
        df = df.dropna()
        texts = df[constants.x_label].tolist()
        labels = df[constants.y_label].tolist()
        
        features = []
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        
        for i, model_path in enumerate(models):
            print('*****************************')
            print(f"********* Model {i + 1} ***********")
            print('*****************************')
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            classifier = pipeline(constants.task,
                                  model=model,
                                  tokenizer=tokenizer, 
                                  device="cuda:0")
            results = classifier(texts, **tokenizer_kwargs)
            results = [result["label"] for result in results]
            results = [constants.label_mapping[result] for result in results]
            
            features.append(results)
        
        features = np.array(features, dtype=np.int32).transpose()
        y_pred = mode(features, axis=1)[0].reshape(-1)
        y_pred = [constants.idx_mapping[item] for item in y_pred]
        
        return labels, y_pred
        
if __name__ == '__main__':
    voting = Voting()
    evaluation = Evaluation()
    
    models = [
        '/home/annt/linh/timi/output/training_base/models/best_mdeberta',
        '/home/annt/linh/timi/output/training_base/models/best_review',
        '/home/annt/linh/timi/output/training_base/models/best_validation',
        '/home/annt/linh/timi/output/training_k_fold/models/best',
    ]
    
    print('v9')
    y_true, y_pred = voting.inference(models = models, test_path='/home/annt/linh/timi/data/test/raw/test_v9_subset_label.csv')
    evaluation.show_classfication_report(y_true=y_true, y_pred=y_pred)
    evaluation.export_output(folder_path=constants.output_voting_path, y_pred=y_pred, y_true=y_true)
    
    print('v10')
    y_true, y_pred = voting.inference(models = models, test_path='/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv')
    evaluation.show_classfication_report(y_true=y_true, y_pred=y_pred)
    evaluation.export_output(folder_path=constants.output_voting_path, y_pred=y_pred, y_true=y_true)