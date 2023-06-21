# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
from transformers import AutoModelForSequenceClassification, pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Modules
from kbqa.src.ftech.experiments.src.model import constants


class Evaluation:

    def __init__(self, model_path: str = constants.model_path, tokenizer=constants.tokenizer,
                 task: str = constants.task, test_path: str = constants.test_path,
                 x_label: str = constants.x_label, y_label: str = constants.y_label):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.task = task
        self.test_path = test_path
        self.x_label = x_label
        self.y_label = y_label
        self.texts, self.labels, self.y_pred = None, None, None

    def handle(self):
        """
        Dự đoán kết quả trên tập test 
        """
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except:
            return 
        classifier = pipeline(self.task, 
                              model=model,
                              tokenizer=self.tokenizer, 
                              device="cuda:0")
        testset = pd.read_csv(self.test_path)
        testset = testset.dropna()
        # testset = testset.drop_duplicates()
        testset[self.x_label] = testset[self.x_label].map(lambda x: x.lower())

        texts = list(testset[self.x_label])
        labels = list(testset[self.y_label])
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        results = classifier(texts, **tokenizer_kwargs)
        results = [result["label"] for result in results]

        return texts, labels, results

    def checking_set_attribute(self):
        """
        Kiểm tra xem đã dự đoán hay chưa
        """
        if not self.labels:
            self.texts, self.labels, self.y_pred = self.handle()

    def compute_accuracy(self):
        self.checking_set_attribute()
        return accuracy_score(y_true=self.labels, y_pred=self.y_pred, digits=4)

    def compute_precision(self):
        self.checking_set_attribute()
        return precision_score(y_true=self.labels, y_pred=self.y_pred, digits=4)

    def compute_recall(self):
        self.checking_set_attribute()
        return recall_score(y_true=self.labels, y_pred=self.y_pred, digits=4)

    def compute_f1(self):
        self.checking_set_attribute()
        return f1_score(y_true=self.labels, y_pred=self.y_pred, digits=4)

    def show_classfication_report(self, y_true=None, y_pred=None):
        self.checking_set_attribute()
        if not y_pred or not y_true:
            y_true=self.labels
            y_pred=self.y_pred
        
        print(classification_report(
            y_true=y_true, y_pred=y_pred, digits=4, labels = np.unique(y_true), zero_division = 1))

    def export_confusion_matrix(self, cfm_path, y_true=None, y_pred=None, classes = constants.classes):
        self.checking_set_attribute()
        if not y_pred or not y_true:
            y_true=self.labels
            y_pred=self.y_pred

        cfm = confusion_matrix(
            y_true=y_true, y_pred=y_pred, normalize='true', labels = classes)
        df_cfm = pd.DataFrame(
            cfm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        cfm_plot = sns.heatmap(df_cfm, annot=True)
        cfm_plot.figure.savefig(cfm_path)

    def export_pred_csv(self, csv_path: str, texts = None, y_true=None, y_pred=None):
        self.checking_set_attribute()
        if not y_pred or not y_true or not texts:
            y_true= self.labels
            y_pred= self.y_pred
            texts = self.texts
            
        out_df = pd.DataFrame({
            'Text': texts,
            'Label': y_true,
            'Predict': y_pred
        })
        out_df.to_csv(csv_path, index=False)
        
    def export_output(self, folder_path:str, texts = None, y_true=None, y_pred=None):
        if not y_pred or not y_true or not texts:
            y_true= self.labels
            y_pred= self.y_pred
            texts = self.texts
        
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"{folder_path}/out_{time_stamp}"
        os.mkdir(output_folder)
        self.export_pred_csv(csv_path=f"{output_folder}/out.csv",
                             texts = texts, 
                             y_true=y_true, 
                             y_pred=y_pred)
        self.export_confusion_matrix(cfm_path=f"{output_folder}/cmf.png",
                                     y_true=y_true, 
                                     y_pred=y_pred)
        
#---------------------------------------------------- FOR OUT OF DOMAIN ---------------------
        
    def convert_class(self, x):
        if x == 'oos':
            return 'oos'
        else:
            return 'in_domain'
    
    def show_classfication_report_oos(self, y_true=None, y_pred=None):
        self.checking_set_attribute()
        if not y_pred or not y_true:
            y_true=self.labels
            y_pred=self.y_pred
            
        y_true = list(map(self.convert_class, y_true))
        y_pred = list(map(self.convert_class, y_pred))
        
        print(classification_report(
            y_true=y_true, y_pred=y_pred, digits=4, labels=['in_domain', 'oos'], zero_division = 1))
        
    def export_confusion_matrix_oos(self, cfm_path, y_true=None, y_pred=None):
        self.checking_set_attribute()
        if not y_pred or not y_true:
            y_true=self.labels
            y_pred=self.y_pred
        y_true = list(map(self.convert_class, y_true))
        y_pred = list(map(self.convert_class, y_pred))

        cfm = confusion_matrix(
            y_true=y_true, y_pred=y_pred, normalize='true', labels = ['in_domain', 'oos'])
        df_cfm = pd.DataFrame(
            np.round(cfm, 4), index=['in_domain', 'oos'], columns=['in_domain', 'oos'])
        plt.figure(figsize=(10, 7))
        cfm_plot = sns.heatmap(df_cfm, annot=True)
        cfm_plot.figure.savefig(cfm_path)


if __name__ == '__main__':
    
    # print('lan 9')
    # evaluation = Evaluation(
    #     model_path='/home/annt/linh/timi/output/training_base/models/best_review',
    #     test_path='/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv'
    # )
    # evaluation.show_classfication_report()
    # evaluation.export_output(folder_path=constants.output_training_base_path)
    
    print('lan 10')
    evaluation = Evaluation(
        model_path='/home/annt/linh/timi/output/best',
        test_path='/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv'
    )
    evaluation.show_classfication_report()
    # evaluation.export_output(folder_path=constants.output_training_base_path)
    

