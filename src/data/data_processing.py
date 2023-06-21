# System Libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Installed libraries
import py_vncorenlp
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
import pandas as pd

# Modules
from kbqa.src.ftech.experiments.src.model import constants
from kbqa.src.ftech.experiments.src.data.custom_dataset import CustomDataset


class DataProcessing:

    def __init__(self, path: str = '', x_label: str = 'Text', y_label: str = 'Label'):
        self.path = path
        self.x_label = x_label
        self.y_label = y_label

    def __segment(self, sentence: str) -> str:
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=constants.vncorenlp_path)
        output = rdrsegmenter.word_segment(sentence)[0]
        rdrsegmenter.close()
        output = word_tokenize(output, format="text")
        return output

    def mapping_data(self, dataset):
        label_mapping = constants.label_mapping
        # dataset = dataset.dropna()
        texts = dataset[self.x_label]
        labels = dataset[self.y_label]
        labels = [label_mapping[label] for label in labels]
        return texts, labels

    def read_dataset(self, segment=False):
        dataset = pd.read_csv(self.path)
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        if segment:
            dataset[self.x_label] = dataset[self.x_label].apply(self.__segment)

        texts, labels = self.mapping_data(dataset)
        return texts, labels

    def encoding_data(self, 
                      train_texts: list, 
                      test_texts: list, 
                      train_labels: list, 
                      test_labels: list):

        tokenizer = constants.tokenizer
        train_encodings = tokenizer.batch_encode_plus(train_texts, truncation=True, padding=True, max_length = 512)
        test_encodings = tokenizer.batch_encode_plus(test_texts, truncation=True, padding=True, max_length = 512)

        train_dataset = CustomDataset(train_encodings, train_labels)
        test_dataset = CustomDataset(test_encodings, test_labels)

        return train_dataset, test_dataset

    def handle_data(self, segment=False):
        train_texts, train_labels = self.read_dataset(segment=segment)
        train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, 
                                                                              train_labels, 
                                                                              test_size=0.1, 
                                                                              random_state=42)

        train_dataset, test_dataset = self.encoding_data(list(train_texts), 
                                                         list(test_texts), 
                                                         list(train_labels), 
                                                         list(test_labels))
        return train_dataset, test_dataset


if __name__ == '__main__':
    print('Run ProcessData.py file')
