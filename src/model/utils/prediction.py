# Installed libraries
from transformers import AutoModelForSequenceClassification, pipeline

# Modules
from kbqa.src.ftech.experiments.src.model import constants


class Prediction:

    def __init__(self, model_path: str = constants.model_path, tokenizer=constants.tokenizer,
                 task: str = constants.task, test_path: str = constants.test_path,
                 x_label: str = constants.x_label, y_label: str = constants.y_label):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.task = task
        self.test_path = test_path
        self.x_label = x_label
        self.y_label = y_label
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.classifier = pipeline(self.task, 
                                   model=self.model, 
                                   tokenizer=self.tokenizer, 
                                   device="cuda:0")

    def predict(self, input):
        output = self.classifier(input)
        print("Result: ", output)


if __name__ == '__main__':
    prediction = Prediction(
        model_path='/home/annt/linh/.linh/models/model_205/9430_no_segment_infoxlm'
    )
    while True:
        sentence = input("Input test: ")
        prediction.predict(sentence)
        print('--------------------------------------------')
