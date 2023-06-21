#
import os

# Machine learning and NLP libraries
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "microsoft/infoxlm-base" 
# model_name = "PaddlePaddle/ernie-m-base"
# model_name = "microsoft/mdeberta-v3-base"
task = 'sentiment-analysis'
x_label = 'Text'
y_label = 'Label'

# Question type constants

# label_mapping = {
#     "ASK_ROLE": 0,
#     "ASK_PERSON": 1,
#     "ASK_DATETIME": 2,
#     "ASK_LOCATION": 3,
#     "ASK_QUANTITY": 4,
#     "ASK_DEPARTMENT": 5,
#     "ASK_ORGANIZATION": 6
# }
# idx_mapping = {
#     0: "ASK_ROLE",
#     1: "ASK_PERSON",
#     2: "ASK_DATETIME",
#     3: "ASK_LOCATION",
#     4: "ASK_QUANTITY",
#     5: "ASK_DEPARTMENT",
#     6: "ASK_ORGANIZATION"
# }
# classes = ["ASK_DATETIME", "ASK_DEPARTMENT", "ASK_LOCATION",
#            "ASK_ORGANIZATION", "ASK_PERSON", "ASK_QUANTITY", "ASK_ROLE"]

# init function
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Dataset path
# train_path = "/home/annt/linh/question_type/data/processed/train_v207_review_person.csv"
# test_path = '/home/annt/linh/question_type/data/processed/test_v203_relabel.csv'
# output_path = '/home/annt/linh/question_type/src/model/output/out.csv'


# model_path = '/home/annt/linh/question_type/src/model/output/best'
# vncorenlp_path = '/home/annt/linh/question_type/src/model/modules/vncorenlp'
# output_dir = '/home/annt/linh/question_type/src/model/output/QuestionTypeClassification'
# cfm_path = "/home/annt/linh/question_type/src/model/output/cfm.png"

# # Ensemble learning
# base_models_folder = '/home/annt/linh/kbqa/kbqa/src/ftech/experiments/src/model/ensemble/my_model/base_models'
# meta_model_folder = '/home/annt/linh/kbqa/kbqa/src/ftech/experiments/src/model/ensemble/my_model/meta_model'
# out_stacking_voting = '/home/annt/linh/question_type/src/model/output/out_voting.csv'

#----------------------------------------------------------------------------------------------------------
# Timi constants

tokenizer_kwargs={"max_length": 512, "truncation": True}

label_mapping = {
    "company_ask_info_base": 0,
    "company_ask_office": 1,
    "company_ask_policy": 2,
    "company_ask_product": 3,
    "company_ask_recruitment": 4,
    "company_ask_rule": 5,
    "company_ask_time_keeper": 6,
    "knowledge_ask_person_info": 7,
    "oos":8
}
idx_mapping = {
    0: "company_ask_info_base",
    1: "company_ask_office",
    2: "company_ask_policy",
    3: "company_ask_product",
    4: "company_ask_recruitment",
    5: "company_ask_rule",
    6: "company_ask_time_keeper",
    7: "knowledge_ask_person_info",
    8: "oos"
}
classes = [
    "company_ask_info_base", "company_ask_office", "company_ask_policy",
    "company_ask_product", "company_ask_recruitment", "company_ask_rule",
    "company_ask_time_keeper", "knowledge_ask_person_info", "oos"
]

# init function
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dataset path
train_path = "/home/annt/linh/timi/data/train/review/train_review_remove_entity.csv"
test_path = '/home/annt/linh/timi/data/test/raw/test_v10_subset_label.csv'


model_path = '/home/annt/linh/timi/output/best'                                                         
vncorenlp_path = '/home/annt/linh/question_type/src/model/modules/vncorenlp'
# cfm_path = "/home/annt/linh/timi/output/cfm.png"
output_dir = '/home/annt/linh/timi/output/save_model_log'

# output 
output_training_base_path = '/home/annt/linh/timi/output/training_base/out'

output_kfold_path = '/home/annt/linh/timi/output/training_k_fold/out'

output_skfold_path = '/home/annt/linh/timi/output/training_s_k_fold/out'
output_voting_path = '/home/annt/linh/timi/output/voting/out'
output_stacking_path = '/home/annt/linh/timi/output/stacking/out'

# Ensemble learning
base_models_folder = '/home/annt/linh/timi/output/stacking/base_models'
meta_model_folder = '/home/annt/linh/timi/output/stacking/meta_model'
out_stacking_voting = '/home/annt/linh/timi/output/stacking/out_voting.csv'