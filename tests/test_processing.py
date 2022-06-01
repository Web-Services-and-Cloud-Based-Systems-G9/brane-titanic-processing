import os
import yaml
from main import train_and_predict_wrapper, transform_fields_wrapper, drop_unuseful_columns_wrapper
import pytest
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.csv')
TEST_PATH = os.path.join(ROOT_DIR, 'test.csv')


def test_processing_pipeline():
    os.environ["TESTING"] = "1"
    os.environ["TRAIN_FILE"] = TRAIN_PATH
    os.environ["TEST_FILE"] = TEST_PATH

    os.environ["UNUSEFUL_COLUMNS"] = "4"
    os.environ["UNUSEFUL_COLUMNS0"] = "Cabin"
    os.environ["UNUSEFUL_COLUMNS1"] = "Name"
    os.environ["UNUSEFUL_COLUMNS2"] = "Ticket"
    os.environ["UNUSEFUL_COLUMNS3"] = "PassengerId"

    os.environ["FIELDS_TO_TRANSFORM"] = "4"
    os.environ["FIELDS_TO_TRANSFORM0"] = "Age"
    os.environ["FIELDS_TO_TRANSFORM1"] = "Sex"
    os.environ["FIELDS_TO_TRANSFORM2"] = "Embarked"
    os.environ["FIELDS_TO_TRANSFORM3"] = "Fsize"

    os.environ["FIELD_TO_PREDICT"] = 'Survived'
    os.environ["ALGORITHM"] = 'decision_tree'
    os.environ["FIELDS_TO_USE"] = "6"
    os.environ["FIELDS_TO_USE0"] = "Pclass"
    os.environ["FIELDS_TO_USE1"] = "Sex"
    os.environ["FIELDS_TO_USE2"] = "Age"
    os.environ["FIELDS_TO_USE3"] = "Age" #"Fare"
    os.environ["FIELDS_TO_USE4"] = "Embarked"
    os.environ["FIELDS_TO_USE5"] = "Fsize"

    try:
        yaml_result = drop_unuseful_columns_wrapper()
        test_result = yaml.safe_load(yaml_result)
        print(test_result)
        os.environ["TRAIN_FILE"] = test_result["output"][0]
        os.environ["TEST_FILE"] = test_result["output"][1]

        yaml_result = transform_fields_wrapper()
        test_result = yaml.safe_load(yaml_result)
        print(test_result)
        os.environ["TRAIN_FILE"] = test_result["output"][0]
        os.environ["TEST_FILE"] = test_result["output"][1]

        yaml_result = train_and_predict_wrapper()
        test_result = yaml.safe_load(yaml_result)
        print(test_result)
        if "output" not in test_result and not isinstance(test_result["output"], float):
            assert False
        assert True
    except Exception as e:
        assert False


def test_transform_fields_wrapper():
    os.environ["TESTING"] = "1"
    os.environ["TRAIN_FILE"] = TRAIN_PATH
    os.environ["TEST_FILE"] = TEST_PATH
    os.environ["FIELDS_TO_TRANSFORM"] = "4"
    os.environ["FIELDS_TO_TRANSFORM0"] = "Age"
    os.environ["FIELDS_TO_TRANSFORM1"] = "Sex"
    os.environ["FIELDS_TO_TRANSFORM2"] = "Embarked"
    os.environ["FIELDS_TO_TRANSFORM3"] = "Fsize"
    try:
        yaml_result = transform_fields_wrapper()
        test_result = yaml.safe_load(yaml_result)
        print(test_result)
        if "output" in test_result and isinstance(test_result["output"], list) and len(test_result["output"]) == 2:
            new_train = pd.read_csv(test_result["output"][0])
            new_test = pd.read_csv(test_result["output"][1])
            assert (
                new_train["Age"].isna().sum() == 0 and
                new_test["Age"].isna().sum() == 0 and
                sum(new_train["Sex"].isin([1, 0])) != 0 and
                sum(new_test["Sex"].isin([1, 0])) != 0 and
                sum(new_train["Embarked"].isin([0, 1, 2])) != 0 and
                sum(new_test["Embarked"].isin([0, 1, 2])) != 0
            )
        else:
            assert False
    except Exception as e:
        assert False


def test_drop_unuseful_columns_wrapper():
    os.environ["TESTING"] = "1"
    os.environ["TRAIN_FILE"] = TRAIN_PATH
    os.environ["TEST_FILE"] = TEST_PATH
    os.environ["UNUSEFUL_COLUMNS"] = "4"
    os.environ["UNUSEFUL_COLUMNS0"] = "Cabin"
    os.environ["UNUSEFUL_COLUMNS1"] = "Name"
    os.environ["UNUSEFUL_COLUMNS2"] = "Ticket"
    os.environ["UNUSEFUL_COLUMNS3"] = "PassengerId"
    non_existent_columns = {"Cabin", "Name", "Tickets", "PassengerId"}
    try:
        yaml_result = drop_unuseful_columns_wrapper()
        test_result = yaml.safe_load(yaml_result)
        print(test_result)
        if "output" in test_result and isinstance(test_result["output"], list) and len(test_result["output"]) == 2:
            new_train = pd.read_csv(test_result["output"][0])
            new_test = pd.read_csv(test_result["output"][1])
            assert (
                    len(set(new_train.columns).intersection(non_existent_columns)) == 0 and
                    len(set(new_test.columns).intersection(non_existent_columns)) == 0
            )
        else:
            assert False
    except Exception as e:
        assert False
