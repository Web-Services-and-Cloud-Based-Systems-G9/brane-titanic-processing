#!/usr/bin/env python3
import os
import sys
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple
import pandas as pd
import numpy as np
import uuid


def read_datasets(train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test


def write_datasets(train: pd.DataFrame, test: pd.DataFrame, train_fn: str, test_fn: str) -> Tuple[str, str]:
    output_train, output_test = get_file_names(train_fn, test_fn)
    train.to_csv(output_train, index=False)
    test.to_csv(output_test, index=False)
    return output_train, output_test


def get_file_names(train: str, test: str) -> Tuple[str, str]:
    prefix = "/data/"
    if "TESTING" in os.environ and os.environ['TESTING'] == "1":
        prefix = ""
    new_train_fn = prefix + str(uuid.uuid4()) + ".csv"
    new_test_fn = prefix + str(uuid.uuid4()) + ".csv"
    return new_train_fn, new_test_fn


def drop_unuseful_columns(train_file: str, test_file: str, unuseful_columns: List[str]) -> Tuple[str, str]:
    train, test = read_datasets(train_file, test_file)
    train = train.drop(unuseful_columns, axis=1)
    test = test.drop(unuseful_columns, axis=1)
    output_train, output_test = write_datasets(train, test, train_file, test_file)
    return output_train, output_test


def transform_fields(train_file: str, test_file: str, fields_to_transform: List[str]) -> Tuple[str, str]:
    train, test = read_datasets(train_file, test_file)
    if "Age" in fields_to_transform:
        index_nan_age_test = list(test["Age"][test["Age"].isnull()].index)
        for i in index_nan_age_test:
            age_pred = test["Age"][(
                        (test["SibSp"] == test.iloc[i]["SibSp"]) & (test["Parch"] == test.iloc[i]["Parch"]) & (
                            test["Pclass"] == test.iloc[i]["Pclass"]))].median()
            age_med = test["Age"].median()
            if not np.isnan(age_pred):
                test["Age"].iloc[i] = age_pred
            else:
                test["Age"].iloc[i] = age_med
        index_nan_age = list(train["Age"][train["Age"].isnull()].index)
        for i in index_nan_age:
            age_pred = train["Age"][(
                        (train["SibSp"] == train.iloc[i]["SibSp"]) & (train["Parch"] == train.iloc[i]["Parch"]) & (
                            train["Pclass"] == train.iloc[i]["Pclass"]))].median()
            age_med = train["Age"].median()
            if not np.isnan(age_pred):
                train["Age"].iloc[i] = age_pred
            else:
                train["Age"].iloc[i] = age_med
    if "Sex" in fields_to_transform:
        train["Sex"] = [1 if each == "male" else 0 for each in train["Sex"]]
        test["Sex"] = [1 if each == "male" else 0 for each in test["Sex"]]
    if "Embarked" in fields_to_transform:
        train["Embarked"] = train["Embarked"].fillna("C")
        train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        test["Embarked"] = test["Embarked"].fillna("C")
        test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    if "Fsize" in fields_to_transform:
        train["Fsize"] = train["SibSp"] + train["Parch"] + 1
        test["Fsize"] = test["SibSp"] + test["Parch"] + 1
    output_train, output_test = write_datasets(train, test, train_file, test_file)
    return output_train, output_test


def train_and_predict(train_file: str, test_file: str, field_to_predict: str, fields_to_use: List[str], algorithm: str) -> str:
    train, test = read_datasets(train_file, test_file)
    Y_train = train[field_to_predict]
    X_train = train.drop(field_to_predict, axis=1)
    if fields_to_use is None:
        X_test = test.copy()
    else:
        X_train = X_train[fields_to_use]
        X_test = test[fields_to_use].copy()
    if algorithm == "decision_tree":
        model_used = DecisionTreeClassifier()
    else:
        model_used = RandomForestClassifier(n_estimators=100)
    model_used.fit(X_train, Y_train)
    Y_pred = model_used.predict(X_test)
    acc_decision_tree = round(model_used.score(X_train, Y_train) * 100, 2)
    # print("Training Accuracy: {}%".format(acc_decision_tree))

    prefix = "/data/"
    if "TESTING" in os.environ and os.environ['TESTING'] == "1":
        prefix = ""
    prediction_file = prefix + str(uuid.uuid4()) + ".csv"
    with open(prediction_file, 'w') as f:
        f.write("%s\n" % acc_decision_tree)
        for item in Y_pred:
            f.write("%s\n" % item)
    return str(prediction_file)


def drop_unuseful_columns_wrapper():
    arg_train_file = os.environ["TRAIN_FILE"]
    arg_test_file = os.environ["TEST_FILE"]
    arg_unuseful_columns = os.environ["UNUSEFUL_COLUMNS"].split(',')
    output = drop_unuseful_columns(arg_train_file, arg_test_file, arg_unuseful_columns)
    output = ",".join(output)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


def transform_fields_wrapper():
    arg_train_file = os.environ["TRAIN_FILE"]
    arg_test_file = os.environ["TEST_FILE"]
    arg_fields_to_transform = os.environ["FIELDS_TO_TRANSFORM"].split(',')
    output = transform_fields(arg_train_file, arg_test_file, arg_fields_to_transform)
    output = ",".join(output)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


def train_and_predict_wrapper():
    arg_train_file = os.environ["TRAIN_FILE"]
    arg_test_file = os.environ["TEST_FILE"]
    arg_algorithm = os.environ["ALGORITHM"]
    arg_field_to_predict = os.environ["FIELD_TO_PREDICT"]
    arg_fields_to_use = os.environ["FIELDS_TO_USE"].split(',')
    output = train_and_predict(arg_train_file, arg_test_file, arg_field_to_predict, arg_fields_to_use, arg_algorithm)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "drop_unuseful_columns":
        drop_unuseful_columns_wrapper()

    elif command == "transform_fields":
        transform_fields_wrapper()

    elif command == "train_and_predict":
        train_and_predict_wrapper()
