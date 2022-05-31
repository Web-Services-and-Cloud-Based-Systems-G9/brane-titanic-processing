# Brane Data Processing Package for the Titanic dataset

The following repository contains the implementation of a **data processing package in brane** for Data Science tasks on a Kaggle competition dataset related to the sinking of the Titanic. This project correspond to Assignment 4.b of the Web Services and Cloud Based Systems at University of Amsterdam (G9).

## The Package (`titanicprocessing`)
The package definition is inside the `container.yml` file in the root of this repository. The package includes the train and testing datasets in the root of the package. This package is composed of three methods that can be used as building blocks in any [BraneScript](https://wiki.enablingpersonalizedinterventions.nl/user-guide/branescript/introduction.html) pipeline. 
- Drop Unuseful Columns 
- Transform Features
- Train and Predict

### Drop Unuseful Columns 

- **Method Global Name**: `drop_unuseful_columns` 
- **Description**: This method loads the given `train_file` and `test_file` into two Pandas dataframes and drop the columns given in the `unuseful_columns` list and stores these dataframes into new .csv files. The method returns a list of two strings containing the file names of the both new .csv files. The new file names are generated using UUID4.
- **INPUT**: 
  - `train_file`(str): File name of the train data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use. 
  - `test_file`(str): File name of the test data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use.   
  - `unuseful_columns`(List[str]): List of the names of the columns to drop from both train and test files.
- **OUTPUT**:
  - `output`(List[str]): Names of the files that were generated with the output dataframes. First element is the new train set file name, and second element is the new test set file name.

### Transform Features
- **Method Global Name**: `transform_fields` 
- **Description**: This method loads the given `train_file` and `test_file` into two Pandas dataframes and transform the columns given in the `fields_to_transform` list and stores these dataframes into new .csv files. The method returns a list of two strings containing the file names of the both new .csv files. The new file names are generated using UUID4.
- **INPUT**: 
  - `train_file`(str): File name of the train data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use. 
  - `test_file`(str): File name of the test data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use.  
  - `fields_to_transform`(List[str]): List of the names of the columns to apply a transformation from both train and test files.
- **OUTPUT**:
  - `output`(List[str]): Names of the files that were generated with the output dataframes. First element is the new train set file name, and second element is the new test set file name.


### Train and Predict
- **Method Global Name**: `train_and_predict` 
- **Description**: This method loads the given `train_file` and `test_file` into two Pandas dataframes. Next, using the fields on `fields_to_use`, fits a Decision Tree Classifier to predict the values on `field_to_predict`. Finally, using the fitted model, predictions are made for the test set. The method returns the prediction accuracy scored on the training set (for now)
- **INPUT**: 
  - `train_file`(str): File name of the train data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use. 
  - `test_file`(str): File name of the test data to use. On first instance, you only have `'train.csv'` and `'test.csv'` to use.  
  - `fields_to_use`(List[str]): List of the names of the columns to use for the training.
  - `field_to_predict`(str): Name of the column with the field to use as response (i.e. the feature to predict).
- **OUTPUT**:
  - `output`(real): Prediction accuracy scored on the training set (for now)


## Unit Tests (locally)
Unit tests were implemented in [Pytest](https://docs.pytest.org/en/6.2.x/contents.html). There are three tests that needs to pass. Each test checks the correctness YAML output of each method. These methods check if the output have a correct format and execute son checks on the output data to verify that the operations have been done successfully. To execute locally you can execute the following commands:
1. Install pipenv `pip3 install pipenv`
2. Run the tests `pipenv run test`


## Building with Brane (locally)
1. [Install](https://onnovalkering.gitbook.io/brane/getting-started/installation) Brane CLI.
2. Run `brane build container.yml`

The repository is built in such a way that a `brane import` can also be done using the following command:  

```
brane import Web-Services-and-Cloud-Based-Systems-G9/brane-titanic-processing
```

## Automated Tests
This repository a **GitHub Action** workflows configured that runs automated tests to ensure that the methods of the packages work correctly (`.github/workflows/pytest.yml`). In addition to this, it makes sure that the package can be built in Brane successfully by implementing a test which tries to do the building process of the package (`.github/workflows/ci.yml`).

## Usage example
Once you have build the package with Brane, you can use the following examples to try the package. 

TODO


## DOI

TODO