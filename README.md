# SMS Spam Classification Using a Naive Bayes Classifier

## Dependencies
The program is written in Python 3.6.1, but it should work for any version of Python 3.
See the requirements.txt file for the required libraries. If you have pip installed, you can install
all the required libraries by typing the following command in the command-line:
```bash
$ pip install -r requirements.txt
```

## Description of the files
* __training.py__: generates the model by training a naive bayes classifier.
* __main.py__: classifies a text message into spam/not spam using the model learned by training.py.
* __spam.csv__: the dataset used to train the classifier.

## Generate the model
The first step is to generate the model (which consists of training 
a naive bayes classifier). For that, you need to execute the file training.py:
```bash
$ python training.py
```
This will create two files: __clf.pkl__ and __vectorizer.pkl__.

## Classification of a new text message
The main.py file is used for that:
```bash
$ python main.py path/to/textfile
```

## References
The dataset (the file __spam.csv__) was obtained from [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

For more details on how the model (multinomial naive bayes) was chosen, check out my kaggle kernel [here](https://www.kaggle.com/devghiles/step-by-step-solution-with-f1-score-as-a-metric).
