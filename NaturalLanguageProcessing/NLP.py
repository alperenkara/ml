# 27.12.2018
# Natural Language Processing
# First step is data cleaning
# Second bag of words
# Third text classification
# Dataset : text gender classification 
# nltk, natural language tool kit

import pandas as pd
import sys

data = pd.read_csv(r"D:\mygit\ml\NaturalLanguageProcessing\gender-classifier.csv",encoding = "latin1")
# data = pd.read_csv("./gender-classifier.csv",encoding = "latin1")
