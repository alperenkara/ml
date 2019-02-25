# 27.12.2018
# Natural Language Processing
# First step is data cleaning
# Second bag of words
# Third text classification
# Dataset : text gender classification 
# nltk, natural language tool kit

import pandas as pd
import sys
# reading the dataset
data = pd.read_csv(r"D:\mygit\ml\NaturalLanguageProcessing\gender-classifier.csv",encoding = "latin1")
# data = pd.read_csv("./gender-classifier.csv",encoding = "latin1")
# concanetane two data table according to column
data = pd.concat([data.gender,data.description],axis=1)
# drop NaN values from rows 
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender] 

# data cleaning 
import re 

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ", first_description)

description = description.lower()

import nltk
nltk.download("stopwords")
nltk.download('punkt')
# stopwords is going to save in corpus folder
from nltk.corpus import stopwords
# splitting the words according to spaces
# description = description.split()

# another way of the splint, tokenizer 
description = nltk.word_tokenize(description)

# cleaning unnecessary words

description = [word for word in description if not word in set(stopwords.words("english"))]