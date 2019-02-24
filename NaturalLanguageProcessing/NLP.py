# 27.12.2018
# Alperen Kara
# Natural Language Processing
# First step is data cleaning
# Second bag of words
# Third text classification
# Dataset : text gender classification 
# nltk, natural language tool kit

import pandas as pd
import sys

# data = pd.read_csv(r"D:\mygit\ml\NaturalLanguageProcessing\gender-classifier.csv",encoding = "latin1")
data = pd.read_csv("./gender-classifier.csv",encoding = "latin1")
# concat : combining two data table
data = pd.concat([data.gender,data.description],axis=1)
# axix = 0, drop from rows 
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender ]
# 24.02.2018
# pattern search
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description) # change with space apart from a-Z 
description = description.lower() # make lower case

import nltk # natural language processing library
nltk.download("stopwords") # downloading to folder of corpus
nltk.download("punkt")
nltk.download('wordnet')
from nltk.corpus import stopwords
# make words list from descriptions
#description = description.split()

# we are able to use tokenzier also 
# it provides us to split "not" words
description = nltk.word_tokenize(description)
# erase unnecassary words
description = [word for word in description if not word in set(stopwords.words("english"))]

import nltk as nlp 

lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
# merge the words and create a new sentence
description = " ".join(description)


for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description) # change with space apart from a-Z 
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]

    
