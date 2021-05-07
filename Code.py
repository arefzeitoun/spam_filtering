import pandas as pd                                             # For Data Exploration, Manipulation
import string                                                   # For punctuation 
import numpy as np                                              # To create arrays
import nltk                                                     # For Text Pre-processing                   
from nltk.corpus import stopwords                               # To remove stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize                         # Tokenize text into words
from nltk.stem import SnowballStemmer                           # Reducing word to it's root
from sklearn.feature_extraction.text import CountVectorizer     # Create Bag of Words
from sklearn.model_selection import train_test_split            # Split data into groups (Testing and Training)
from sklearn.naive_bayes import MultinomialNB                   # Selecting the Multinomial Algorithm 
from sklearn.metrics import accuracy_score                      # Display Accuracy 

# Opening Dataset 
# Make Sure Dataset is in the same directory as code file
df = pd.read_csv("emails.csv")

# Displaying random 5 rows
df.sample(5)

# Limiting DataFrame to first two columns 
df = df.iloc[:,0:2]

# Printing information about DataFrame
df.info()

# Sum of Null (Empty) Values by Column
df.isnull().sum()

# Drop null values 
df.dropna(inplace = True)     

# Drop duplicates
df.drop_duplicates(inplace = True)    

# We put all the stopwords into a set called "stop_words"
stop_words = set(stopwords.words("english")) 

# We write a function called "santizer" that takes a text string "dirty_text" and removes punctuation 

def sanitizer(dirty_text):
    clean_text = ""             # Initializing variable
    for char in dirty_text: 
        if char not in string.punctuation:         # Comparing if character in word is one of these !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
            clean_text += char.lower()             # Lowercase Character and add it to clean text
    return clean_text

df["Cleaned Text"] = df["text"].apply(sanitizer)         # Applying function "sanitizer" to all all text in column "text" and adding to new column "Cleaned Text"

# Now we display our updated dataframe
df.head()

# Using the Snowball Stemmer 
stemmer = SnowballStemmer('english')         
corpus = []                                  # Initializing corpus list 

# Building Corpus of Words 
for text in df["Cleaned Text"]:             
    tokens = word_tokenize(text)
    stem = ""
    for word in tokens:
        if word not in stop_words:               # Removing Stop Words
            stem += stemmer.stem(word) + ' '             
    corpus.append(stem) 

# Bag of Words/ Vectorization
vectorizer = CountVectorizer()                  
vectorizer.fit(corpus)                          
vector = vectorizer.transform(corpus)
x = vector.toarray()
y = df["spam"].values

# Train-Test Split = (80-20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Naive Bayes 
classifier = MultinomialNB()
classifier.fit(x_train , y_train)

# Use testing data to make predictions
y_pred = classifier.predict(x_test)

# Comparing prediction to actual values and measure accuracy
print("Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred)))
print("{} out of {} correctly predicted".format(accuracy_score(y_test, y_pred,normalize=False), len(y_test)))
