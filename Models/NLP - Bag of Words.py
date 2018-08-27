# Natural Language Processing - Bag of Words Model

# importing the data
import pandas as pd
reviews = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

reviews_corpus = []
for i in range(0, 1000):
    # only keeping letters - replacing everything else by a blank space
    reviews_1 = re.sub('[^a-zA-Z]',' ', reviews['Review'][i])
    # converting to a lower case
    reviews_1 = reviews_1.lower()
    # removing the non-significant words (prepositions, conjunctions, pronoun etc.)
    # converting the significant words to stem version
    reviews_1 = reviews_1.split()
    reviews_1 = [ps.stem(word) for word in reviews_1 if not word in set(stopwords.words('english'))]
    # converting the review back to a line (single string)
    reviews_1 = ' '.join(reviews_1)
    # appending to the corpus
    reviews_corpus.append(reviews_1)
    
# creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
review_sparseMat = cv.fit_transform(reviews_corpus).toarray()

# Now building the classificaton models on sparse matrix
# sparse matrix made above acts as the x-variable
y_var = reviews.iloc[:, -1].values

# using naive bayes model
# splitting the data into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(review_sparseMat, y_var,
                                                    test_size = 0.20,
                                                    random_state = 0)
from sklearn.naive_bayes import GaussianNB
reviews_nb = GaussianNB()
reviews_nb.fit(x_train, y_train)

# making predictions
reviews_pred = reviews_nb.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
reviews_confMat = confusion_matrix(y_test, reviews_pred)