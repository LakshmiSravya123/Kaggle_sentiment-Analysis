
import pickle
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import VotingClassifier

#dataset
with open('/content/drive/My Drive/Colab Notebooks/data_train.pkl', 'rb') as f:
   train_set = pickle.load(f)
X_train=train_set[0]
y_train=train_set[1]
with open('/content/drive/My Drive/Colab Notebooks/data_test.pkl', 'rb') as f:
   test_set = pickle.load(f)
X_test=train_set[0]

#preprocessing removing stopwords,using word and character vectoriser
vectorizer = FeatureUnion([
('word_vectorizer',  TfidfVectorizer(
stop_words = 'english',
analyzer='word',
)),

('char_vectorizer', TfidfVectorizer(
stop_words = 'english',
analyzer='char',
ngram_range=(2, 8)
))
 ])
vectorizer.fit(train_set[0])
train_features = vectorizer.transform(train_set[0])
test_features = vectorizer.transform(test_set)

#svm classifier 
clf_svm=SVC(C=0.2501,tol=0.0001,gamma='auto',kernel='rbf').fit(train_features, y_train)
prediction_svm=clf_svm.predict(test_features)
#Accuracy of 57.3

#voting classifier ....faster approach without sklearn
#https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/-(logic taken)
predictions_voting=[]
for index in range(len(prediction_svm)):
#can add as many classifiers in the list (using only svm)
  a=[prediction_svm[index]]
  b=a.count
  predictions_voting.append(max(set(a), key=b))
#voting classifier along with svm gives accuracy of 59.7 

#Multinomial naive bayes with laplace smoothing with alpha value of alpha=0.2305010111
clf_naive = MultinomialNB(alpha=0.2305010111).fit(train_features, y_train)
predicted_naive= clf_naive.predict(test_features)
#gives accuracy of 58.4

#logistic regression
scikit_log_reg = LogisticRegression(verbose=1, solver='saga',random_state=0, C=5, penalty='l2',max_iter=3000).fit(train_features, y_train)
predictions_log=scikit_log_reg.predict(test_features)
#gives accuracy of 55.7

#RandomForestClassifier
rf = RandomForestClassifier(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=80, bootstrap=True ).fit(train_features, y_train)
predictions_random=rf.predict(test_features)
#gives accuracy of 47.8

