import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import pickle

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the datasets
train_df = pd.read_csv('reddit/reddit_training.csv')
test_df = pd.read_csv('reddit/reddit_test.csv')

# For development, let's use a smaller subset of the data
train_df = train_df.sample(frac=1, random_state=42)
test_df = test_df.sample(frac=0.3, random_state=42)

# Define preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return ' '.join(lemmatized_text)

# Preprocess text data
train_df['processed_text'] = train_df['body'].apply(preprocess_text)
test_df['processed_text'] = test_df['body'].apply(preprocess_text)

# Extract contextual information (e.g., parent comment)
def extract_context_features(df, parent_col='parent_id', text_col='processed_text'):
    context_dict = df.set_index('id')[text_col].to_dict()
    df['context_text'] = df[parent_col].map(context_dict).fillna('')
    df['full_text'] = df[text_col] + ' ' + df['context_text']
    return df

train_df = extract_context_features(train_df)
test_df = extract_context_features(test_df)

# Define a feature extractor class for lexical, pragmatic, and incongruity features
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            feature_set = []
            words = text.split()
            
            # Lexical features
            feature_set.append(len(words))
            feature_set.append(sum([len(word) for word in words]) / len(words) if words else 0)

            # Pragmatic features
            feature_set.append(text.count('!'))
            feature_set.append(text.count('?'))
            
            # Explicit incongruity features
            positive_words = ['love', 'great', 'good', 'awesome', 'best', 'amazing']
            negative_words = ['hate', 'bad', 'terrible', 'worst', 'awful', 'horrible']
            pos_count = sum([1 for word in words if word in positive_words])
            neg_count = sum([1 for word in words if word in negative_words])
            feature_set.append(pos_count)
            feature_set.append(neg_count)
            feature_set.append(pos_count - neg_count)

            # Implicit incongruity features
            implicit_phrases = ['so much that', 'not really', 'yeah right', 'as if', 'just kidding']
            feature_set.append(sum([1 for phrase in implicit_phrases if phrase in text]))

            features.append(feature_set)
        
        return pd.DataFrame(features)

# Extract features
vectorizer = CountVectorizer(max_features=500)  # Reduced number of features
tfidf_transformer = TfidfTransformer()

X_train_counts = vectorizer.fit_transform(train_df['full_text'])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = vectorizer.transform(test_df['full_text'])
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Extract additional features
extractor = FeatureExtractor()
X_train_additional_features = extractor.fit_transform(train_df['full_text'])
X_test_additional_features = extractor.transform(test_df['full_text'])

# Combine TF-IDF features with additional features
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_additional_features))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_additional_features))

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_combined_res, y_train_res = smote.fit_resample(X_train_combined, train_df['sarcasm_tag'])

# Perform hyperparameter tuning with a smaller search space
param_grid = {'C': [1, 1], 'kernel': ['linear']}
svm_clf = SVC()
grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train_combined_res, y_train_res)

# Train the best model
best_svm_clf = grid_search.best_estimator_
best_svm_clf.fit(X_train_combined_res, y_train_res)

# Save the model to disk
with open('best_svm_clf_model.pkl', 'wb') as f:
    pickle.dump(best_svm_clf, f)

# Save the vectorizer and tfidf transformer to disk
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_transformer.pkl', 'wb') as f:
    pickle.dump(tfidf_transformer, f)

# Save the feature extractor to disk
with open('feature_extractor.pkl', 'wb') as f:
    pickle.dump(extractor, f)

# Evaluation on training data
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)