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
from sklearn.metrics import classification_report



# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


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

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Not Sarcasm', 'Sarcasm'], output_dict=True)
    return report

def mainn(test_df, eval_flag):
    # Preprocess text data
    test_df['processed_text'] = test_df['body'].apply(preprocess_text)

    # Extract contextual information (e.g., parent comment)
    def extract_context_features(df, parent_col='parent_id', text_col='processed_text'):
        context_dict = df.set_index('id')[text_col].to_dict()
        df['context_text'] = df[parent_col].map(context_dict).fillna('')
        df['full_text'] = df[text_col] + ' ' + df['context_text']
        return df

    test_df = extract_context_features(test_df)

    # Load the saved model, vectorizer, tfidf transformer, and feature extractor
    with open('best_svm_clf_model.pkl', 'rb') as f:
        best_svm_clf = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('tfidf_transformer.pkl', 'rb') as f:
        tfidf_transformer = pickle.load(f)

    with open('feature_extractor.pkl', 'rb') as f:
        extractor = pickle.load(f)

    # Extract features
    X_test_counts = vectorizer.transform(test_df['full_text'])
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    X_test_additional_features = extractor.transform(test_df['full_text'])
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_additional_features))

    # Predict on test set
    y_pred = best_svm_clf.predict(X_test_combined)

    # Add predictions to test dataframe
    test_df['sarcasm_pred'] = y_pred

    # Save the predictions to a new CSV file
    test_df.to_csv('reddit_test_with_predictions.csv', index=False)

    report = ""
    if eval_flag == 1:
        # Evaluate the model
        y_true = test_df['sarcasm_tag']  # Assuming you have the true labels in 'sarcasm_tag' column
        report = evaluate_model(y_true, y_pred)
        
        # Convert the classification report to DataFrames
        report_df = pd.DataFrame.from_dict({key: value for key, value in report.items() if isinstance(value, dict)}, orient='index')
        accuracy_df = pd.DataFrame.from_dict({'accuracy': report['accuracy']}, orient='index')

        # Merge DataFrames
        report_df = pd.concat([report_df, accuracy_df])
        # Print the evaluation report
        report = report_df
        print("Evaluation Report:")
        print(report)

    # Print the predictions for the entire test dataset
    print(test_df[['body', 'sarcasm_pred']])
    return test_df[['index', 'body', 'author', 'id', 'sarcasm_pred']].rename(columns={'index': 'Index', 'id': 'ID', 'author': 'Author', 'body': 'Body' , 'sarcasm_pred': 'Sarcasm Tag'}).reindex(columns=['Index', 'ID', 'Author', 'Body', 'Sarcasm Tag']), report


def predict_sarcasm(text):
    # Load the saved model, vectorizer, tfidf transformer, and feature extractor
    with open('best_svm_clf_model.pkl', 'rb') as f:
        best_svm_clf = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('tfidf_transformer.pkl', 'rb') as f:
        tfidf_transformer = pickle.load(f)

    with open('feature_extractor.pkl', 'rb') as f:
        extractor = pickle.load(f)

    # Preprocess text data
    processed_text = preprocess_text(text)

    # Extract contextual features
    context_text = ''  # Assuming no context for single prompt
    full_text = processed_text + ' ' + context_text

    # Extract features
    X_counts = vectorizer.transform([full_text])
    X_tfidf = tfidf_transformer.transform(X_counts)
    X_additional_features = extractor.transform([full_text])
    X_combined = np.hstack((X_tfidf.toarray(), X_additional_features))

    # Predict sarcasm
    sarcasm_pred = best_svm_clf.predict(X_combined)[0]  # Assuming a single prediction
    
    return sarcasm_pred