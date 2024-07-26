

#Here I made text files from wikipidia by copying it to notepad and formed  three different files for training the model, such as football_train, cricket_train and basketball_train.
#And similarly created three unseen text files from respective sports.
#Used TF-IDF vectorizer object with options to convert text to lowercase and remove English stop words.
#And finally trained a classifier  for document classification.
#At the bottom is the output in # comments.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Define a function to read the text files
def read_text_file(file_path):
    with open(file_path, 'r', encoding='latin1') as file:
        text = file.read()
    return text


football_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\football_train.txt"
cricket_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\cricket_train.txt"
basketball_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\basketball_train.txt"


football_text = read_text_file(football_file_path)# reading the file
cricket_text = read_text_file(cricket_file_path)
basketball_text = read_text_file(basketball_file_path)

# Combine all text data into a list
all_text = [football_text, cricket_text, basketball_text]

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')#TF-IDF vectorizer object with options to convert text to lowercase and remove English stop words.

# Fit TF-IDF vectorizer on the text data
tfidf_vectorizer.fit(all_text)

# Transform text data into TF-IDF features and convert them to dense arrays
football_tfidf_features = tfidf_vectorizer.transform([football_text]).toarray()
cricket_tfidf_features = tfidf_vectorizer.transform([cricket_text]).toarray()
basketball_tfidf_features = tfidf_vectorizer.transform([basketball_text]).toarray()

# Print the shapes of TF-IDF features after conversion
print("Football TF-IDF shape:", football_tfidf_features.shape)
print("Cricket TF-IDF shape:", cricket_tfidf_features.shape)
print("Basketball TF-IDF shape:", basketball_tfidf_features.shape)

# Concatenate the TF-IDF features
X_train = np.concatenate([football_tfidf_features, cricket_tfidf_features, basketball_tfidf_features])

# Print the shape of the concatenated TF-IDF features
print("Concatenated TF-IDF shape:", X_train.shape)

# Labels for the training data
y_train = np.array([0] * football_tfidf_features.shape[0] + [1] * cricket_tfidf_features.shape[0] + [2] * basketball_tfidf_features.shape[0])

# Training a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Paths to the test files
football_test_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\f_test.txt"
cricket_test_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\c_test.txt"
basketball_test_file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\b_test.txt"

# Read the test files
football_test_text = read_text_file(football_test_file_path)
cricket_test_text = read_text_file(cricket_test_file_path)
basketball_test_text = read_text_file(basketball_test_file_path)

# Transform test data into TF-IDF features and convert them to dense arrays
football_test_tfidf_features = tfidf_vectorizer.transform([football_test_text]).toarray()
cricket_test_tfidf_features = tfidf_vectorizer.transform([cricket_test_text]).toarray()
basketball_test_tfidf_features = tfidf_vectorizer.transform([basketball_test_text]).toarray()

# Predict labels for the test data
football_predicted_label = classifier.predict(football_test_tfidf_features)
cricket_predicted_label = classifier.predict(cricket_test_tfidf_features)
basketball_predicted_label = classifier.predict(basketball_test_tfidf_features)

# Flatten the predicted labels arrays
football_predicted_label = football_predicted_label.flatten()  #These lines flatten the arrays of predicted labels to convert them into 1D arrays, as they were originally returned as 2D arrays.
cricket_predicted_label = cricket_predicted_label.flatten()
basketball_predicted_label = basketball_predicted_label.flatten()

# Map predicted labels back to their names
labels = ['Football', 'Cricket', 'Basketball']
predicted_labels = [labels[label] for label in [football_predicted_label[0], cricket_predicted_label[0], basketball_predicted_label[0]]]

#Finally, these lines print out the predicted labels for each test file, providing insights into which category each test sample belongs to based on the predictions made by the classifier.
print("Predicted labels for f_test.txt:", predicted_labels[0])
print("Predicted labels for c_test.txt:", predicted_labels[1])
print("Predicted labels for b_test.txt:", predicted_labels[2])


#---------------------this was the output-----------------------------
#Football TF-IDF shape: (1, 317)
#Cricket TF-IDF shape: (1, 317)
#Basketball TF-IDF shape: (1, 317)
#Concatenated TF-IDF shape: (3, 317)
#Predicted labels for f_test.txt: Football
#Predicted labels for c_test.txt: Cricket
#Predicted labels for b_test.txt: Basketball