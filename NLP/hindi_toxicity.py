import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer
from tensorflow import keras
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer = CountVectorizer()
Tvectorizer = TfidfVectorizer()
def load_dataset(file_path):
    df = pd.read_csv(file_path, header=0, names=['review', 'sentiment'])
    hui = []
    for i in df.sentiment.values:
        if i =='negative':
            hui.append(1)
        else:
            hui.append(0)
    df = df.drop(columns='sentiment')
    df['sentiment'] = hui
    return df

def preprocess(text):
    new_text = []
    for i in text.split(" "):
        i = '' if i.startswith('@') and len(i) > 1 else i
        i = '' if i.startswith('http') else i
        i = i.replace("#","")
        new_text.append(i)
    return " ".join(new_text).strip().replace("  ", " ")

def tokenize_texts(texts, tokenizer, max_length=60):
    tokenized_texts = []
    for text in tqdm(texts):
        tokenized_output = tokenizer(text)
        input_ids = tokenized_output['input_ids']
        padded_input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(input_ids))
        tokenized_texts.append(padded_input_ids)
    return np.array(tokenized_texts)

def bow(text):
  X_bow = vectorizer.fit_transform(X)
  return X_bow

def tf_idf(text):
  X_tfidf = Tvectorizer.fit_transform(X)
  return X_tfidf

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=50000)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(60,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=35)
    return model

def predict_toxicity(model, text, tokenizer):
    preprocessed_text = preprocess(text)
    tokenized_text = tokenize_texts([preprocessed_text], tokenizer)[0]
    return model.predict(np.array([tokenized_text]))[0]

def predict_toxicitynlp(model,statement,vectorizer):
    statement_bow = vectorizer.transform([statement])
    prediction = model.predict(statement_bow)
    return prediction[0]


if __name__ == "__main__":
    # Load dataset
    file_path = r"C:\Users\palal oza/Downloads/hi_3500.csv"
    df = load_dataset(file_path)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Tokenize texts
    X = df.review.values
    y = df.sentiment.values
    X_tokenized = tokenize_texts(X, tokenizer)
    xbow=bow(X)
    xtfidf=tf_idf(X)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_tokenized, y, random_state=91)
    X1_train, X1_test, y1_train, y1_test = train_test_split(xbow, y, random_state=5)
    X2_train, X2_test, y2_train, y2_test = train_test_split(xtfidf, y, random_state=5)
    # Train models
    rf_modelt = train_random_forest(X_train, y_train)
    rf_bow =train_random_forest(X1_train, y1_train)
    rf_tfidf=train_random_forest(X2_train, y2_train)
    lr_model = train_logistic_regression(X_train, y_train)
    lr_model_tfidf=train_logistic_regression(X2_train, y2_train)
    nn_model = train_neural_network(X_train, y_train)

    # Evaluate models
    print("Random Forest Classifier Report(After tokenization):")
    print(classification_report(rf_modelt.predict(X_test), y_test))

    print("Random Forest Classifier Report(Using BAG of words):")
    print(classification_report(rf_bow.predict(X1_test), y1_test))

    print("Random Forest Classifier Report(Using TF-IDF):")
    print(classification_report(rf_tfidf.predict(X2_test), y2_test))
    
    print("Logistic Regression Classifier Report(with tokenization):")
    print(classification_report(lr_model.predict(X_test), y_test))

    print("Logistic Regression Classifier Report(with TD-IDF):")
    print(classification_report(lr_model_tfidf.predict(X2_test), y2_test))

    test_loss, test_acc = nn_model.evaluate(X_test, y_test)
    print('Neural Network Test Accuracy:', test_acc)

    # Test user input
    input_text = input("Enter the Hindi text to test toxicity: ")
    rf_toxicity_token = predict_toxicity(rf_modelt, input_text, tokenizer)
    rf_toxicity_bow = predict_toxicitynlp(rf_bow, input_text,vectorizer)
    rf_toxicity_tfidf = predict_toxicitynlp(rf_tfidf, input_text,Tvectorizer)
    lr_toxicity = predict_toxicity(lr_model, input_text, tokenizer)
    lr_toxicity_tfidf = predict_toxicitynlp(lr_model_tfidf, input_text, Tvectorizer)
    nn_toxicity = predict_toxicity(nn_model, input_text, tokenizer)

    print("Random Forest Toxicity Score after tokenization:", rf_toxicity_token)
    print("Random Forest Toxicity Score after BAG of Words:", rf_toxicity_bow)
    print("Random Forest Toxicity Score after TD-IDF:", rf_toxicity_tfidf)
    print("Logistic Regression Toxicity Score after tokenization:", lr_toxicity)
    print("Logistic Regression Toxicity Score after TD-IDF:", lr_toxicity_tfidf)
    print("Neural Network Toxicity Score:", nn_toxicity)


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.metrics import accuracy_score
def toxc(toxicity):
    if toxicity == 0:
        return("non-toxic.")
    else:
        return("toxic.")
# Function to predict toxicity and update UI
def predict_toxicity_and_update_ui():
    # Get user input
    input_text = text_entry.get()

    # Predict toxicity for all models
    rf_toxicity_token = predict_toxicity(rf_modelt, input_text, tokenizer)
    rftstr=toxc(rf_toxicity_token)
    print(rftstr)
    rf_toxicity_bow = predict_toxicitynlp(rf_bow, input_text, vectorizer)
    rfbstr=toxc(rf_toxicity_bow)
    rf_toxicity_tfidf = predict_toxicitynlp(rf_tfidf, input_text, Tvectorizer)
    rftfstr=toxc(rf_toxicity_tfidf)
    lr_toxicity = predict_toxicity(lr_model, input_text, tokenizer)
    lrtstr=toxc(lr_toxicity)
    lr_toxicity_tfidf = predict_toxicitynlp(lr_model_tfidf, input_text, Tvectorizer)
    lrtfstr=toxc(lr_toxicity_tfidf)
    nn_toxicity = predict_toxicity(nn_model, input_text, tokenizer)
    
    rf_pred_label.configure(text=f"Random Forest(after tokenization): {rftstr}")
    rf_pred_label1.configure(text=f"Random Forest(after Bag of words): {rfbstr}")
    rf_pred_label2.configure(text=f"Random Forest(after TF-IDF): {rftfstr}")
    lr_pred_label.configure(text=f"Logistic Regression(after tokenization): {lrtstr}")
    lr_pred_label1.configure(text=f"Logistic Regression(TF-IDF): {lrtfstr}")
    nn_pred_label.configure(text=f"Neural Network: {nn_toxicity}")

    update_ui_colors(rf_pred_label, rf_toxicity_token)
    update_ui_colors(rf_pred_label1, rf_toxicity_bow)
    update_ui_colors(rf_pred_label2, rf_toxicity_tfidf)
    update_ui_colors(lr_pred_label, lr_toxicity)
    update_ui_colors(lr_pred_label1, lr_toxicity_tfidf)
    update_ui_colors(nn_pred_label, nn_toxicity)

def update_ui_colors(label, prediction):
    if prediction == 1:
        label.config(foreground="red")
    else:
        label.config(foreground="green")


root = tk.Tk()
root.title("Toxicity Prediction")

window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

font_size = 20
font = ("Helvetica", font_size)

text_entry_label = tk.Label(root, text="Enter Hindi text:", font=font)
text_entry_label.pack(pady=10)
text_entry = tk.Entry(root, width=50, font=font)
text_entry.pack(pady=5)


predict_button = ttk.Button(root, text="Predict Toxicity", command=predict_toxicity_and_update_ui, style='TButton', width=20)
predict_button.pack(pady=10)

rf_pred_label = tk.Label(root, font=font,text="Random Forest(after tokenization):")
rf_pred_label.pack()
rf_pred_label1 = tk.Label(root, font=font,text="Random Forest(After Bag of words):")
rf_pred_label1.pack()
rf_pred_label2 = tk.Label(root,font=font, text="Random Forest(after TF-IDF):")
rf_pred_label2.pack()
lr_pred_label = tk.Label(root, font=font,text="Logistic Regression(after tokenization):")
lr_pred_label.pack()
lr_pred_label1 = tk.Label(root,font=font, text="Logistic Regression(after TF-IDF):")
lr_pred_label1.pack()
nn_pred_label = tk.Label(root,font=font, text="Neural Network:")
nn_pred_label.pack()

style = ttk.Style()
style.configure('TButton', foreground="blue",font=font)

root.mainloop()
