# importing required libraries for deployment and operations
from flask import Flask, render_template, request, jsonify
import pickle, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# PorterStreamer object for stemming
ps = PorterStemmer()


# flask app object
app = Flask(import_name=__name__)



# important function
# This function helps in preparing data to feed ML model with processed data.
def text_preprocess(text:str):
    # converts data to lowercase
    lower_case = text.lower()
    lower_case_list = nltk.word_tokenize(lower_case)

    # storing the processed words
    words = []
    # removing punctuation and stop words
    for i in lower_case_list:
        if i.isalnum() and i not in stopwords.words('english')  :
            words.append(i)

    # storing stemed words
    stemed_words = []
    # performing stemming
    for j in words:
        stemed_words.append(ps.stem(j))

    return " ".join(stemed_words)

# Loading model and vectorizer which i imported
def spam_or_ham(data:str):

    # loading model and vectorizer 
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    # preparing data for model
    processed_data = text_preprocess(data)
    print(data) # for help 

    # prediction on data
    prediction = model.predict(tfidf.transform([processed_data]))
    print(prediction[0]) # for help

    # THE OUTPUTS FOR JS
    output_pred = 'SPAM'
    if prediction[0] == 0:
        output_pred = 'HAM'
        print(output_pred) # for help
        return output_pred
    
    print(output_pred) # for help
    return output_pred


# main route
@app.route(rule='/', methods=["GET", "POST"])
def main():
    print('in main function')
    return render_template("base.html")

# after click on predict button
@app.route('/predict', methods=['POST'])
def predict():
    sms_input = request.get_json()['sms']
    store_output = spam_or_ham(sms_input)
    return jsonify({'output': store_output})

"""The line if __name__ == "__main__": is a common construct in Python scripts. Let’s break down what it means and why it’s used:

Purpose:
The purpose of this construct is to differentiate between whether a Python script is being executed as the main program or if it is being imported as a module into another script.
It helps prevent unintended execution of code when a script is imported.

Explanation:
When Python runs a script, it sets a special variable called __name__.
If the script is the main program being executed directly (not imported), __name__ is set to "__main__".
If the script is imported as a module into another program, __name__ is set to the name of the module (e.g., the filename without the .py extension).

Usage:
The code block following if __name__ == "__main__": will only run when the script is executed directly (as the main program).
It won’t run when the script is imported as a module."""

if __name__ == "__main__":
    app.run(debug=True)