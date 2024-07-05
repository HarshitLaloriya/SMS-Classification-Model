from flask import Flask, render_template, request, jsonify
import pickle, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

app = Flask(import_name=__name__)

def text_preprocess(text:str):
    lower_case = text.lower()
    lower_case_list = nltk.word_tokenize(lower_case)
    
    words = []
    for i in lower_case_list:
        if i.isalnum() and i not in stopwords.words('english')  :
            words.append(i)

    stemed_words = []
    for j in words:
        
        stemed_words.append(ps.stem(j))
        
    return " ".join(stemed_words)

def spam_or_ham(data:str):
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    processed_data = text_preprocess(data)
    print(data)
    prediction = model.predict(tfidf.transform([processed_data]))
    print(prediction[0])
    output_pred = 'SPAM'
    if prediction[0] == 0:
        output_pred = 'HAM'
        print(output_pred)
        return output_pred
    print(output_pred)
    return output_pred


@app.route(rule='/', methods=["GET", "POST"])
def main():
    print('in main function')
    return render_template("base.html")

@app.route('/predict', methods=['POST'])
def predict():
    sms_input = request.get_json()['sms']
    store_output = spam_or_ham(sms_input)
    return jsonify({'output': store_output})

if __name__ == "__main__":
    app.run(debug=True)