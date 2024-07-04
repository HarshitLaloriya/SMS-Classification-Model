from flask import Flask, render_template
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



@app.route(rule='/')
def main():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    txt = "'hello i act acted acting actor 20% better ~`!@#$%^&*()_+=-{[\\|;:<>,./?\"\']}')"
    print(text_preprocess(txt))

    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)