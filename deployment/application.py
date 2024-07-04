from flask import Flask, render_template

app = Flask(import_name=__name__)

@app.route(rule='/')
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)