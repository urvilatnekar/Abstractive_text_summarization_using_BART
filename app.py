from flask import Flask, render_template, request
from newspaper import Article #Article class from the newspaper library for article scraping
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
model_name = 'facebook/bart-large-cnn' #The pre-trained BART model and tokenizer are loaded from 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_article(article): #function is defined to generate a summary for a given article text.
    # Load BART model and tokenizer

    # Tokenize and encode the article
    inputs = tokenizer.encode(article, return_tensors='pt',
max_length=1024, truncation=True) #The article text is tokenized and encoded using the tokenizer.

    # Generate summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=150,
early_stopping=True) #The BART model generates a summary using the encoded input text.
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) #The generated summary is decoded using the tokenizer.

    return summary

@app.route('/', methods = ['GET', 'POST']) #The / route serves as the homepage.
def index():
    if request.method == 'GET': #If the request method is GET, the application renders the initial HTML template with no summary displayed.
        return render_template('index.html', show_summary=False)
#If the request method is POST, the application extracts the article URL from the form, scrapes the article's content and title using newspaper,
# generates a summary using the summarize_article function, and renders the HTML template with the article text, title, and generated summary.
    elif request.method == 'POST':
        url = request.form['url']
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title

        summary = summarize_article(text)
        return render_template('index.html', summary=summary, text=text, show_summary=True, title=title)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True) #The application runs using app.run(), which starts the Flask development server.
