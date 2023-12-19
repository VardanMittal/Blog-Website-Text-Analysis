import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
import csv

class TextAnalysis:
    def __init__(self):
        self.read = pd.read_excel("./Input.xlsx")
    def syllables(self, word):
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def data_analysis(self, data, url, id, db):
        words = word_tokenize(data)
        words_count = len(words)
        sentences = sent_tokenize(data)
        words = [word.lower() for word in words if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(data)
        avg_sentence_length = len(words) / len(sentences)
        tagged_words = nltk.pos_tag(words)
        personal_pronouns = [word[0] for word in tagged_words if word[1] == 'PRP']
        complex_words = [word for word in filtered_words if self.syllables(word) >= 3]
        percentage_complex_words = len(complex_words) * 100 / len(filtered_words)
        analysis = {
            "URL_ID": id,
            "URL": url,
            "POSITIVE SCORE":sentiment_scores['pos'],
            "NEGATIVE SCORE":sentiment_scores['neg'],
            "POLARITY SCORE": sentiment_scores['compound'],
            "SUBJECTIVITY SCORE":sia.polarity_scores(data)['compound'],
            "AVG SENTENCE LENGTH":avg_sentence_length,
            "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
            "FOG INDEX": 0.4 * (avg_sentence_length+percentage_complex_words),
            "AVG NUMBER OF WORDS PER SENTENCE": len(sentences)/words_count,
            "COMPLEX WORD COUNT": len(complex_words),
            "WORD COUNT": words_count,
            "SYLLABLES PER WORD":sum(self.syllables(word) for word in filtered_words) / len(filtered_words),
            "PERSONAL PRONOUNS": len(personal_pronouns),
            "AVG WORD LENGTH": sum(len(word) for word in words) / words_count
        }
        db.append(analysis)
        return db
    def output(self,data):
        fields= ["URL_ID", "URL", "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX", "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT","WORD COUNT", "SYLLABLES PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"]

        with open("Output.csv", "w") as file:
            writer = csv.DictWriter(file, fieldnames = fields)
            writer.writeheader()
            writer.writerows(data)
    def pipeline(self):
        db = []
        for j,i in self.read.iterrows():
            with requests.Session() as s:
                data = " "
                page = s.get(i["URL"])
                soup = BeautifulSoup(page.content, "html5lib")
                data += soup.title.string
                divs = soup.find('div', attrs = {'class':'td-post-content tagdiv-type'})
                if divs:
                    paragraphs = divs.find_all('p')
                    for paragraph in paragraphs:
                        data += paragraph.get_text()
                analysed_data = self.data_analysis(data,i["URL"], i["URL_ID"], db)
        self.output(analysed_data)


offer = TextAnalysis()

offer.pipeline()