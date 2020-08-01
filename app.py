import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle
import pandas as pd
import re
import nltk #For Stop Words i.e the,a,an
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to remove conjucation to make all words in present
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  
corpus = []

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/sentiment_predict',methods=['GET','POST'])
def pred():
	if request.method == 'POST':
		for i in range(0,1000):
		    review = re.sub('[^a-zA-Z]',' ',df['Review'][i]) #cleaned all commaas,stops and all
		    review = review.lower() #convert all words to lowercase
		    review = review.split()
		    ps = PorterStemmer()
		    all_stopwords = stopwords.words('english')
		    all_stopwords.remove('not')
		    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
		    review = ' '.join(review)
		    corpus.append(review)

		from sklearn.feature_extraction.text import CountVectorizer
		cv = CountVectorizer() #take most frequent Words
		X = cv.fit_transform(corpus).toarray()
		y = df['Liked']

		from sklearn.model_selection import train_test_split

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

		from sklearn.naive_bayes import GaussianNB

		nb = GaussianNB()

		nb.fit(X_train,y_train)

		pickle.dump(nb,open('sentiment_model.pkl','wb'))

		model=pickle.load(open('sentiment_model.pkl','rb'))

		text = request.form['Review']
		new_review = re.sub('[^a-zA-Z]', ' ', text)
		new_review = new_review.lower()
		new_review = new_review.split()
		ps = PorterStemmer()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
		new_review = ' '.join(new_review)
		new_corpus = [new_review]
		#print(new_corpus)
		new_X_test = cv.transform(new_corpus).toarray()
		new_y_pred = model.predict(new_X_test)
		#print(new_y_pred)
		if new_y_pred==1:
			tt = 'Positive Review'
		else:
		    tt= "Negative Review"

		return render_template('sentiment.html',prediction_text='{}'.format(tt))
	else:
		return render_template('sentiment.html',prediction_text='{}'.format('Analyzing...'))

@app.route('/summarize')
def summary():
	return render_template("text_summarize.html")

@app.route('/summary_predict',methods=['POST'])
def summarize():
	raw_text = request.form['Text']
	stopWords = set(stopwords.words("english"))
	word_frequencies = {}  
	for word in nltk.word_tokenize(raw_text):  
		if word not in stopWords:
			if word not in word_frequencies.keys():
				word_frequencies[word] = 1
			else:
				word_frequencies[word] += 1

	maximum_frequncy = max(word_frequencies.values())
	for word in word_frequencies.keys():  
		word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

	sentence_list = nltk.sent_tokenize(raw_text)
	sentence_scores = {}
	for sent in sentence_list:
		for word in nltk.word_tokenize(sent.lower()):
			if word in word_frequencies.keys():
				if len(sent.split(' ')) < 30:
					if sent not in sentence_scores.keys():
						sentence_scores[sent] = word_frequencies[word]
					else:
						sentence_scores[sent] += word_frequencies[word]

	summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
	summary = ' '.join(summary_sentences)
	clean_text = summary
	return render_template('summary_predict.html',prediction_text='{}'.format(clean_text),text='{}'.format(raw_text))

if __name__ == "__main__":
	app.run(debug=True)