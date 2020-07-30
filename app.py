import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle
import pandas as pd

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/sentiment_predict',methods=['GET','POST'])
def pred():
	if request.method == 'POST':
		import re
		import nltk #For Stop Words i.e the,a,an
		nltk.download('stopwords')
		from nltk.corpus import stopwords
		from nltk.stem.porter import PorterStemmer #to remove conjucation to make all words in present
		corpus = []
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

if __name__ == "__main__":
	app.run(debug=True)