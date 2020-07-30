from flask_wtf import FlaskForm
from wtforms import SubmitField,TextAreaField

class SentimentForm(FlaskForm):
    Review = TextAreaField('Review',
                           validators=[DataRequired()])
    submit = SubmitField('Predict')