
from webpage import app
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SubmitField, ValidationError
from wtforms.validators import DataRequired, NumberRange
import yfinance as yf


def validate_ticker(field):
    ticker = yf.Ticker(field.data)
    try:
        info = ticker.info
    except:
        raise ValidationError("Invalid Ticker Symbol")

class InputForm(FlaskForm):
    ticker = StringField(label='Ticker symbol of stock: ', validators=[DataRequired()])
    start_price = FloatField(label='Price purchased at: ', validators=[NumberRange(min=0, message="Enter a positive value"), DataRequired()])
    num_shares = IntegerField(label='Number of shares purchased: ', validators=[NumberRange(min=0, message="Enter a positive value"), DataRequired()])
    future_days = IntegerField(label='Number of days into the future to predict: ', validators=[NumberRange(min=0, message="Enter a positive value"), DataRequired()])
    submit = SubmitField(label='Submit')