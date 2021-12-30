import flask

from webpage import app
from flask import Flask, render_template, request, flash
from webpage.forms import InputForm
from model import predict


@app.route('/', methods=['GET', 'POST'])
def home_page():
    form = InputForm()
    if form.validate_on_submit():
        flash("valid")
        x = predict(ticker=form.ticker.data, start_price=form.start_price.data, num_shares=form.num_shares.data,
                    future_days=form.future_days.data)
        if x != -1:
            return render_template('prediction.html', predictions=x)
        else:
            flash("Invalid Ticker Symbol")
            return render_template('input.html', form=form)

    if form.errors != {}:
        for err in form.errors.values():
            print(f'There was an error: {err}')
        return render_template('input.html', form=form)
    return render_template('input.html', form=form)



