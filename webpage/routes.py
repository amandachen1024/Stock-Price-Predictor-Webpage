from webpage import app
from flask import Flask, render_template
from webpage.forms import InputForm
from model import predict


@app.route('/')
def home_page():
    form = InputForm()
    while not form.submit == 'Submit':
        render_template('input.html', form=form)
    if form.validate():
        x = predict(ticker=form.ticker.data, start_price=form.start_price.data, num_shares=form.num_shares.data,
                future_days=form.future_days.data)
        return render_template('prediction.html', per_share=x)
    else:
        return render_template('prediction.html', per_share=0)

    if form.errors != {}:
        for err in form.errors.values():
            print(f'There was an error: {err}')
    return render_template('input.html', form=form)



