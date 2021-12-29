from flask import Flask
import os

from flask_wtf import CSRFProtect

SECRET_KEY = os.urandom(32)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SESSION_COOKIE_SECURE'] = False
csrf = CSRFProtect(app)
from webpage import routes

