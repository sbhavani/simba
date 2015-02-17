__author__ = 'sb'

import os
from flask import Flask, render_template, request, url_for

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the default URL, which loads the form
@app.route('/')
def form():
    return render_template('index.html')

# Define a route for the action of the form, for example '/hello/'
# We are also defining which type of requests this route is
# accepting: POST requests in this case
@app.route('/hello/', methods=['POST'])
def hello():
    lion=request.form['yourname']
    return render_template('results.html', lion=lion)

# Run the app :)
if __name__ == '__main__':
  app.run(
        host="0.0.0.0",
        port=int("8080")
  )
