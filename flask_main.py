from flask import Flask, request
from flask import render_template
from werkzeug.utils import secure_filename

import plotly
import plotly.graph_objs as go
import json, os
import numpy as np;
import pandas as pd
from collections import Counter

import fashion_model
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

app = Flask(__name__)
app.config['UPLOAD_DIR'] = './static/upload_files'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model, labels = fashion_model.load_model(device, label_type='category')


@app.route('/upload')
def upload_main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <form action="http://141.223.140.20:5000/predict-image" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit">
        </form>
    </body>
    </html>"""


@app.route('/predict-image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_DIR'], fname)
        f.save(path)
        trans = transforms.ToTensor()
        x = trans(Image.open(path))[None,:]
        x = x.to(device)
        y = model(x)[0]
        y = nn.functional.softmax(y)
        print(y)
        res = { }
        for i in range(len(labels)//2):
            res[labels[i]] = float(y[i])
        return json.dumps(res)


def create_plot(labels, values):
    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    values = [4500, 2500, 1053, 500]
    tot = sum(values)
    data = [ go.Pie(labels=labels, values=values, 
                    textinfo='label+percent', hovertemplate=f"%{{value}}/{tot}<extra></extra>",
                    hole=.3) ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/')
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('test.html', name=name)

@app.route('/plot')
def dashboard(name=None):
    graph = create_plot()
    return render_template('test_dashboard.html', plot=graph)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
