import numpy as np
import pandas as pd

from surprise import dump
import surprise
import json


def recommend(style):
    #uid 1 = summer, uid 2 = classic, uid 3 = rose

    if style == 'summer':
        uid = 1
    elif style == 'classic':
        uid = 2
    else:
        uid = 3

    r_cols = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv('item_list.csv', names=r_cols,  sep=',')


    _, loaded_algo = dump.load('recommend_model')

    predict = []

    for i in set(ratings['item_id']):
        if loaded_algo.predict(uid, i)[2] is None:
                predict.append((loaded_algo.predict(uid, i)[1]+'.jpg',round(loaded_algo.predict(uid, i)[3])))

    predict.sort(key = lambda x:x[1], reverse = True)

    somedict = { }
    for i in range(3):
        somedict[f'image{i}'] = { }
        somedict[f'image{i}']['path'] =  predict[i][0]
        somedict[f'image{i}']['rating'] = predict[i][1]


    return json.dumps(somedict)

recommend('classic')
