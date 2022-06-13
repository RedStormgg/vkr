from codecs import utf_8_decode
from flask import Blueprint, request, flash
import numpy as np

from project.userpath import getUserDirPath
from .approx import GRNN, ransac, RBFNet, least_squares, pca, getdeterminationcoef
import traceback
from flask_login import login_required, current_user
from datetime import datetime


uploadfile = Blueprint('uploadfile', __name__)


@uploadfile.route('/upload', methods=['POST'])
def upload_post():

    if 'data' not in request.files:
        return "нет файла"

    file = request.files['data']
    noSave = request.form['noSave'] == "true"
    
    if file.filename == '':
        return "нет файла"

    rawtext = utf_8_decode(file.stream.read())[0]
    lineindex = 0
    xlist = []
    ylist = []
    for line in rawtext.splitlines():
        lineindex = lineindex + 1
        xy = line.split(";")
        if (len(xy) != 2):
            return ("Несоответствие данных X и Y!", 400)
        try:
            x = float(xy[0])
            y = float(xy[1])
        except:
            return ("Неверный формат данных! Строка " + str(lineindex) + ": \"" + line + "\"", 400)
        xlist.append(x)
        ylist.append(y)

    if (current_user.is_authenticated and not noSave):
        file.stream.seek(0)
        userDirpath = getUserDirPath(current_user.name, current_user.email)
        file.save(userDirpath + "/" + str(datetime.now()).replace(":","-").replace(" ","_"))
        
        
    rbfxlist = np.array(xlist)
    rbfylist = np.array(ylist)

    rbfnet = RBFNet(lr=1e-2, k=2)
    rbfnet.fit(rbfxlist, rbfylist)
    rbfy_pred = list([x[0] for x in rbfnet.predict(rbfxlist).tolist()])

    grnnxlist = np.array(xlist).reshape(-1,1)
    grnnylist = np.array(ylist)
    print(grnnxlist)
    grnn = GRNN(n_splits=4)
    
    grnn.fit(grnnxlist, grnnylist)
    grnny_pred = grnn.predict(grnnxlist).tolist()


    ransac_pred = [
        xy[1] for xy in ransac(
            list([x, ylist[xi]] for xi, x in enumerate(xlist)),
            max_distance=(max(ylist) - min(ylist))
        )
    ]

    leastsq_data = np.array(list([x, ylist[xi]] for xi, x in enumerate(xlist)))
    leastsq_pred = [xy[1] for xy in least_squares(leastsq_data)]

    pca_data = np.array(list([x, ylist[xi]] for xi, x in enumerate(xlist)))
    pca_pred = [xy[1] for xy in pca(pca_data)]


    return {
        "graphs": (
            {
                "id": "x",
                "name": "X",
                "data": list(xlist),
                "href":"" 
            },
            {
                "id": "y",
                "name": "Y",
                "data": list(ylist),
                "href":""  
            },
            {
                "id": "rbfy",
                "name": "RBF",
                "data": rbfy_pred,
                "href":"https://en.wikipedia.org/wiki/Radial_basis_function_network",
                "coef": getdeterminationcoef(ylist,rbfy_pred) 
            },
            {
                "id": "grnny",
                "name": "GRNN",
                "data": grnny_pred,
                "href":"https://en.wikipedia.org/wiki/General_regression_neural_network",
                "coef": getdeterminationcoef(ylist,grnny_pred)   
            },
            {
                "id": "ransac",
                "name": "RANSAC",
                "data": ransac_pred,
                "href":"https://ru.wikipedia.org/wiki/RANSAC",
                "coef": getdeterminationcoef(ylist,rbfy_pred)   
            },
            {
                "id": "leastsq",
                "name": "МНК",
                "data": leastsq_pred,
                "href":"http://www.cleverstudents.ru/articles/mnk.html",
                "coef": getdeterminationcoef(ylist,leastsq_pred)  
            },
            {
                "id": "pca",
                "name": "PCA",
                "data": pca_pred,
                "href":"https://ru.wikipedia.org/wiki/Метод_главных_компонент",
                "coef": getdeterminationcoef(ylist,pca_pred) 
            }
        )
    }
