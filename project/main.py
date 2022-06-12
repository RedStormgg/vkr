from flask import Blueprint, render_template

from project.userpath import getUserDirPath
from . import db
from flask_login import login_required, current_user
from os import listdir
from os.path import isfile, join

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
def profile():
    if not(current_user.is_authenticated):
        return "Отказано в доступе! Авторизуйтесь!" 

    userDirpath = getUserDirPath(current_user.name, current_user.email)
    files = [f for f in listdir(userDirpath) if isfile(join(userDirpath, f))]
    return render_template('profile.html', name=current_user.name, files=files)


from flask import send_from_directory

@main.route('/files/<path:path>')
def send_report(path):
    if not(current_user.is_authenticated):
        return "Отказано в доступе! Авторизуйтесь!"
    print(path)
    userDirpath = getUserDirPath(current_user.name, current_user.email)
    print(userDirpath)
    return send_from_directory(userDirpath, path)