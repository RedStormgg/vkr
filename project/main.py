import os
from flask import Blueprint, render_template

from project.userpath import getUserDirPath
from . import db
from flask_login import login_required, current_user
from os import listdir
from os.path import isfile, join
from flask import send_from_directory

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/exist/<path:path>')
@login_required
def exist(path):
    return render_template('index.html', path=path)

@main.route('/profile')
@login_required
def profile():
    userDirpath = getUserDirPath(current_user.name, current_user.email)
    files = [f for f in listdir(userDirpath) if isfile(join(userDirpath, f))]
    return render_template('profile.html', name=current_user.name, files=files)



@main.route('/files/<path:path>')
@login_required
def send_report(path):
    userDirpath = getUserDirPath(current_user.name, current_user.email)
    abspath = os.path.abspath(userDirpath)
    return send_from_directory(abspath, path)

@main.route('/delete/<path:path>')
@login_required
def delete_userfile(path):
    userDirpath = getUserDirPath(current_user.name, current_user.email)
    abspath = os.path.abspath(userDirpath)
    os.remove(os.path.join(abspath, path))
    return "OK"