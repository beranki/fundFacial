from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from website.models import User, Transaction, db, DB_NAME
from website.views import b_views

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "sdjfoayusgsDAS"
    app.config['SQLALCHEMY_DATABASE_URI'] =  f"sqlite:///{DB_NAME}"
    db.init_app(app)
    app.register_blueprint(b_views, url_prefix = "/")
    
    if not (os.path.exists("website/" + DB_NAME)):
        db.create_all(app=app)

    login_manager = LoginManager()
    login_manager.login_view = 'views.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    return app