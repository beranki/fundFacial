from flask_login import UserMixin
from sqlalchemy.sql import func
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
DB_NAME = "users.db"

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    #extra_data = db.Column(db.String(200))
    amount = db.Column(db.Float)
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    #user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    from_id = db.Column(db.Integer)
    to_id = db.Column(db.Integer)
    status = db.Column(db.String(10))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(100))
    username = db.Column(db.String(100))
    balance = db.Column(db.Float, default=10000.00)
    #transactions = db.relationship('Transaction')
