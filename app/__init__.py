# app/__init__.py

from flask_restplus import Api
from flask import Blueprint

#from .main.controller.user_controller import api as user_ns
#from .main.controller.auth_controller import api as auth_ns
from .main.controller.news_controller import api as news_ns


blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='FLASK RESTPLUS API BOILER-PLATE WITH CELERY AND WEBPAGE',
          version='1.0',
          description='a boilerplate for flask restplus web service'
          )

#api.add_namespace(user_ns, path='/user')
api.add_namespace(news_ns, path='/news')
#api.add_namespace(auth_ns)

#api.add_namespace(web_ns)