import os
import unittest

from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager
from flask import render_template,request
from flask_executor import Executor

from app import blueprint
from app.main import create_app
#from app.main import db
#from app.main.model import user, blacklist
import logging
from logging import Formatter, FileHandler
from logging.handlers import RotatingFileHandler

from app.main.templates.forms import *
from app.main.newstone.word2vec import w2v
from app.main.newstone.CNN_Model import dmodel
from app.main.newstone.sentdic import sentdic
from modelconfig import *




app = create_app(os.getenv('NEWSTONE_ENV') or 'dev')
app.register_blueprint(blueprint)

app.app_context().push()

#manager = Manager(app)

#migrate = Migrate(app, db)
#manager.add_command('db', MigrateCommand)



#@manager.command
def run(debug = False):
	port = 5000
	model_path =gas_model
	word2vec_path = new_word2vec_path

	app.config['word2vec_path'] = total_word2vec_path
	app.config['model_path'] = model_path
	app.config['delidx_path'] = 'app/main/model/deletedidx.npy'
	app.config['labelspread_path'] = 'app/main/model/labelspreading_result.pkl'
	app.debug = debug
	####
	w2v.init_app(app)
	dmodel.init_app(app)
	sentdic.init_app(app)
	if not app.debug:
		log_filename = 'logs/error.log'
		os.makedirs(os.path.dirname(log_filename), exist_ok=True)

		#print(log_filename)

		file_handler = FileHandler(log_filename, mode='w', encoding=None, delay=True)
		file_handler.setFormatter(
			Formatter('%(asctime)s %(5)s: %(message)s [in %(pathname)s:%(lineno)d]')
		)
		app.logger.setLevel(logging.DEBUG)
		file_handler.setLevel(logging.DEBUG)
		app.logger.addHandler(file_handler)
		app.logger.info('server start')

		#@app.before_request
		#def log_request_info():
#			app.logger.debug('request Headers: %s', request.headers)
			# app.logger.debug('Body: %s', request.get_data())
		@app.after_request
		def after_request(response):
			#app.logger.debug('response Headers: %s', response.headers)
			return response

	#print(app.logger)

	app.run(host='0.0.0.0', port = port, threaded=True, debug=debug)



#@manager.command
def test():
	"""Runs the unit tests."""
	tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
	result = unittest.TextTestRunner(verbosity=2).run(tests)
	if result.wasSuccessful():
		return 0
	return 1

if __name__ == '__main__':

	#manager.run()
	run()