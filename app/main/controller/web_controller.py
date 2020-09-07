# -*- coding: utf-8 -*-

from flask import request, jsonify, render_template
from flask_restplus import Resource
from pandas.io.json import json_normalize
import json

from app.main.util.dto import Web

