import yaml
import os
from data import DB, Data
from utilts import logs
import os, subprocess, pickle, requests, json
from flask import Flask, jsonify, request

from datetime import datetime


def create_app():
    config_path = os.environ.get('CONFIGS', "configs/config_test.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logs.record_log("start gunicorn")
    apps = Flask(__name__)
    logs.record_log("run gunicorn")
    print(config['data'])
    db = DB(config['data']['database'])
    data = Data(config['data']['token'])

    @apps.route('/', methods=['GET'])
    @logs.catch()
    def index():
        return jsonify({'hello': 'world'})

    @apps.route('/update_daily', methods=['POST'])
    @logs.catch()
    def update_daily():
        inputs = json.loads(request.data)  # flask.request
        daily = data.get_daily(inputs['ts_code'])
        db.insert("daily", daily)
        return jsonify('done')

    @apps.route('/get_daily', methods=['get'])
    @logs.catch()
    def extra_daily():
        return jsonify(db.extract("daily"))

    return apps


if __name__ == '__main__':
    trader_app = create_app()

    trader_app.run(host="0.0.0.0", port=6001, debug=True)

else:
    trader_app = create_app()
