import yaml
import os
from data import db, Data
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
    # db = DB(config['data']['database'])
    data = Data(config['data']['token'])

    @apps.route('/', methods=['GET'])
    @logs.catch()
    def index():
        return jsonify({'hello': 'world'})

    @apps.route('/update', methods=['POST'])
    @logs.catch()
    def update_tables():
        try:
            inputs = json.loads(request.data)  # flask.request
            logs.record_log(f'inputs = {inputs}')
            if inputs:
                tables = inputs.get('tables')
            else:
                # tables = ["daily", "stock_basic"]
                tables = ["stock_basic", "daily", ]
            today = inputs.get('today')
            for table in tables:
                record = data.get_table(table, today)

                db.update_daily(table, record)
            # return jsonify('done')
            return jsonify(db.extract_table(day=today))
        except Exception as e:
            return jsonify({'error': str(e)})

    @apps.route('/get_daily', methods=['post', 'get'])
    @logs.catch()
    def extra_daily():
        inputs = json.loads(request.data)  # flask.request
        logs.record_log(f'inputs = {inputs}')
        return jsonify(db.extract_table())

    return apps


if __name__ == '__main__':

    # lsof -t -i:6001 | xargs -r kill

    port = 6001
    # command = f"lsof -t -i:{port} | xargs -r kill"
    # os.system(command)
    trader_app = create_app()

    trader_app.run(host="0.0.0.0", port=port, debug=True)

else:
    trader_app = create_app()
