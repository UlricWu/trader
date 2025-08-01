import yaml
from data import db, Data
from utilts import logs
import os, json
from flask import Flask, jsonify, request


def create_app():
    config_path = os.environ.get('CONFIGS', "trader/config/config_test.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logs.record_log("start gunicorn")
    apps = Flask(__name__)
    logs.record_log("run gunicorn")
    logs.record_log(config['data'])
    # db = config['data']['database']
    database = db.Database()
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
                # print(record)
                db.update_daily(table, record)
            # print(os.path.curdir)

            # subprocess.check_call(["sqlite3", database, f".backup "])

            db.back_db()
            return jsonify('done')

        except Exception as e:
            return jsonify({'error': str(e)})

    @apps.route('/get_daily', methods=['post', 'get'])
    @logs.catch()
    def extra_daily():
        inputs = json.loads(request.data)  # flask.request

        if not inputs:
            today = "20250305"
        else:
            today = inputs['today']
        logs.record_log(f'inputs = {inputs} with today = {today}')
        return db.extract_table(end_day=today).to_json(orient='records')

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
