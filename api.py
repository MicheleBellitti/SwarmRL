from flask import Flask, jsonify, request
import flask_cors
from flask_restful import Api, Resource

from threading import Thread
import time
import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 3000  # The port used by the server


app = Flask(__name__)
flask_cors.CORS(app=app)
api = Api(app)

# Thread to run simulation separately
simulation_thread = None


@app.route("/")
def hello():
    return app.send_static_file("index.html")


def send_command_to_simulation(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, 65432))
        s.sendall(command.encode())
        data = s.recv(1024)
    return data


class ControlSimulation(Resource):
    def post(self, command):

        send_command_to_simulation(command)
        print(command)
        return jsonify({"status": "success", "command": command})


@app.route("/api/status")
def status():
    data = {
        "state": "running",
        "current_episode": 10,
        "total_episodes": 100,
        "latest_results": {
            "episode": 10,
            "steps": 100,
            "total_reward": 100,
            "avg_reward": 1.0,
        },
    }
    return jsonify(data)


@app.route("/api/data")
def data():
    data = {
        "state": "running",
        "current_episode": 10,
        "total_episodes": 100,
        "latest_results": {
            "episode": 10,
            "steps": 100,
            "total_reward": 100,
            "avg_reward": 1.0,
        },
    }
    return jsonify(data)


if __name__ == "__main__":
    api.add_resource(ControlSimulation, "/api/control/<command>")

    app.run(host=HOST, port=PORT, debug=True)
