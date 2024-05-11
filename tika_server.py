
#tika server module
import socket
import subprocess
import os
from os.path import dirname, abspath, join
import atexit
import signal
import logging


# Singleton pattern to ensure we only start one Tika server
class TikaServer:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(TikaServer, cls).__new__(cls)
            cls.instance.process = cls.start_tika_server()
        return cls.instance

    @staticmethod
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    @staticmethod
    def start_tika_server():
        tika_jar = join(dirname(abspath(__file__)), 'tika', 'tika-server-standard-2.9.2.jar')
        if not os.path.isfile(tika_jar):
            raise FileNotFoundError(f"Tika server JAR not found: {tika_jar}")

        port = TikaServer.find_free_port()
        os.environ['TIKA_SERVER_ENDPOINT'] = f'http://localhost:{port}'
        os.environ['TIKA_SERVER_JAR'] = f"file:///{tika_jar}"
        command = ['java', '-jar', tika_jar, f'-p{port}']
        logging.info(f"port number for tika: {port}")
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        if self.instance and self.instance.process:
            self.instance.process.terminate()
            self.instance.process.wait()

#code to close tika server
def close_tika_server(app, tika_server):
    def stop_tika_server():
        if tika_server:
            tika_server.stop()
            print("Tika server stopped")

    def signal_handler(sig, frame):
        stop_tika_server()
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(stop_tika_server)
