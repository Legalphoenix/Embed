import socket
import subprocess
import os
import signal
import logging

from shared_config import TIKA_SERVER_JAR, TIKA_SERVER_PORT

class TikaServer:
    def __init__(self):
        self.process = self.start_tika_server()

    def start_tika_server(self):
        if not os.path.isfile(TIKA_SERVER_JAR):
            raise FileNotFoundError(f"Tika server JAR not found: {TIKA_SERVER_JAR}")

        command = ['java', '-jar', TIKA_SERVER_JAR, f'-p{TIKA_SERVER_PORT}']
        logging.info(f"Starting Tika server on port {TIKA_SERVER_PORT}")
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

tika_server = TikaServer()

def signal_handler(sig, frame):
    tika_server.stop()
    logging.info("Tika server stopped")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
