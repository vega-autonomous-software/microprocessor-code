import socket
import struct
import threading
import json
import time


class TcpBroadcastServer:
    def __init__(self, bind_host="0.0.0.0", bind_port=9000):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.server = None
        self.clients = []
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.bind_host, self.bind_port))
        self.server.listen(5)
        self.running = True

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        print(f"[TCP] Broadcasting on {self.bind_host}:{self.bind_port}")

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.clients.append(conn)
                print(f"[TCP] Client connected: {addr}")
            except Exception as e:
                if self.running:
                    print(f"[TCP] Accept error: {e}")

    def _broadcast_packet(self, packet: bytes):
        dead = []

        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(packet)
                except Exception:
                    dead.append(client)

            for client in dead:
                try:
                    client.close()
                except Exception:
                    pass
                if client in self.clients:
                    self.clients.remove(client)

    def send_bytes(self, payload: bytes):
        packet = struct.pack("<I", len(payload)) + payload
        self._broadcast_packet(packet)

    def send_json(self, obj: dict):
        payload = json.dumps(obj).encode("utf-8")
        self.send_bytes(payload)

    def stop(self):
        self.running = False

        with self.lock:
            for client in self.clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.clients.clear()

        if self.server:
            try:
                self.server.close()
            except Exception:
                pass