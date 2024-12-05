import json
import socket

class Client():
    def __init__(self, gym_host="localhost", gym_port=5555):
        self.gym_host = gym_host
        self.gym_port = gym_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.gym_host, self.gym_port))
    
    def send_command(self, command, data=None):
        request = {"command": command}
        if data:
            request.update(data)
        self.sock.sendall(json.dumps(request).encode())
        response = self.sock.recv(1024).decode()
        return json.loads(response)
    
    def get_next_state(self, command, data):
        response = self.send_command(command, data)
        state = response["state"]
        reward = response["reward"]
        done = response["done"]
        return state, reward, done
    
    def shutdown_gym(self):
        _ = self.send_command("shutdown")
        print("Gym has been shutdown.")