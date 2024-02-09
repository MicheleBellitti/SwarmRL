import socket
import threading
import pygame
from simulate import SimulationManager
import json

# Socket setup
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432       # Port to listen on (non-privileged ports are > 1023)
simulation_manager = SimulationManager()

async def send_status():
    if simulation_manager.get_state():
        await # implement sedning status to api process
async def handle_client_connection(client_socket):
    
    while True:
        if command := client_socket.recv(1024).decode('utf-8'):
            print(f"Received command: {command}")
            
            if command == "start":
                await simulation_manager.start()
            elif command == "pause":
                simulation_manager.pause()
            elif command == "resume":
                simulation_manager.resume()
            elif command == "stop":
                simulation_manager.quit()
                break
            else: break
    client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"Simulation Socket Server listening on {HOST}:{PORT}")
    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        
        handle_client_connection(client_socket)
            
        

if __name__ == "__main__":
    pygame.init()
    start_server()
    pygame.quit()
