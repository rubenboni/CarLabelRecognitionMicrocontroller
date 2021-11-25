import logging
import threading
import socket
import time

class CameraConnection:

    connection = None
    isAvailable = True
    ip = None

    def __init__(self,_connection,_ip):
        self.connection= _connection
        self.ip = _ip




class ThreadConnections (threading.Thread):

    listConnections=None
    semaphoreConnection=None
    cameraSemaphore=None

    def __init__(self, _semaphoreConnection, _listConnections,_cameraSemaphore):
        threading.Thread.__init__(self)
        self.semaphoreConnection=_semaphoreConnection
        self.listConnections = _listConnections
        self.cameraSemaphore = _cameraSemaphore

    def run(self):
        while True:
            try:
                logging.debug("Thread connection empieza bucle")
                socketServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_address = ("192.168.0.21",12100)
                socketServer.bind(server_address)
                socketServer.listen(10)
                print("Escuchando peticiones:")
                
                while True:
                    logging.debug("Esperando conexión...")
                    connection,ipClient =socketServer.accept()
                    logging.debug("Añadiendo conexión a la lista")
                    self.semaphoreConnection.acquire()
                    self.listConnections.append(CameraConnection(connection,ipClient[0]))
                    self.semaphoreConnection.release()
                    logging.debug("Conexión añadida")
                    self.cameraSemaphore.release()
                    print("Se ha conectado",ipClient, "en el puerto", 12100)
            except Exception as excep:
                print("Error iniciando el servidor")
                logging.exception(excep)
                time.sleep(10)
                continue

