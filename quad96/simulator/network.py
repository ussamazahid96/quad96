import zmq
import time
import threading
import numpy as np


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains the defination for the ZMQ socket connection used to connect to the unity simulator

class ZMQPlug:
    def __init__(self, state_ptr, queue, port=4567):
        print("[Network Plug] Attaching Network Plug")        
        self.data_queue = queue
        self.state_ptr = state_ptr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:"+str(port))
        self.running = True
        message = None
        print("[Network Plug] Waiting to connect...")
        # waiting for the response from the unity simulator
        while message is None and self.running:
            message = self.socket.recv()
            if message:
                print("[Network Plug] Connected to {}.".format(message.decode("utf-8")))
                self.socket.send(b"ack_reply")
        # start the thread to send the current state of the drone to simulator
        self.start_thread()

    def send_state(self):
        while self.running:
            ack = self.socket.recv()
            if ack:
                data = np.copy(self.state_ptr[0])
                data[3:6] = self.state_ptr[1][0:3]
                out = data.astype(np.float64).tostring()
                self.socket.send(out)

    def start_thread(self):
        self.thread_object = threading.Thread(target=self.send_state)
        self.thread_object.start()

    def close(self):
        if self.running:
            print("[Network Plug] Closing Connection....")
            self.running = False
            self.thread_object.join()
            self.context.destroy()      