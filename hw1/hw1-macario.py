import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue


class Queue():
    def __init__(self, n_servers: int, arr_rate: float, serv_rate: float, queue_len=None):
        """
        Queue
        ---
        This class models M/M/m queues
        """
        self.n_servers = n_servers
        self.arr_rate = arr_rate
        self.serv_rate = serv_rate
        self.n_packets = 0
        self.serv_occupied = [0] * n_servers
        self.queue_len = queue_len

        self.last_event_time = 0
        # List containing (time, n_packets) values (sampled at n_packets variations)
        self.N_t = []
        # Cumulative sum of N_t*DeltaT, where DeltaT is the time distance between N_t variations
        self.ut = 0

    def arrival(self, curr_time: float, event_set: PriorityQueue):
        """
        arrival
        ---
        Perform operations needed at arrival.
        The packet is added to the queuing system.

        The method also schedules the next arrival in the queue.

        Input parameters:
        - time: current time.
        - event_set: (priority queue) future event set (for scheduling). Used to place
        the next scheduled arrival.
        """
        if self.queue_len is None:
            self.n_packets += 1

    def departure(self, time: float, event_set: PriorityQueue):
        """
        departure
        ---
        Perform the operations needed at a departure (end of service), i.e., 
        remove the packet that is leaving the system and possibly place another
        in service.

        Input parameters:
        - time: current time, extracted from the event in the FES.
        - event_set: (priority queue) future event set (for scheduling). Used to place
        the next scheduled departure.
        """
