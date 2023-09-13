import random
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np


class Packet():
    def __init__(self, id: int, t_arr: float):
        """
        Packet
        ---
        Class modeling packets being processed by the queuing system.

        ### Input parameters
        - id: packet ID (integer unique identifier) # FIXME: maybe useless (ID corresponds to position in list?)
        - t_arr: arrival time
        """
        self.t_arr = t_arr
        self.id = id


"""
Event scheduling

Event syntax:
- arrival:
- end of service:
"""


class Queue():
    def __init__(self, n_servers: int, arr_rate: float, serv_rate: float, queue_len: int | None = None):
        """
        Queue
        ---
        This class models M/M/m queues
        """
        self.n_servers = n_servers
        self.being_served = [-1] * self.n_servers
        self.arr_rate = arr_rate
        self.serv_rate = serv_rate
        self.n_packets = 0
        self.packets_list = []
        self.pkt_id = 0

        self.serv_occupied = [0] * n_servers
        self.queue_len = queue_len

        ############
        # N. arrivals eval. at time t - elem (t, n)
        self.n_arr = 0
        self.n_arr_time = []
        # N. departures eval. at time t - elem (t, n)
        self.n_dep = 0
        self.n_dep_time = []
        # Measurements of system delays for each packet that departs - (t, T_i)
        self.sys_delay = []

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
            # Add packet to the queue
            self.n_packets += 1
            self.packets_list.append(Packet(id=self.pkt_id, t_arr=curr_time))

            self.n_arr += 1
            self.n_arr_time.append((curr_time, self.n_arr))

            if self.n_packets <= self.n_servers:
                # If free servers, start services
                serv_index = self.being_served.index(-1)
                # Add the id of the packet being served
                self.being_served[serv_index] = self.pkt_id
                serv_time = random.expovariate(self.serv_rate)

                new_event = (curr_time + serv_time,
                             ["end of serv", serv_index, self.pkt_id])
                event_set.put(new_event)

            self.pkt_id += 1

            # Schedule next arrival
            inter_arr_time = random.expovariate(self.arr_rate)
            new_event = (curr_time + inter_arr_time, ["arrival"])
            event_set.put(new_event)

    def departure(self, curr_time: float, event_set: PriorityQueue, args: list):
        """
        departure
        ---
        Perform the operations needed at a departure (end of service), i.e., 
        remove the packet that is leaving the system and possibly place another
        in service.

        Input parameters:
        - curr_time: current time, extracted from the event in the FES.
        - event_set: (priority queue) future event set (for scheduling). Used to place
        the next scheduled departure.
        - args: list of parameters contained in the event (extracted from event set)
        """
        server_id = args[1]
        packet_id = args[2]

        # Extract packet from list 'self.packets_list'
        i = 0
        while i < len(self.packets_list) and self.packets_list[i].id != packet_id:
            i += 1
        if self.packets_list[i].id != packet_id:
            raise ValueError(
                f"The packet with the provided id ({packet_id}) is not in the queue!")

        finished_pkt = self.packets_list.pop(i)

        # Update measurements
        self.n_dep += 1
        self.n_dep_time.append((curr_time, self.n_dep))

        self.sys_delay.append((curr_time, curr_time - finished_pkt.t_arr))

        self.n_packets -= 1

        # Make server idle
        assert self.being_served[server_id] == packet_id, f"The server is not serving the right client according to 'self.being_served'"
        self.being_served[server_id] = -1

        # If there are still elements in the queue, try to serve a new pkt
        if self.n_packets >= self.n_servers:
            # Find packet to be served
            j = 0
            while j < len(self.packets_list) and self.packets_list[j].id in self.being_served:
                j += 1
            if j < self.n_packets:
                if self.packets_list[j].id in self.being_served:
                    raise ValueError(
                        f"Should be able to serve a new packet, but no good packet was found!")

                # Add the id of the packet being served - it will
                # be served by the server that just became free
                self.being_served[server_id] = self.packets_list[j].id
                serv_time = random.expovariate(self.serv_rate)

                new_event = (curr_time + serv_time,
                             ["end of serv", server_id, self.packets_list[j].id])
                event_set.put(new_event)

    def print_results(self, sim_time: float):
        """
        print_results
        ---
        Evaluate and print the results at the end of the simulation.

        The returned plots are:
        - N_t vs. t (time average of the number of packets in the system)
        - T_t vs. t (time average of the system delay)

        Additionally, the function evaluates the empirical average arrival rate 
        via Little's theorem (over the entire run), and compares it with the 
        actual arrival rate used.
        """
        # Time average of the number of packets:
        N_tau = []      # Format: (t, n)
        i = -1
        j = -1
        t_curr = 0
        while i < len(self.n_arr_time) - 1 and j < len(self.n_dep_time) - 1:
            # Find lowest
            if self.n_arr_time[i + 1][0] - t_curr < self.n_dep_time[j + 1][0] - t_curr:
                # Next is an arrival
                i += 1
                curr_alpha = self.n_arr_time[i][1]
                curr_beta = self.n_arr_time[j][1] if j >= 0 else 0
                t_curr = self.n_arr_time[i][0]
                N_tau.append((t_curr, curr_alpha - curr_beta))
            elif self.n_arr_time[i + 1][0] - t_curr > self.n_dep_time[j + 1][0] - t_curr:
                # Next is a departure
                j += 1
                curr_alpha = self.n_arr_time[i][1]
                curr_beta = self.n_arr_time[j][1]
                t_curr = self.n_dep_time[j][0]
                N_tau.append((t_curr, curr_alpha - curr_beta))
            else:
                # Two events at the same time - should not happen
                # The number of packets does not change
                pass

        # Average number of packets in the queue
        avg_N = []
        Ndeltat = 0
        for i in range(1, len(N_tau)):
            Ndeltat += N_tau[i - 1][1] * (N_tau[i][0] - N_tau[i - 1][0])
            avg_N.append((N_tau[i][0], Ndeltat/N_tau[i][0]))
        avg_N_np = np.array(avg_N)

        plt.figure()
        plt.plot(avg_N_np[:, 0], avg_N_np[:, 1], 'b')
        plt.title("Time average of the number of packets in the queue")
        plt.xlabel("t")
        plt.ylabel("N_tau")
        plt.grid()
        # plt.show()

        # Average system delay
        avg_T = np.zeros((len(self.sys_delay) - 1, 2))
        sum_Ti = 0
        prev_timestamp = self.sys_delay[0][0]
        for i in range(1, len(self.sys_delay)):
            timestamp = self.sys_delay[i][0]
            sum_Ti += self.sys_delay[i][1]
            avg_T[i - 1, 0] = timestamp
            avg_T[i - 1, 1] = sum_Ti / i
            prev_timestamp = timestamp

        plt.figure()
        plt.plot(avg_T[:, 0], avg_T[:, 1], 'r')
        plt.title("Time average of the system delay")
        plt.xlabel("t")
        plt.ylabel("T_t")
        plt.grid()
        plt.show()

        # Little's theorem - evaluate on the last samples
        print(
            f"Empirical result: lambda_t = {avg_N_np[-1, 1] / avg_T[-1, 1]}\nActual arrival rate: {self.arr_rate}")


def main(sim_time: float, arr_rate: float, serv_rate: float, n_serv: int):
    """
    Main program loop.

    # Input parameters
    - sim_time: total simulation time (seconds)
    - arr_rate: lambda (in 1/s)
    - serv_rate: mu (in 1/s)
    - n_serv: number of servers (the queue is M/M/n_serv)
    """

    event_set = PriorityQueue()

    queue = Queue(n_serv, arr_rate, serv_rate)

    time = 0

    # Schedule an arrival to start simulation
    event_set.put((time, ["arrival"]))

    while time < sim_time:
        # Extract next event
        (time, event_args) = event_set.get()

        if event_args[0] == "arrival":
            # queue.events.append((time, "arrival"))
            queue.arrival(time, event_set)
        elif event_args[0] == "end of serv":
            # queue.events.append((time, "end of serv"))
            queue.departure(time, event_set, event_args)

    # Perform evaluations and plots
    queue.print_results(time)


if __name__ == "__main__":
    random.seed(660603047)
    # Perform simulation
    main(3600, 5, 10, 2)
