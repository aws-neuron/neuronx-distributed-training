# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import time


class Throughput:
    def __init__(self, moving_avg_window_size):
        self.seqs_per_iteration = None  # batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()
        self.throughput_peak = 0
        self.throughput_sum = 0
        self.throughputs = []

    def set_seqs_per_iteration(self, batch_size, world_size, grad_accum_usteps):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        self.throughputs.append(throughput)
        return throughput
