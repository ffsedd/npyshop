#!/usr/bin/env python3
from collections import deque
import logging


class History():

    def __init__(self, max_length=10):
        self.max_length = max_length
        self.undo_queue = deque([], max_length)
        self.redo_queue = deque([], max_length)
        self.original = None  # keep original image as loaded
        self.toggle_original = False  # toggle state

    def add(self, arr):
        ''' add array to history, discard redo '''
        if self.max_length > 0:
            self.undo_queue.append(arr.copy())
            self.redo_queue.clear()
            logging.debug(f"added to history, len:{len(self.undo_queue)}")

    def undo(self):
        ''' get last array from history and move it to redo '''
        if len(self.undo_queue) > 1:
            last = self.undo_queue.pop()
            self.redo_queue.append(last)
            previous = self.undo_queue[-1]
            logging.debug(f"undo queue len: {len(self.undo_queue)}")
            return previous

    def last(self):
        ''' get last array from history and leave it there '''
        if len(self.undo_queue) > 1:
            return self.undo_queue[-1]

    def redo(self):
        ''' get last array from redo '''
        if len(self.redo_queue) > 0:
            next_arr = self.redo_queue.pop()
            self.undo_queue.append(next_arr.copy())
            logging.debug(f"redo queue len: {len(self.redo_queue)}")
            return next_arr
