#!/usr/bin/env python3
from collections import deque
import logging
#import inspect

class History():

    def __init__(self, max_length=10):
        self.max_length = max_length
        self.undo_queue = deque([], max_length)
        self.redo_queue = deque([], max_length)
        self.original = None  # keep original image as loaded
        self.toggle_original = False  # toggle state
#        self.log = []

    def __repr__(self):
        undo = [i['func_name'] for i in self.undo_queue]
        redo = [i['func_name'] for i in self.redo_queue]
        return "undo:" + str(undo) + "\nredo:" +  str(redo)


    def add(self, arr,  func_name):
        ''' add array to history, discard redo '''
        if self.max_length == 0:
            logging.debug("history disabled")
            return

        self.undo_queue.append({'func_name' : func_name, 'arr' : arr.copy() })
        self.redo_queue.clear()  # discard redo queue
        logging.debug(f"added to history: {func_name}, len:{len(self.undo_queue)}")
        logging.info(self)
#        self.log.append(func_name)  # save caller function name to history
#        logging.debug(f"modification log: {self.log}")

    def undo(self):
        ''' get last array from history and move it to redo '''

        if len(self.undo_queue) <= 1:
            logging.debug("nothing to undo")
            return
        current = self.undo_queue.pop()
        self.redo_queue.append(current)

        previous = self.undo_queue[-1]
        logging.debug(f"undone {current['func_name']}, \
                                undo queue len: {len(self.undo_queue)}")
        logging.info(self)

        return previous

    def last(self):
        ''' get last array from history and leave it there '''
        if len(self.undo_queue) > 1:
            return self.undo_queue[-1]

    def redo(self):
        ''' get last array from redo '''
        if len(self.redo_queue) == 0:
            logging.debug("nothing to redo")
            return
        next_item = self.redo_queue.pop()
        self.undo_queue.append(next_item)  # deep copy needed?
        logging.debug(f"redo queue len: {len(self.redo_queue)}")
        logging.info(self)

        return next_item
