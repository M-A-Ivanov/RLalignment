import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Event(FileSystemEventHandler):
    def __init__(self, observer, function):
        super(Event, self).__init__()
        self.function = function
        self.observer = observer

    def on_created(self, event):
        self.function(event.src_path)


class Fluffy:
    def __init__(self, function, path):
        self.observer = Observer()
        self.event_handler = Event(self.observer, function)
        self.observer.schedule(self.event_handler, path, recursive=True)

    def SniffAround(self):
        self.observer.start()



