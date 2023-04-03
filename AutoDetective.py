'''
Main program for autoDetective
launches falsifier and simulator in different threads
'''
from queue import Queue
import time
import logging

from multiprocessing import Process, Manager

import falsifier
import simulator
import data_logging


GLOBAL_LOGGING_LEVEL = logging.DEBUG
data_logging.LOGGING_LEVEL = GLOBAL_LOGGING_LEVEL


def launch_falsifier(log_queue: Queue):
    falsifier.main(log_queue)


def launch_simulator(log_queue: Queue):
    simulator.main(log_queue)


def log_consumer(log_queue: Queue):
    while True:
        item = log_queue.get()
        print(item)
# TODO: Implement Queue Logging Consumer thread that feeds to a file for now?


if __name__ == '__main__':
    manager = Manager()
    log_queue = manager.Queue()
    logger = data_logging.set_queue_logger('AutoDetective', log_queue)

    falsifier_thread = Process(target=launch_falsifier,
                               name='Falsifier-Process',
                               args=(log_queue,),
                               )
    simulator_thread = Process(target=launch_simulator,
                               name='Simulator-Process',
                               args=(log_queue,),
                               )
    log_consumer_thread = Process(target=log_consumer,
                                  name='Log-Consumer',
                                  args=(log_queue,),
                                  )
    logger.debug('Launching Instance of Falsifier')
    falsifier_thread.start()
    logger.debug('Launching Instance of Simulator')
    simulator_thread.start()
    logger.debug('Launching Instance of Log Conusmer')
    log_consumer_thread.start()

    threads = [falsifier_thread, simulator_thread, log_consumer_thread]
    logger.debug([f"State : {t.exitcode}, Name : {t.name}" for t in threads])
    while all([t.exitcode is None for t in threads]):
        logger.debug([f"State : {t.exitcode}, Name : {t.name}" for t in threads])
        time.sleep(1)

    logger.error('Killing all threads since one failed')
    logger.debug([f"State : {t.exitcode}, Name : {t.name}" for t in threads])
    # time.sleep(3)
    [t.terminate() for t in threads]
