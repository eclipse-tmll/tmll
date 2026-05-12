"""
This module contains the Logger class which is used to log messages to the console.

Methods:
    - info: Log an info message to the console.
    - error: Log an error message to the console.
    - warning: Log a warning message to the console.
"""

from loguru import logger
import sys


class Logger:
    logger.remove()
    logger.add(sys.stderr, colorize=True, format="{message}")

    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose

    def __log_message(self, tag: str, message: str, color: str) -> None:
        message = f"<bold><{color}>[{tag}]</{color}> <green>{self.name}:</green></bold> {message}"
        logger.opt(colors=True).info(message)

    def info(self, message: str) -> None:
        self.__log_message("INFO", message, "blue")

    def error(self, message: str) -> None:
        self.__log_message("ERROR", message, "red")

    def warning(self, message: str) -> None:
        self.__log_message("WARNING", message, "yellow")
