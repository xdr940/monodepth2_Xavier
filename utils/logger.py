from blessings import Terminal
import sys


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, ):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.t = Terminal()

    def write(self, string,location=(0,7)):
        with self.t.location(*location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return



