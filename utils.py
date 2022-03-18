import logging
from torchsummary import summary


class Logger:
    def __init__(self, name=__name__):
        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.DEBUG)

        # create a handler, print log info to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # define format
        formatter = logging.Formatter('%(asctime)s %(filename)s-[line:%(lineno)d]'
                                      '-%(levelname)s-[%(name)s]: %(message)s',
                                      datefmt='%a, %d %b %Y %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        return self.logger


def model_info(model):
    summary(model, (3, 448, 448), batch_size=1, device="cpu")


if __name__ == '__main__':
    log = Logger('model').get_log
    log.info('this is a test')
    logging.info(123)
