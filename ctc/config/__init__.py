import argparse


class ApplicationArguments():
    def __init__(self):
        arguments = self._parseArgumentsFromCommandLine()
        self.trainDataset = arguments.trainDataset
        self.validationDataset = arguments.validationDataset
        self.testDataset = arguments.testDataset
        print('Initializing application with', vars(self))

    def _parseArgumentsFromCommandLine(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-tr', '--trainDataset', metavar='trainDataset', dest='trainDataset',
                            help='Train dataset csv path', default='data/local/librivox-train-clean-1-wav.csv')
        parser.add_argument('-v', '--validationDataset', metavar='validationDataset', dest='validationDataset',
                            help='Validation dataset csv path', default='data/local/librivox-dev-clean-wav.csv')
        parser.add_argument('-te', '--testDataset', metavar='testDataset', dest='testDataset',
                            help='Test dataset csv path', default='data/local/librivox-test-clean-wav.csv')
        return parser.parse_args()
