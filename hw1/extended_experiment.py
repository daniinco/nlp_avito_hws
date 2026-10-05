import argparse

import experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--seconds', type=int, default=1500)
    args = parser.parse_args()
    experiment.solution.MAX_TRAINING_TIME_SECONDS = args.seconds
    experiment.train(args.config)
