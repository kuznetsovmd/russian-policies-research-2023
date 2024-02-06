import argparse
import sys

from src.preprocess import main as preprocess_main
from src.models import main as models_main
from src.coherence import main as coherence_main
from src.excerpts import main as excerpts_main
from src.experiments import main as experiments_main


def main(args):
    if args.preprocess or args.all:
        preprocess_main()
    if args.models or args.all:
        models_main()
    if args.coherence or args.all:
        coherence_main()
    if args.excerpts or args.all:
        excerpts_main()
    if args.experiments or args.all:
        experiments_main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='privacy-ontology',
        description='Command line tool to control ontology framework'
    )

    parser.add_argument('-e', '--excerpts', default=False, action='store_true')
    parser.add_argument('-x', '--experiments', default=False, action='store_true')
    parser.add_argument('-c', '--coherence', default=False, action='store_true')
    parser.add_argument('-m', '--models', default=False, action='store_true')
    parser.add_argument('-p', '--preprocess', default=False, action='store_true')
    parser.add_argument('-a', '--all', default=False, action='store_true')
    args = parser.parse_args()

    try:
        main(args)
        sys.exit(0)
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(130)
