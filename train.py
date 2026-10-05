import argparse

from spam_detector.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the local spam detector."
    )
    parser.add_argument(
        "--data", help="Optional ham/spam labeled TSV; otherwise download UCI data."
    )
    train(parser.parse_args().data)
