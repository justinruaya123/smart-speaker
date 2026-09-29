"""Train the streaming VCM from the frozen, explicitly selected configuration."""


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from vcm.training import train
    train(args.config, args.output)


if __name__ == '__main__':
    main()
