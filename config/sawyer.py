from util import str2bool


def add_arguments(parser):
    """
    Adds a list of arguments to argparser for the reacher environment.
    """
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--kp", type=float, default=150.)
    parser.add_argument("--kd", type=float, default=20.)
    parser.add_argument("--ki", type=float, default=0.1)


def get_default_config():
    """
    Gets default configurations for the reacher environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Default Configuration for Reacher Environment")
    add_argument(parser)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", type=str2bool, default=False)

    config = parser.parse_args([])
    return config

