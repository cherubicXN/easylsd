# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys


def setup_logger(name, save_dir, out_file='log.txt', json_format=False, rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if json_format:
        try:
            from pythonjsonlogger import jsonlogger  # lazy import
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s: %(message)s"
            )
        except Exception as e:
            raise ImportError(
                "python-json-logger is required for json_format=True. Install it via 'pip install python-json-logger'."
            ) from e
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s"
        )

    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, out_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
