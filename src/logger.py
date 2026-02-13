import logging


# def setup_logging(log_file="logging.log"):
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode="a"),
#             logging.StreamHandler()
#         ]
#     )


# def log_metrics(logger, stage, model_name, metrics):
#     logger.info("=" * 40)
#     logger.info(f"Stage : {stage}")
#     logger.info(f"Model : {model_name}")

#     for k, v in metrics.items():
#         logger.info(f"{k.upper():<6}: {v:.4f}")

#     logger.info("=" * 40)    
    
    ###############################################
    
def setup_logging(log_file="logging.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )


def log_metrics(logger, stage, model_name, metrics):
    logger.info("=" * 40)
    logger.info(f"Stage : {stage}")
    logger.info(f"Model : {model_name}")

    for k, v in metrics.items():
        logger.info(f"{k.upper():<6}: {v:.4f}")

    logger.info("=" * 40)