from config import Configs
import logging

configs = Configs()

async def initialize_configs():
    await configs.initialize()
    logging.info("configs initialized")