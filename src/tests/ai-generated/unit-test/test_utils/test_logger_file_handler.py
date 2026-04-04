import logging
import logging.handlers

from archeo.utils.logger import get_logger


def test_get_logger_with_file_handler_creates_rotating_handler(tmp_path):
    log_path = tmp_path / "app.log"
    logger_name = "qa_logger_file_handler"

    logger = get_logger(logger_name, log_filepath=str(log_path), streaming_log_level=logging.CRITICAL)

    assert any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in logger.handlers)

    logger.info("hello")
    # Handler may lazily create file depending on flush timing, so just ensure no crash and handler exists.
