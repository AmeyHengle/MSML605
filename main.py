from log_tracking import setup_run_logging


def main() -> None:
    logger = setup_run_logging("main")
    logger.info("Hello from ml605-project!")


if __name__ == "__main__":
    main()
