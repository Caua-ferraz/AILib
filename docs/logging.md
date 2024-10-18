# Logging

Effective logging is essential for monitoring, debugging, and maintaining AI applications. **AILib** integrates Python's built-in `logging` module, providing a flexible and configurable logging system tailored to your project's needs. This section covers how to set up and utilize logging within AILib.

## Overview

AILib's logging system allows you to:

- **Monitor** the progress and status of your AI workflows.
- **Debug** issues by tracking detailed execution information.
- **Customize** logging levels and formats based on your requirements.

## Setting Up Logging

AILib uses the `Config` class to manage logging settings. To set up logging, follow these steps:

1. **Define Logging Settings in Configuration**

   Ensure your `config.json` includes a `logging` section:

   ```json
   {
       "logging": {
           "level": "INFO",
           "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
       }
   }
   ```

2. **Initialize Logging**

   Use the `setup_logging` function to configure logging based on your settings.

   ```python
   from ailib import Config, setup_logging

   # Load configuration
   config = Config.load_config('config.json')

   # Set up logging
   setup_logging(config)
   ```

## Logging Levels

AILib supports standard logging levels:

- **DEBUG**: Detailed information, typically of interest only when diagnosing problems.
- **INFO**: Confirmation that things are working as expected.
- **WARNING**: An indication that something unexpected happened, or indicative of some problem.
- **ERROR**: Due to a more serious problem, the software has not been able to perform some function.
- **CRITICAL**: A very serious error, indicating that the program itself may be unable to continue running.

### Example: Changing Logging Level

To change the logging level to `DEBUG`, update your configuration:

```json
{
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## Using the Logger

After setting up logging, you can use loggers throughout your application.

```python
import logging
from ailib import AIModel, AILibError

# Get a logger instance
logger = logging.getLogger(__name__)

try:
    # Code that may raise exceptions
    pass
except AILibError as e:
    logger.error(f"AILib Error: {e}")
except Exception as e:
    logger.exception(f"Unexpected Error: {e}")
```

### Example Output

```
2023-10-01 12:00:00,000 - __main__ - INFO - Starting model training.
2023-10-01 12:00:05,000 - __main__ - INFO - Model training completed.
```

## Advanced Logging Configuration

For more advanced logging setups, such as logging to files or integrating with external logging systems, you can extend the `setup_logging` function or configure loggers as needed.

### Logging to a File

```python
import logging
from ailib import Config, setup_logging, AILibError

# Modify the logging setup to include a file handler
def setup_logging_with_file(config: Config, log_file: str):
    try:
        logging.basicConfig(
            level=config.logging.get("level", "INFO"),
            format=config.logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging is set up with file handler.")
    except Exception as e:
        raise AILibError(f"Setting up logging with file failed: {e}")

# Usage
config = Config.load_config('config.json')
setup_logging_with_file(config, 'ailib.log')
```

### Integrating with External Logging Systems

You can integrate AILib's logging with external systems (e.g., Logstash, Sentry) by adding appropriate handlers to the logging configuration.

```python
import logging
from logging.handlers import SMTPHandler
from ailib import Config, setup_logging, AILibError

def setup_logging_with_email(config: Config):
    try:
        logging.basicConfig(
            level=config.logging.get("level", "INFO"),
            format=config.logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger = logging.getLogger(__name__)

        # Configure email handler
        mail_handler = SMTPHandler(
            mailhost=("smtp.example.com", 587),
            fromaddr="error@ailib.com",
            toaddrs=["admin@example.com"],
            subject="AILib Error Notification",
            credentials=("user", "password"),
            secure=()
        )
        mail_handler.setLevel(logging.ERROR)
        logger.addHandler(mail_handler)

        logger.info("Logging is set up with email handler.")
    except Exception as e:
        raise AILibError(f"Setting up logging with email handler failed: {e}")

# Usage
config = Config.load_config('config.json')
setup_logging_with_email(config)
```

## Best Practices

- **Consistent Logging**: Use consistent logging levels and messages across your application to maintain clarity.
- **Sensitive Information**: Avoid logging sensitive information such as passwords or personal data.
- **Performance Considerations**: Be mindful of the performance implications of extensive logging, especially in large-scale applications.
- **Log Rotation**: Implement log rotation to manage log file sizes and prevent disk space issues.

## Next Steps

- Explore [Error Handling](error_handling.md) to manage and respond to exceptions effectively.
- Learn how to integrate logging with [Configuration Management](configuration.md) for streamlined settings management.
