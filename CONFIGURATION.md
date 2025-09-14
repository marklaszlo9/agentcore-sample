# Agent Response Logging Configuration

This document describes how to configure the agent response logging system using environment variables.

## Overview

The agent response logging system supports configuration through environment variables, allowing you to customize logging behavior without modifying code. The system includes automatic validation and consistency adjustments.

## Environment Variables

### Logging Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `AGENT_LOGGING_ENABLED` | boolean | `true` | Enable/disable agent response logging |
| `AGENT_LOG_LEVEL` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `AGENT_LOGGING_INCLUDE_RESPONSE_CONTENT` | boolean | `false` | Include full response content in logs |
| `AGENT_LOGGING_INCLUDE_USER_FACING` | boolean | `false` | Include user-facing information in logs |
| `AGENT_LOGGING_STRUCTURED_FORMAT` | boolean | `true` | Use structured JSON log format |
| `AGENT_LOG_GROUP_NAME` | string | `/bedrockagent/agent-responses` | CloudWatch log group name |
| `AGENT_LOGGING_USE_CLOUDWATCH` | boolean | `true` | Enable CloudWatch logging |
| `AGENT_LOGGING_CONSOLE_FALLBACK` | boolean | `true` | Fall back to console if CloudWatch fails |
| `AGENT_LOGGING_ASYNC` | boolean | `true` | Enable asynchronous logging |
| `AGENT_LOGGING_BATCH_SIZE` | integer | `10` | Number of log entries per batch |
| `AGENT_LOGGING_BATCH_TIMEOUT` | float | `5.0` | Batch timeout in seconds |
| `AGENT_LOGGING_MAX_QUEUE_SIZE` | integer | `1000` | Maximum queue size for async logging |

### User-Facing Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `AGENT_SHOW_INFO` | boolean | `false` | Show agent information in responses |
| `AGENT_SHOW_NAME` | boolean | `true` | Show agent name |
| `AGENT_SHOW_TYPE` | boolean | `true` | Show agent type |
| `AGENT_SHOW_MODEL` | boolean | `false` | Show model information |
| `AGENT_SHOW_EXECUTION_TIME` | boolean | `false` | Show execution time |
| `AGENT_FORMAT_STYLE` | string | `bracket` | Format style (bracket, prefix, suffix, none) |
| `AGENT_CUSTOM_NAMES` | JSON | `{}` | Custom agent display names |

## Boolean Values

Boolean environment variables accept the following values:

**True values:** `true`, `True`, `TRUE`, `1`, `yes`, `on`, `enabled`
**False values:** `false`, `False`, `FALSE`, `0`, `no`, `off`, `disabled`

## Custom Agent Names

The `AGENT_CUSTOM_NAMES` environment variable should contain a JSON object mapping internal agent names to user-friendly display names:

```bash
export AGENT_CUSTOM_NAMES='{"knowledge_agent": "Envision Expert", "general_agent": "Sustainability Advisor"}'
```

## Usage Examples

### Basic Configuration

```bash
# Enable logging with DEBUG level
export AGENT_LOGGING_ENABLED=true
export AGENT_LOG_LEVEL=DEBUG

# Show agent information to users
export AGENT_SHOW_INFO=true
export AGENT_FORMAT_STYLE=bracket
```

### Production Configuration

```bash
# Production logging setup
export AGENT_LOGGING_ENABLED=true
export AGENT_LOG_LEVEL=INFO
export AGENT_LOGGING_USE_CLOUDWATCH=true
export AGENT_LOG_GROUP_NAME=/production/agent-responses
export AGENT_LOGGING_BATCH_SIZE=50
export AGENT_LOGGING_BATCH_TIMEOUT=10.0

# Hide agent info from users in production
export AGENT_SHOW_INFO=false
```

### Development Configuration

```bash
# Development logging setup
export AGENT_LOGGING_ENABLED=true
export AGENT_LOG_LEVEL=DEBUG
export AGENT_LOGGING_USE_CLOUDWATCH=false
export AGENT_LOGGING_CONSOLE_FALLBACK=true
export AGENT_LOGGING_INCLUDE_RESPONSE_CONTENT=true

# Show detailed agent info for debugging
export AGENT_SHOW_INFO=true
export AGENT_SHOW_MODEL=true
export AGENT_SHOW_EXECUTION_TIME=true
export AGENT_FORMAT_STYLE=prefix
```

## Programmatic Usage

### Using ConfigurationManager

```python
from agent_logging import ConfigurationManager, ConfigurationError

try:
    # Load configuration from environment
    logging_config = ConfigurationManager.load_logging_config()
    user_facing_config = ConfigurationManager.load_user_facing_config()
    
    print(f"Logging enabled: {logging_config.enabled}")
    print(f"Log level: {logging_config.log_level}")
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Using with AgentResponseLogger

```python
from agent_logging import AgentResponseLogger, ConfigurationError

try:
    # Create logger from environment configuration
    logger = AgentResponseLogger.from_environment()
    
    # Or create with default configuration (loads from environment)
    logger = AgentResponseLogger()
    
except ConfigurationError as e:
    print(f"Logger configuration error: {e}")
    # Fall back to default configuration
    logger = AgentResponseLogger(LoggingConfig())
```

### Configuration Validation

```python
from agent_logging import ConfigurationManager

# Validate current environment configuration
errors = ConfigurationManager.validate_environment()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
else:
    print("Configuration is valid")

# Get configuration summary
summary = ConfigurationManager.get_configuration_summary()
print(f"Environment variables set: {len(summary['environment_variables'])}")
print(f"Validation errors: {len(summary['validation_errors'])}")
```

## Configuration Validation

The system automatically validates configuration values and provides helpful error messages:

- **Log levels** must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Batch size** must be between 1 and 10,000
- **Batch timeout** must be between 0.1 and 300 seconds
- **Queue size** must be between 1 and 100,000
- **Format style** must be one of: bracket, prefix, suffix, none
- **CloudWatch log group names** must start with '/' and be under 512 characters

## Consistency Adjustments

The system automatically adjusts configuration for consistency:

- If logging is disabled, async logging and CloudWatch are automatically disabled
- Invalid combinations are corrected with warnings logged

## Error Handling

Configuration errors are handled gracefully:

- Invalid environment variables raise `ConfigurationError` with descriptive messages
- If environment loading fails, the system falls back to default configuration
- Validation errors include suggestions for valid values

## Testing Configuration

Use the provided demo script to test your configuration:

```bash
python examples/configuration_management_demo.py
```

This script demonstrates all configuration features and validates your environment setup.