"""
Agent Response Logging Module

This module provides structured logging capabilities for tracking agent responses
in the multi-agent orchestrator system. It includes data models for agent information,
response data, and logging configuration, as well as the core AgentResponseLogger class.
"""

import json
import logging
import os
import asyncio
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import boto3
from watchtower import CloudWatchLogHandler


@dataclass
class AgentInfo:
    """Data model for agent identification and metadata"""
    agent_name: str
    agent_type: str  # "orchestrator", "knowledge", "general", "custom"
    model_id: str
    session_id: str
    timestamp: datetime
    execution_time_ms: int


@dataclass
class ResponseData:
    """Data model for agent response information"""
    query: str
    response: str
    response_length: int
    success: bool
    metadata: Dict[str, Any]


@dataclass
class RoutingInfo:
    """Data model for orchestrator routing decisions"""
    selected_agent: str
    routing_reasoning: str
    available_agents: list
    confidence_score: Optional[float] = None


@dataclass
class LoggingConfig:
    """Configuration for agent response logging"""
    enabled: bool = True
    log_level: str = "INFO"
    include_response_content: bool = False
    include_user_facing: bool = False
    structured_format: bool = True
    log_group_name: str = "/bedrockagent/agent-responses"
    use_cloudwatch: bool = True
    console_fallback: bool = True
    # New async and batching configuration
    async_logging: bool = True
    batch_size: int = 10
    batch_timeout_seconds: float = 5.0
    max_queue_size: int = 1000
    # Circuit breaker and retry configuration
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_exponential_base: float = 2.0


@dataclass
class UserFacingConfig:
    """Configuration for user-facing agent identification"""
    show_agent_info: bool = False
    show_agent_name: bool = True
    show_agent_type: bool = True
    show_model_info: bool = False
    show_execution_time: bool = False
    format_style: str = "bracket"  # "bracket", "prefix", "suffix", "none"
    custom_agent_names: Dict[str, str] = field(default_factory=dict)  # Map internal names to user-friendly names
    
    def __post_init__(self):
        if not self.custom_agent_names:
            self.custom_agent_names = {
                "knowledge_agent": "Envision Framework Specialist",
                "general_sustainability_agent": "Sustainability Expert", 
                "orchestrator": "Agent Coordinator",
                "custom_envision_agent": "Envision Assistant",
                "multi_agent_orchestrator": "Multi-Agent System"
            }


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors"""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class LoggingCircuitBreaker:
    """
    Circuit breaker pattern implementation for logging failures
    
    Prevents cascading failures by temporarily disabling CloudWatch logging
    when it's consistently failing, allowing fallback to console logging.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result if successful
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - CloudWatch logging disabled")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.state
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitBreakerState.CLOSED


class RetryHandler:
    """
    Exponential backoff retry handler for CloudWatch operations
    
    Implements retry logic with exponential backoff for transient failures
    in CloudWatch logging operations.
    """
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 30.0, exponential_base: float = 2.0):
        """
        Initialize retry handler
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds between retries
            exponential_base: Base for exponential backoff calculation
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute_with_retry(self, func, *args, **kwargs):
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result if successful
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt failed, re-raise the exception
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Log retry attempt
                retry_logger = logging.getLogger("retry_handler")
                retry_logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                time.sleep(delay)
        
        raise last_exception
    
    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable
        
        Args:
            error: Exception to check
            
        Returns:
            True if error should be retried, False otherwise
        """
        # Define retryable error patterns
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "throttl",
            "rate limit",
            "service unavailable",
            "internal server error",
            "502",
            "503",
            "504"
        ]
        
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in retryable_patterns)


class ConfigurationManager:
    """
    Manages loading and validation of logging configuration from environment variables
    
    Provides centralized configuration management with validation, default values,
    and environment variable integration for all logging-related settings.
    """
    
    # Environment variable mappings
    ENV_VAR_MAPPINGS = {
        # LoggingConfig mappings
        'AGENT_LOGGING_ENABLED': ('enabled', bool),
        'AGENT_LOG_LEVEL': ('log_level', str),
        'AGENT_LOGGING_INCLUDE_RESPONSE_CONTENT': ('include_response_content', bool),
        'AGENT_LOGGING_INCLUDE_USER_FACING': ('include_user_facing', bool),
        'AGENT_LOGGING_STRUCTURED_FORMAT': ('structured_format', bool),
        'AGENT_LOG_GROUP_NAME': ('log_group_name', str),
        'AGENT_LOGGING_USE_CLOUDWATCH': ('use_cloudwatch', bool),
        'AGENT_LOGGING_CONSOLE_FALLBACK': ('console_fallback', bool),
        'AGENT_LOGGING_ASYNC': ('async_logging', bool),
        'AGENT_LOGGING_BATCH_SIZE': ('batch_size', int),
        'AGENT_LOGGING_BATCH_TIMEOUT': ('batch_timeout_seconds', float),
        'AGENT_LOGGING_MAX_QUEUE_SIZE': ('max_queue_size', int),
        
        # Circuit breaker and retry configuration
        'AGENT_LOGGING_CIRCUIT_BREAKER_ENABLED': ('circuit_breaker_enabled', bool),
        'AGENT_LOGGING_CIRCUIT_BREAKER_FAILURE_THRESHOLD': ('circuit_breaker_failure_threshold', int),
        'AGENT_LOGGING_CIRCUIT_BREAKER_RECOVERY_TIMEOUT': ('circuit_breaker_recovery_timeout', int),
        'AGENT_LOGGING_RETRY_ENABLED': ('retry_enabled', bool),
        'AGENT_LOGGING_RETRY_MAX_ATTEMPTS': ('retry_max_attempts', int),
        'AGENT_LOGGING_RETRY_BASE_DELAY': ('retry_base_delay', float),
        'AGENT_LOGGING_RETRY_MAX_DELAY': ('retry_max_delay', float),
        'AGENT_LOGGING_RETRY_EXPONENTIAL_BASE': ('retry_exponential_base', float),
        
        # UserFacingConfig mappings
        'AGENT_SHOW_INFO': ('show_agent_info', bool),
        'AGENT_SHOW_NAME': ('show_agent_name', bool),
        'AGENT_SHOW_TYPE': ('show_agent_type', bool),
        'AGENT_SHOW_MODEL': ('show_model_info', bool),
        'AGENT_SHOW_EXECUTION_TIME': ('show_execution_time', bool),
        'AGENT_FORMAT_STYLE': ('format_style', str),
    }
    
    # Valid values for specific configuration options
    VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    VALID_FORMAT_STYLES = {'bracket', 'prefix', 'suffix', 'none'}
    
    @classmethod
    def load_logging_config(cls, config_dict: Optional[Dict[str, Any]] = None) -> LoggingConfig:
        """
        Load LoggingConfig from environment variables with validation
        
        Args:
            config_dict: Optional dictionary to override environment variables
            
        Returns:
            LoggingConfig instance with validated settings
            
        Raises:
            ConfigurationError: If configuration validation fails
        """
        # Start with default values
        config_values = {}
        
        # Load from environment variables
        for env_var, (field_name, field_type) in cls.ENV_VAR_MAPPINGS.items():
            if field_name in LoggingConfig.__dataclass_fields__:
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    try:
                        config_values[field_name] = cls._convert_env_value(env_value, field_type)
                    except (ValueError, TypeError) as e:
                        raise ConfigurationError(
                            f"Invalid value for {env_var}: {env_value}. Expected {field_type.__name__}. Error: {e}"
                        )
        
        # Override with provided config_dict if given
        if config_dict:
            config_values.update(config_dict)
        
        # Create LoggingConfig instance
        try:
            config = LoggingConfig(**config_values)
        except TypeError as e:
            raise ConfigurationError(f"Failed to create LoggingConfig: {e}")
        
        # Auto-adjust configuration for consistency
        config = cls._adjust_config_consistency(config)
        
        # Validate the configuration
        cls._validate_logging_config(config)
        
        return config
    
    @classmethod
    def load_user_facing_config(cls, config_dict: Optional[Dict[str, Any]] = None) -> UserFacingConfig:
        """
        Load UserFacingConfig from environment variables with validation
        
        Args:
            config_dict: Optional dictionary to override environment variables
            
        Returns:
            UserFacingConfig instance with validated settings
            
        Raises:
            ConfigurationError: If configuration validation fails
        """
        # Start with default values
        config_values = {}
        
        # Load from environment variables
        for env_var, (field_name, field_type) in cls.ENV_VAR_MAPPINGS.items():
            if field_name in UserFacingConfig.__dataclass_fields__:
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    try:
                        config_values[field_name] = cls._convert_env_value(env_value, field_type)
                    except (ValueError, TypeError) as e:
                        raise ConfigurationError(
                            f"Invalid value for {env_var}: {env_value}. Expected {field_type.__name__}. Error: {e}"
                        )
        
        # Handle custom agent names from environment
        custom_names_env = os.environ.get('AGENT_CUSTOM_NAMES')
        if custom_names_env:
            try:
                custom_names = json.loads(custom_names_env)
                if isinstance(custom_names, dict):
                    config_values['custom_agent_names'] = custom_names
                else:
                    raise ConfigurationError("AGENT_CUSTOM_NAMES must be a JSON object")
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON in AGENT_CUSTOM_NAMES: {e}")
        
        # Override with provided config_dict if given
        if config_dict:
            config_values.update(config_dict)
        
        # Create UserFacingConfig instance
        try:
            config = UserFacingConfig(**config_values)
        except TypeError as e:
            raise ConfigurationError(f"Failed to create UserFacingConfig: {e}")
        
        # Validate the configuration
        cls._validate_user_facing_config(config)
        
        return config
    
    @classmethod
    def _convert_env_value(cls, value: str, target_type: type) -> Any:
        """
        Convert environment variable string to target type
        
        Args:
            value: String value from environment variable
            target_type: Target type to convert to
            
        Returns:
            Converted value
            
        Raises:
            ValueError: If conversion fails
        """
        if target_type == bool:
            return cls._str_to_bool(value)
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return value
        else:
            raise ValueError(f"Unsupported type conversion: {target_type}")
    
    @classmethod
    def _str_to_bool(cls, value: str) -> bool:
        """
        Convert string to boolean with flexible parsing
        
        Args:
            value: String value to convert
            
        Returns:
            Boolean value
            
        Raises:
            ValueError: If string cannot be converted to boolean
        """
        value_lower = value.lower().strip()
        if value_lower in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        elif value_lower in ('false', '0', 'no', 'off', 'disabled'):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")
    
    @classmethod
    def _adjust_config_consistency(cls, config: LoggingConfig) -> LoggingConfig:
        """
        Adjust configuration for consistency and logical dependencies
        
        Args:
            config: LoggingConfig instance to adjust
            
        Returns:
            Adjusted LoggingConfig instance
        """
        # Create a copy to avoid modifying the original
        adjusted_values = asdict(config)
        
        # Auto-disable async logging if logging is disabled
        if not config.enabled and config.async_logging:
            adjusted_values['async_logging'] = False
        
        # Auto-disable CloudWatch if logging is disabled
        if not config.enabled and config.use_cloudwatch:
            adjusted_values['use_cloudwatch'] = False
        
        return LoggingConfig(**adjusted_values)
    
    @classmethod
    def _validate_logging_config(cls, config: LoggingConfig) -> None:
        """
        Validate LoggingConfig values
        
        Args:
            config: LoggingConfig instance to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate log level
        if config.log_level.upper() not in cls.VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"Invalid log level: {config.log_level}. "
                f"Valid levels: {', '.join(cls.VALID_LOG_LEVELS)}"
            )
        
        # Validate batch size
        if config.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got: {config.batch_size}")
        
        if config.batch_size > 10000:
            raise ConfigurationError(f"Batch size too large (max 10000), got: {config.batch_size}")
        
        # Validate batch timeout
        if config.batch_timeout_seconds <= 0:
            raise ConfigurationError(
                f"Batch timeout must be positive, got: {config.batch_timeout_seconds}"
            )
        
        if config.batch_timeout_seconds > 300:  # 5 minutes max
            raise ConfigurationError(
                f"Batch timeout too large (max 300s), got: {config.batch_timeout_seconds}"
            )
        
        # Validate max queue size
        if config.max_queue_size <= 0:
            raise ConfigurationError(f"Max queue size must be positive, got: {config.max_queue_size}")
        
        if config.max_queue_size > 100000:
            raise ConfigurationError(f"Max queue size too large (max 100000), got: {config.max_queue_size}")
        
        # Validate log group name format
        if config.use_cloudwatch and config.log_group_name:
            if not config.log_group_name.startswith('/'):
                raise ConfigurationError(
                    f"CloudWatch log group name must start with '/', got: {config.log_group_name}"
                )
            
            if len(config.log_group_name) > 512:
                raise ConfigurationError(
                    f"Log group name too long (max 512 chars), got: {len(config.log_group_name)}"
                )
        
        # Validate async logging dependencies (should be handled by consistency adjustment)
        if config.async_logging and not config.enabled:
            raise ConfigurationError("Async logging requires logging to be enabled")
        
        # Validate circuit breaker configuration
        if config.circuit_breaker_failure_threshold <= 0:
            raise ConfigurationError(
                f"Circuit breaker failure threshold must be positive, got: {config.circuit_breaker_failure_threshold}"
            )
        
        if config.circuit_breaker_recovery_timeout <= 0:
            raise ConfigurationError(
                f"Circuit breaker recovery timeout must be positive, got: {config.circuit_breaker_recovery_timeout}"
            )
        
        # Validate retry configuration
        if config.retry_max_attempts <= 0:
            raise ConfigurationError(
                f"Retry max attempts must be positive, got: {config.retry_max_attempts}"
            )
        
        if config.retry_base_delay <= 0:
            raise ConfigurationError(
                f"Retry base delay must be positive, got: {config.retry_base_delay}"
            )
        
        if config.retry_max_delay <= 0:
            raise ConfigurationError(
                f"Retry max delay must be positive, got: {config.retry_max_delay}"
            )
        
        if config.retry_exponential_base <= 1.0:
            raise ConfigurationError(
                f"Retry exponential base must be greater than 1.0, got: {config.retry_exponential_base}"
            )
    
    @classmethod
    def _validate_user_facing_config(cls, config: UserFacingConfig) -> None:
        """
        Validate UserFacingConfig values
        
        Args:
            config: UserFacingConfig instance to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate format style
        if config.format_style not in cls.VALID_FORMAT_STYLES:
            raise ConfigurationError(
                f"Invalid format style: {config.format_style}. "
                f"Valid styles: {', '.join(cls.VALID_FORMAT_STYLES)}"
            )
        
        # Validate custom agent names
        if config.custom_agent_names:
            if not isinstance(config.custom_agent_names, dict):
                raise ConfigurationError("Custom agent names must be a dictionary")
            
            for key, value in config.custom_agent_names.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ConfigurationError(
                        f"Custom agent names must be string->string mapping, "
                        f"got {type(key).__name__}->{type(value).__name__}"
                    )
                
                if len(key) > 100 or len(value) > 200:
                    raise ConfigurationError(
                        f"Custom agent name too long: key='{key}' ({len(key)}), "
                        f"value='{value}' ({len(value)})"
                    )
    
    @classmethod
    def get_configuration_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration from environment variables
        
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            "environment_variables": {},
            "logging_config": {},
            "user_facing_config": {},
            "validation_errors": []
        }
        
        # Check environment variables
        for env_var, (field_name, field_type) in cls.ENV_VAR_MAPPINGS.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                summary["environment_variables"][env_var] = env_value
        
        # Try to load and validate configurations
        try:
            logging_config = cls.load_logging_config()
            summary["logging_config"] = asdict(logging_config)
        except ConfigurationError as e:
            summary["validation_errors"].append(f"LoggingConfig: {e}")
        
        try:
            user_facing_config = cls.load_user_facing_config()
            summary["user_facing_config"] = asdict(user_facing_config)
        except ConfigurationError as e:
            summary["validation_errors"].append(f"UserFacingConfig: {e}")
        
        return summary
    
    @classmethod
    def validate_environment(cls) -> List[str]:
        """
        Validate current environment configuration
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            cls.load_logging_config()
        except ConfigurationError as e:
            errors.append(f"LoggingConfig validation failed: {e}")
        
        try:
            cls.load_user_facing_config()
        except ConfigurationError as e:
            errors.append(f"UserFacingConfig validation failed: {e}")
        
        return errors


class StructuredJSONFormatter(logging.Formatter):
    """
    Enhanced structured JSON formatter for agent response logs
    
    Provides consistent JSON formatting with proper field ordering,
    timestamp formatting, and error handling for non-serializable objects.
    """
    
    def __init__(self, include_extra_fields: bool = True):
        """
        Initialize the JSON formatter
        
        Args:
            include_extra_fields: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            JSON-formatted log string
        """
        try:
            # Start with basic log record fields
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger_name": record.name,
                "message": record.getMessage(),
            }
            
            # Add extra fields if enabled and present
            if self.include_extra_fields and hasattr(record, '__dict__'):
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                 'pathname', 'filename', 'module', 'lineno', 'funcName',
                                 'created', 'msecs', 'relativeCreated', 'thread', 
                                 'threadName', 'processName', 'process', 'getMessage',
                                 'exc_info', 'exc_text', 'stack_info']:
                        log_entry[key] = value
            
            return json.dumps(log_entry, default=self._json_serializer, separators=(',', ':'))
            
        except Exception as e:
            # Fallback to simple format if JSON serialization fails
            return f'{{"timestamp": "{datetime.now().isoformat()}", "level": "ERROR", "message": "JSON formatting failed: {str(e)}", "original_message": "{record.getMessage()}"}}'
    
    def _json_serializer(self, obj):
        """
        Custom JSON serializer for non-standard types
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return f"<non-serializable: {type(obj).__name__}>"


class AsyncLogBatcher:
    """
    Asynchronous log batching system for improved performance
    
    Collects log entries in batches and processes them asynchronously
    to reduce the impact on main application performance.
    """
    
    def __init__(self, config: LoggingConfig, logger: logging.Logger):
        """
        Initialize the async log batcher
        
        Args:
            config: LoggingConfig instance with batching settings
            logger: Logger instance to send batched logs to
        """
        self.config = config
        self.logger = logger
        self.log_queue = Queue(maxsize=config.max_queue_size)
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="log_batcher")
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
    
    def add_log_entry(self, log_entry: Dict[str, Any]) -> bool:
        """
        Add a log entry to the batch queue
        
        Args:
            log_entry: Dictionary containing log data
            
        Returns:
            True if entry was added successfully, False if queue is full
        """
        try:
            self.log_queue.put_nowait(log_entry)
            return True
        except:
            # Queue is full, log synchronously as fallback
            return False
    
    def _batch_processor(self):
        """
        Background thread that processes log entries in batches
        """
        while self.running:
            try:
                # Try to get log entries from queue
                current_time = time.time()
                
                # Collect entries until batch size or timeout
                while (len(self.batch_buffer) < self.config.batch_size and 
                       (current_time - self.last_flush_time) < self.config.batch_timeout_seconds):
                    try:
                        entry = self.log_queue.get(timeout=0.1)
                        self.batch_buffer.append(entry)
                    except Empty:
                        current_time = time.time()
                        continue
                
                # Flush batch if we have entries
                if self.batch_buffer:
                    self._flush_batch()
                
            except Exception as e:
                # Log batch processing errors to fallback logger
                fallback_logger = logging.getLogger("batch_processor_error")
                fallback_logger.error(f"Batch processing error: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def _flush_batch(self):
        """
        Flush the current batch of log entries
        """
        if not self.batch_buffer:
            return
            
        try:
            # Submit batch processing to thread pool
            self.executor.submit(self._process_batch, self.batch_buffer.copy())
            self.batch_buffer.clear()
            self.last_flush_time = time.time()
            
        except Exception as e:
            # If batch processing fails, try to log entries individually
            for entry in self.batch_buffer:
                try:
                    self._log_entry_sync(entry)
                except:
                    pass  # Ignore individual failures
            self.batch_buffer.clear()
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """
        Process a batch of log entries
        
        Args:
            batch: List of log entry dictionaries
        """
        for entry in batch:
            try:
                self._log_entry_sync(entry)
            except Exception as e:
                # Log individual entry failures to fallback logger
                fallback_logger = logging.getLogger("batch_entry_error")
                fallback_logger.error(f"Failed to log entry: {e}")
    
    def _log_entry_sync(self, entry: Dict[str, Any]):
        """
        Log a single entry synchronously
        
        Args:
            entry: Log entry dictionary
        """
        try:
            level = entry.get('level', 'INFO')
            message = json.dumps(entry, default=str)
            
            if level == 'ERROR':
                self.logger.error(message)
            elif level == 'WARNING':
                self.logger.warning(message)
            elif level == 'DEBUG':
                self.logger.debug(message)
            else:
                self.logger.info(message)
        except Exception as e:
            # Log the error but don't let it break the batch processing
            fallback_logger = logging.getLogger("batch_sync_error")
            fallback_logger.error(f"Failed to log entry synchronously: {e}")
    
    def flush_all(self):
        """
        Flush all pending log entries immediately
        """
        # Process any remaining entries in queue
        while not self.log_queue.empty():
            try:
                entry = self.log_queue.get_nowait()
                self.batch_buffer.append(entry)
            except Empty:
                break
        
        # Flush current batch
        if self.batch_buffer:
            self._flush_batch()
    
    def shutdown(self):
        """
        Shutdown the batch processor gracefully
        """
        self.running = False
        self.flush_all()
        if self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)


class AgentResponseLogger:
    """
    Central logging component for agent response tracking
    
    Provides structured logging capabilities with CloudWatch integration,
    asynchronous batching, and fallback to console logging when CloudWatch is unavailable.
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the agent response logger
        
        Args:
            config: Optional LoggingConfig instance. If None, loads from environment variables.
        """
        if config is None:
            try:
                config = ConfigurationManager.load_logging_config()
            except ConfigurationError as e:
                # Fall back to default configuration if environment loading fails
                fallback_logger = logging.getLogger("config_error")
                fallback_logger.warning(f"Failed to load configuration from environment: {e}. Using defaults.")
                config = LoggingConfig()
        
        self.config = config
        self.batch_processor = None
        
        # Initialize circuit breaker and retry handler
        self.circuit_breaker = None
        self.retry_handler = None
        if config.circuit_breaker_enabled:
            self.circuit_breaker = LoggingCircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout
            )
        
        if config.retry_enabled:
            self.retry_handler = RetryHandler(
                max_attempts=config.retry_max_attempts,
                base_delay=config.retry_base_delay,
                max_delay=config.retry_max_delay,
                exponential_base=config.retry_exponential_base
            )
        
        # Setup logger with enhanced error handling
        self.logger = self._setup_logger()
        
        # Initialize async batching if enabled
        if config.async_logging and config.enabled:
            try:
                self.batch_processor = AsyncLogBatcher(config, self.logger)
            except Exception as e:
                # Fall back to synchronous logging if async setup fails
                fallback_logger = logging.getLogger("async_setup_error")
                fallback_logger.warning(f"Async logging setup failed, using synchronous: {e}")
    
    @classmethod
    def from_environment(cls) -> 'AgentResponseLogger':
        """
        Create AgentResponseLogger instance from environment variables
        
        Returns:
            AgentResponseLogger instance configured from environment
            
        Raises:
            ConfigurationError: If environment configuration is invalid
        """
        config = ConfigurationManager.load_logging_config()
        return cls(config)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with appropriate handlers and formatters"""
        logger = logging.getLogger("agent_response_logger")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.propagate = False
        
        if not self.config.enabled:
            # Add null handler to prevent logging
            logger.addHandler(logging.NullHandler())
            return logger
            
        # Try to set up CloudWatch logging if enabled
        if self.config.use_cloudwatch:
            try:
                cloudwatch_handler = self._setup_cloudwatch_handler()
                if cloudwatch_handler:
                    logger.addHandler(cloudwatch_handler)
                    return logger
            except Exception as e:
                if self.config.console_fallback:
                    # Log the CloudWatch setup failure and fall back to console
                    console_logger = logging.getLogger("agent_logging_setup")
                    console_logger.warning(f"CloudWatch logging setup failed, falling back to console: {e}")
                else:
                    raise
        
        # Set up console logging (either as fallback or primary)
        if self.config.console_fallback or not self.config.use_cloudwatch:
            console_handler = self._setup_console_handler()
            logger.addHandler(console_handler)
            
        return logger
    
    def _setup_cloudwatch_handler(self) -> Optional[CloudWatchLogHandler]:
        """Set up CloudWatch log handler with enhanced error handling, circuit breaker, and retry logic"""
        def _create_handler():
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                raise Exception("No AWS credentials found")
            
            # Get region, with fallback to us-east-1
            region = session.region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            logs_client = session.client("logs", region_name=region)
            
            # Test CloudWatch connectivity
            logs_client.describe_log_groups(limit=1)
            
            handler = CloudWatchLogHandler(
                log_group_name=self.config.log_group_name,
                boto3_client=logs_client,
                create_log_group=True,
                max_batch_size=self.config.batch_size if self.config.async_logging else 1,
                max_batch_count=100,  # Maximum number of batches to keep in memory
            )
            
            if self.config.structured_format:
                # Use enhanced structured JSON formatter
                formatter = StructuredJSONFormatter(include_extra_fields=True)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - AGENT_RESPONSE - %(levelname)s - %(message)s'
                )
            
            handler.setFormatter(formatter)
            return handler
        
        try:
            # Use circuit breaker if enabled
            if self.circuit_breaker:
                try:
                    # Use retry handler if enabled
                    if self.retry_handler:
                        handler = self.circuit_breaker.call(
                            self.retry_handler.execute_with_retry, 
                            _create_handler
                        )
                    else:
                        handler = self.circuit_breaker.call(_create_handler)
                    return handler
                except Exception as e:
                    if self.config.console_fallback:
                        fallback_logger = logging.getLogger("cloudwatch_circuit_breaker")
                        fallback_logger.warning(
                            f"CloudWatch handler setup failed (circuit breaker: {self.circuit_breaker.get_state().value}): {e}"
                        )
                        return None
                    else:
                        raise
            else:
                # Use retry handler if enabled, otherwise direct call
                if self.retry_handler:
                    handler = self.retry_handler.execute_with_retry(_create_handler)
                else:
                    handler = _create_handler()
                return handler
                
        except Exception as e:
            if not self.config.console_fallback:
                raise
            # Log the specific error for debugging
            fallback_logger = logging.getLogger("cloudwatch_setup_error")
            fallback_logger.error(f"CloudWatch handler setup failed: {e}")
            return None
    
    def _setup_console_handler(self) -> logging.StreamHandler:
        """Set up console log handler with enhanced formatting"""
        handler = logging.StreamHandler()
        
        if self.config.structured_format:
            # Use enhanced structured JSON formatter for console too
            formatter = StructuredJSONFormatter(include_extra_fields=True)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - AGENT_RESPONSE - %(levelname)s - %(message)s'
            )
            
        handler.setFormatter(formatter)
        return handler
    
    def _create_structured_log(self, event_type: str, agent_info: AgentInfo, 
                             additional_data: Dict[str, Any] = None) -> str:
        """Create structured JSON log entry (legacy method for backward compatibility)"""
        log_entry = self._create_structured_log_dict(event_type, agent_info, additional_data)
        return json.dumps(log_entry, default=str)
    
    def _create_structured_log_dict(self, event_type: str, agent_info: AgentInfo, 
                                  additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create structured log entry as dictionary with level-appropriate metadata"""
        # Basic log entry (always included)
        log_entry = {
            "timestamp": agent_info.timestamp.isoformat(),
            "level": "INFO",
            "event_type": event_type,
            "agent_name": agent_info.agent_name,
            "agent_type": agent_info.agent_type,
        }
        
        # Include detailed metadata for DEBUG level, basic for INFO and above
        if self.config.log_level.upper() == "DEBUG":
            # DEBUG: Include detailed agent metadata (requirement 2.3)
            log_entry.update({
                "model_id": agent_info.model_id,
                "session_id": agent_info.session_id,
                "execution_time_ms": agent_info.execution_time_ms,
            })
        else:
            # INFO and above: Include basic agent identification only (requirement 2.4)
            log_entry.update({
                "execution_time_ms": agent_info.execution_time_ms,
            })
        
        if additional_data:
            # For DEBUG level, include all additional data
            if self.config.log_level.upper() == "DEBUG":
                log_entry.update(additional_data)
            else:
                # For INFO and above, exclude only detailed metadata fields
                # but include all other essential operational data
                excluded_fields = ["metadata"]  # Only exclude detailed metadata
                for field, value in additional_data.items():
                    if field not in excluded_fields:
                        log_entry[field] = value
            
        return log_entry
    
    def _log_entry_sync(self, log_entry: Dict[str, Any]) -> None:
        """Log an entry synchronously with enhanced error handling"""
        def _perform_logging():
            level = log_entry.get('level', 'INFO')
            
            if self.config.structured_format:
                message = json.dumps(log_entry, default=str)
            else:
                message = log_entry.get('message', str(log_entry))
            
            if level == 'ERROR':
                self.logger.error(message)
            elif level == 'WARNING':
                self.logger.warning(message)
            elif level == 'DEBUG':
                self.logger.debug(message)
            else:
                self.logger.info(message)
        
        try:
            # Use circuit breaker if enabled and CloudWatch is being used
            if (self.circuit_breaker and self.config.use_cloudwatch and 
                hasattr(self.logger, 'handlers') and 
                any(isinstance(h, CloudWatchLogHandler) for h in self.logger.handlers)):
                
                try:
                    if self.retry_handler:
                        self.circuit_breaker.call(
                            self.retry_handler.execute_with_retry,
                            _perform_logging
                        )
                    else:
                        self.circuit_breaker.call(_perform_logging)
                except Exception as e:
                    # Circuit breaker is open or logging failed, use console fallback
                    if self.config.console_fallback:
                        self._log_to_console_fallback(log_entry)
                    else:
                        raise
            else:
                # Direct logging without circuit breaker
                if self.retry_handler and self.config.use_cloudwatch:
                    self.retry_handler.execute_with_retry(_perform_logging)
                else:
                    _perform_logging()
                    
        except Exception as e:
            # Final fallback to console if all else fails
            if self.config.console_fallback:
                self._log_to_console_fallback(log_entry)
            else:
                fallback_logger = logging.getLogger("sync_logging_error")
                fallback_logger.error(f"Synchronous logging failed: {e}")
    
    def _log_to_console_fallback(self, log_entry: Dict[str, Any]) -> None:
        """Log entry to console as final fallback"""
        try:
            console_logger = logging.getLogger("agent_logging_fallback")
            if not console_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - AGENT_FALLBACK - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                console_logger.addHandler(handler)
                console_logger.setLevel(logging.INFO)
            
            level = log_entry.get('level', 'INFO')
            if self.config.structured_format:
                message = json.dumps(log_entry, default=str)
            else:
                message = log_entry.get('message', str(log_entry))
            
            if level == 'ERROR':
                console_logger.error(message)
            elif level == 'WARNING':
                console_logger.warning(message)
            elif level == 'DEBUG':
                console_logger.debug(message)
            else:
                console_logger.info(message)
                
        except Exception as e:
            # Last resort - print to stderr
            import sys
            print(f"AGENT_LOGGING_EMERGENCY_FALLBACK: {log_entry}", file=sys.stderr)
    
    def log_agent_response(self, agent_info: AgentInfo, response_data: ResponseData) -> None:
        """
        Log agent response with structured format and async batching support
        
        Args:
            agent_info: AgentInfo instance with agent metadata
            response_data: ResponseData instance with response information
        """
        if not self.config.enabled:
            return
            
        try:
            additional_data = {
                "response_length": response_data.response_length,
                "success": response_data.success,
                "metadata": response_data.metadata,
            }
            
            # Add query hash for privacy (don't log full query content by default)
            query_hash = str(hash(response_data.query))[:8]
            additional_data["query_hash"] = query_hash
            
            # Optionally include response content (be careful with sensitive data)
            if self.config.include_response_content:
                additional_data["query"] = response_data.query
                additional_data["response"] = response_data.response
            
            # Create log entry
            if self.config.structured_format:
                log_entry = self._create_structured_log_dict(
                    "agent_response", agent_info, additional_data
                )
            else:
                log_entry = {
                    "level": "INFO",
                    "message": (
                        f"Agent Response - {agent_info.agent_name} ({agent_info.agent_type}) "
                        f"processed query in {agent_info.execution_time_ms}ms, "
                        f"response length: {response_data.response_length}, "
                        f"success: {response_data.success}"
                    )
                }
            
            # Use async batching if available, otherwise log synchronously
            if self.batch_processor and self.config.async_logging:
                if not self.batch_processor.add_log_entry(log_entry):
                    # Queue full, fall back to synchronous logging
                    self._log_entry_sync(log_entry)
            else:
                self._log_entry_sync(log_entry)
            
        except Exception as e:
            # Never let logging failures break the main application
            fallback_logger = logging.getLogger("agent_logging_error")
            fallback_logger.error(f"Failed to log agent response: {e}")
    
    def log_routing_decision(self, agent_info: AgentInfo, routing_info: RoutingInfo) -> None:
        """
        Log orchestrator routing decisions with async batching support
        
        Args:
            agent_info: AgentInfo instance for the orchestrator
            routing_info: RoutingInfo instance with routing decision data
        """
        if not self.config.enabled:
            return
            
        try:
            additional_data = {
                "selected_agent": routing_info.selected_agent,
                "routing_reasoning": routing_info.routing_reasoning,
                "available_agents": routing_info.available_agents,
            }
            
            if routing_info.confidence_score is not None:
                additional_data["confidence_score"] = routing_info.confidence_score
            
            # Create log entry
            if self.config.structured_format:
                log_entry = self._create_structured_log_dict(
                    "routing_decision", agent_info, additional_data
                )
            else:
                log_entry = {
                    "level": "INFO",
                    "message": (
                        f"Routing Decision - Selected {routing_info.selected_agent} "
                        f"from {routing_info.available_agents}, "
                        f"reasoning: {routing_info.routing_reasoning}"
                    )
                }
            
            # Use async batching if available, otherwise log synchronously
            if self.batch_processor and self.config.async_logging:
                if not self.batch_processor.add_log_entry(log_entry):
                    # Queue full, fall back to synchronous logging
                    self._log_entry_sync(log_entry)
            else:
                self._log_entry_sync(log_entry)
            
        except Exception as e:
            fallback_logger = logging.getLogger("agent_logging_error")
            fallback_logger.error(f"Failed to log routing decision: {e}")
    
    def log_agent_error(self, agent_info: AgentInfo, error: Exception, 
                       context: Dict[str, Any] = None) -> None:
        """
        Log agent errors with context and async batching support
        
        Args:
            agent_info: AgentInfo instance with agent metadata
            error: Exception that occurred
            context: Additional context information
        """
        if not self.config.enabled:
            return
            
        try:
            additional_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "success": False,
            }
            
            if context:
                additional_data["error_context"] = context
            
            # Create log entry
            if self.config.structured_format:
                log_entry = self._create_structured_log_dict(
                    "agent_error", agent_info, additional_data
                )
                log_entry["level"] = "ERROR"  # Override level for errors
            else:
                log_entry = {
                    "level": "ERROR",
                    "message": (
                        f"Agent Error - {agent_info.agent_name} ({agent_info.agent_type}) "
                        f"encountered {type(error).__name__}: {str(error)}"
                    )
                }
            
            # Use async batching if available, otherwise log synchronously
            if self.batch_processor and self.config.async_logging:
                if not self.batch_processor.add_log_entry(log_entry):
                    # Queue full, fall back to synchronous logging
                    self._log_entry_sync(log_entry)
            else:
                self._log_entry_sync(log_entry)
            
        except Exception as e:
            fallback_logger = logging.getLogger("agent_logging_error")
            fallback_logger.error(f"Failed to log agent error: {e}")
    
    def flush_logs(self) -> None:
        """
        Flush all pending log entries immediately
        
        Useful for ensuring logs are written before application shutdown
        """
        if self.batch_processor:
            self.batch_processor.flush_all()
    
    def shutdown(self) -> None:
        """
        Shutdown the logger gracefully, flushing all pending logs
        """
        if self.batch_processor:
            self.batch_processor.shutdown()
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get current circuit breaker status
        
        Returns:
            Dictionary with circuit breaker status information
        """
        if not self.circuit_breaker:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "state": self.circuit_breaker.get_state().value,
            "failure_count": self.circuit_breaker.failure_count,
            "failure_threshold": self.circuit_breaker.failure_threshold,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "recovery_timeout": self.circuit_breaker.recovery_timeout
        }
    
    def reset_circuit_breaker(self) -> bool:
        """
        Manually reset the circuit breaker
        
        Returns:
            True if circuit breaker was reset, False if not enabled
        """
        if not self.circuit_breaker:
            return False
        
        self.circuit_breaker.reset()
        return True
    
    def test_cloudwatch_connectivity(self) -> Dict[str, Any]:
        """
        Test CloudWatch connectivity and return status
        
        Returns:
            Dictionary with connectivity test results
        """
        if not self.config.use_cloudwatch:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "CloudWatch logging is disabled"
            }
        
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                return {
                    "enabled": True,
                    "status": "failed",
                    "message": "No AWS credentials found"
                }
            
            region = session.region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            logs_client = session.client("logs", region_name=region)
            
            # Test connectivity
            logs_client.describe_log_groups(limit=1)
            
            return {
                "enabled": True,
                "status": "connected",
                "message": f"Successfully connected to CloudWatch in region {region}",
                "region": region,
                "log_group": self.config.log_group_name
            }
            
        except Exception as e:
            return {
                "enabled": True,
                "status": "failed",
                "message": f"CloudWatch connectivity test failed: {e}"
            }


class AgentResponseWrapper:
    """
    Response wrapper that includes agent identification when enabled
    
    Formats agent responses with optional agent identification based on
    user preferences and configuration settings.
    """
    
    def __init__(self, user_facing_config: Optional[UserFacingConfig] = None):
        """
        Initialize the response wrapper
        
        Args:
            user_facing_config: Optional UserFacingConfig instance. If None, loads from environment variables.
        """
        if user_facing_config is None:
            try:
                user_facing_config = ConfigurationManager.load_user_facing_config()
            except ConfigurationError as e:
                # Fall back to default configuration if environment loading fails
                fallback_logger = logging.getLogger("config_error")
                fallback_logger.warning(f"Failed to load user facing configuration from environment: {e}. Using defaults.")
                user_facing_config = UserFacingConfig()
        
        self.config = user_facing_config
    
    @classmethod
    def from_environment(cls) -> 'AgentResponseWrapper':
        """
        Create AgentResponseWrapper instance from environment variables
        
        Returns:
            AgentResponseWrapper instance configured from environment
            
        Raises:
            ConfigurationError: If environment configuration is invalid
        """
        config = ConfigurationManager.load_user_facing_config()
        return cls(config)
    
    def wrap_response(self, response: str, agent_info: AgentInfo, 
                     additional_info: Dict[str, Any] = None) -> str:
        """
        Wrap agent response with identification information when enabled
        
        Args:
            response: The original agent response
            agent_info: AgentInfo instance with agent metadata
            additional_info: Optional additional information to include
            
        Returns:
            Formatted response with optional agent identification
        """
        if not self.config.show_agent_info:
            return response
        
        # Build agent identification string
        agent_id_parts = []
        
        if self.config.show_agent_name:
            # Use custom name if available, otherwise use agent name
            display_name = self.config.custom_agent_names.get(
                agent_info.agent_name, 
                agent_info.agent_name.replace("_", " ").title()
            )
            agent_id_parts.append(display_name)
        
        if self.config.show_agent_type and agent_info.agent_type != "custom":
            type_display = agent_info.agent_type.replace("_", " ").title()
            if type_display not in " ".join(agent_id_parts):  # Avoid duplication
                agent_id_parts.append(type_display)
        
        if self.config.show_model_info:
            # Simplify model ID for user display
            model_display = self._format_model_id(agent_info.model_id)
            agent_id_parts.append(f"using {model_display}")
        
        if self.config.show_execution_time:
            if agent_info.execution_time_ms < 1000:
                time_display = f"{agent_info.execution_time_ms}ms"
            else:
                time_display = f"{agent_info.execution_time_ms / 1000:.1f}s"
            agent_id_parts.append(f"in {time_display}")
        
        # Add additional info if provided
        if additional_info:
            for key, value in additional_info.items():
                if key == "routing_reasoning" and value:
                    agent_id_parts.append(f"({value})")
                elif key == "knowledge_base_used" and value:
                    agent_id_parts.append("with knowledge base")
        
        if not agent_id_parts:
            return response
        
        agent_identification = " ".join(agent_id_parts)
        
        # Format according to style preference
        return self._format_response(response, agent_identification)
    
    def _format_model_id(self, model_id: str) -> str:
        """
        Format model ID for user-friendly display
        
        Args:
            model_id: Raw model ID from agent
            
        Returns:
            User-friendly model name
        """
        # Map common model IDs to friendly names
        model_mappings = {
            "us.amazon.nova-micro-v1:0": "Nova Micro",
            "us.amazon.nova-lite-v1:0": "Nova Lite", 
            "us.amazon.nova-pro-v1:0": "Nova Pro",
            "us.anthropic.claude-sonnet-4-20250514-v1:0": "Claude Sonnet",
            "us.anthropic.claude-opus-4-20250514-v1:0": "Claude Opus",
            "us.anthropic.claude-haiku-4-20250514-v1:0": "Claude Haiku"
        }
        
        return model_mappings.get(model_id, model_id.split(":")[-2] if ":" in model_id else model_id)
    
    def _format_response(self, response: str, agent_identification: str) -> str:
        """
        Format the response with agent identification according to style
        
        Args:
            response: Original response text
            agent_identification: Agent identification string
            
        Returns:
            Formatted response with agent identification
        """
        if self.config.format_style == "bracket":
            return f"[Response from {agent_identification}]\n\n{response}"
        elif self.config.format_style == "prefix":
            return f"{agent_identification}: {response}"
        elif self.config.format_style == "suffix":
            return f"{response}\n\n— {agent_identification}"
        else:  # "none" or any other value
            return response
    
    def wrap_multi_agent_response(self, response: str, orchestrator_info: AgentInfo,
                                 selected_agent_info: AgentInfo, routing_reasoning: str = None) -> str:
        """
        Wrap multi-agent response with both orchestrator and selected agent info
        
        Args:
            response: The final response from the selected agent
            orchestrator_info: AgentInfo for the orchestrator
            selected_agent_info: AgentInfo for the agent that provided the response
            routing_reasoning: Optional reasoning for agent selection
            
        Returns:
            Formatted response with multi-agent identification
        """
        if not self.config.show_agent_info:
            return response
        
        # For multi-agent, show the selected agent primarily
        additional_info = {}
        if routing_reasoning:
            additional_info["routing_reasoning"] = routing_reasoning
        
        return self.wrap_response(response, selected_agent_info, additional_info)


class LoggingConfigManager:
    """
    Configuration manager for agent response logging
    
    Handles loading configuration from environment variables and provides
    validation and default value handling.
    """
    
    @staticmethod
    def load_from_environment() -> LoggingConfig:
        """
        Load logging configuration from environment variables
        
        Returns:
            LoggingConfig instance with settings from environment or defaults
        """
        return LoggingConfig(
            enabled=os.environ.get("AGENT_LOGGING_ENABLED", "true").lower() == "true",
            log_level=os.environ.get("AGENT_LOGGING_LEVEL", "INFO").upper(),
            include_response_content=os.environ.get("AGENT_LOGGING_INCLUDE_CONTENT", "false").lower() == "true",
            include_user_facing=os.environ.get("AGENT_LOGGING_USER_FACING", "false").lower() == "true",
            structured_format=os.environ.get("AGENT_LOGGING_STRUCTURED", "true").lower() == "true",
            log_group_name=os.environ.get("AGENT_LOGGING_GROUP", "/bedrockagent/agent-responses"),
            use_cloudwatch=os.environ.get("AGENT_LOGGING_CLOUDWATCH", "true").lower() == "true",
            console_fallback=os.environ.get("AGENT_LOGGING_CONSOLE_FALLBACK", "true").lower() == "true",
            # New async and batching configuration
            async_logging=os.environ.get("AGENT_LOGGING_ASYNC", "true").lower() == "true",
            batch_size=int(os.environ.get("AGENT_LOGGING_BATCH_SIZE", "10")),
            batch_timeout_seconds=float(os.environ.get("AGENT_LOGGING_BATCH_TIMEOUT", "5.0")),
            max_queue_size=int(os.environ.get("AGENT_LOGGING_MAX_QUEUE_SIZE", "1000")),
        )
    
    @staticmethod
    def load_user_facing_config_from_environment() -> UserFacingConfig:
        """
        Load user-facing configuration from environment variables
        
        Returns:
            UserFacingConfig instance with settings from environment or defaults
        """
        return UserFacingConfig(
            show_agent_info=os.environ.get("AGENT_USER_FACING_ENABLED", "false").lower() == "true",
            show_agent_name=os.environ.get("AGENT_USER_FACING_SHOW_NAME", "true").lower() == "true",
            show_agent_type=os.environ.get("AGENT_USER_FACING_SHOW_TYPE", "true").lower() == "true",
            show_model_info=os.environ.get("AGENT_USER_FACING_SHOW_MODEL", "false").lower() == "true",
            show_execution_time=os.environ.get("AGENT_USER_FACING_SHOW_TIME", "false").lower() == "true",
            format_style=os.environ.get("AGENT_USER_FACING_FORMAT", "bracket").lower(),
        )
    
    @staticmethod
    def validate_config(config: LoggingConfig) -> LoggingConfig:
        """
        Validate and normalize configuration values
        
        Args:
            config: LoggingConfig instance to validate
            
        Returns:
            Validated LoggingConfig instance with normalized values
        """
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level not in valid_levels:
            config.log_level = "INFO"
        
        # Ensure log group name starts with /
        if config.log_group_name and not config.log_group_name.startswith("/"):
            config.log_group_name = f"/{config.log_group_name}"
        
        # Validate batch configuration
        if config.batch_size < 1:
            config.batch_size = 1
        elif config.batch_size > 100:
            config.batch_size = 100
            
        if config.batch_timeout_seconds < 0.1:
            config.batch_timeout_seconds = 0.1
        elif config.batch_timeout_seconds > 60.0:
            config.batch_timeout_seconds = 60.0
            
        if config.max_queue_size < 10:
            config.max_queue_size = 10
        elif config.max_queue_size > 10000:
            config.max_queue_size = 10000
        
        return config


def create_agent_logger(config: LoggingConfig = None) -> AgentResponseLogger:
    """
    Factory function to create an AgentResponseLogger instance
    
    Args:
        config: Optional LoggingConfig instance. If None, loads from environment.
        
    Returns:
        Configured AgentResponseLogger instance
    """
    if config is None:
        config = LoggingConfigManager.load_from_environment()
    
    config = LoggingConfigManager.validate_config(config)
    return AgentResponseLogger(config)


def create_response_wrapper(user_facing_config: UserFacingConfig = None) -> AgentResponseWrapper:
    """
    Factory function to create an AgentResponseWrapper instance
    
    Args:
        user_facing_config: Optional UserFacingConfig instance. If None, loads from environment.
        
    Returns:
        Configured AgentResponseWrapper instance
    """
    if user_facing_config is None:
        user_facing_config = LoggingConfigManager.load_user_facing_config_from_environment()
    
    return AgentResponseWrapper(user_facing_config)