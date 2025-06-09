"""
Claude client utilities for Excel Agent
"""
import os
import logging
from typing import Optional, Dict, Any
from langchain_anthropic import ChatAnthropic

# Configure logging
logger = logging.getLogger(__name__)

# Available Claude models
CLAUDE_MODELS = {
    "sonnet": "claude-3-sonnet-20240229",
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307"
}

# Default configuration
DEFAULT_CONFIG = {
    "model": "claude-3-sonnet-20240229",
    "temperature": 0,
    "max_tokens": 4096,
    "timeout": 60.0
}


def get_claude_client(
    api_key: Optional[str] = None,
    model: str = "claude-3-sonnet-20240229",
    temperature: float = 0,
    max_tokens: int = 4096,
    timeout: float = 60.0,
    **kwargs
) -> ChatAnthropic:
    """
    Create a Claude client with the provided configuration
    
    Args:
        api_key: Anthropic API key. If None, will try to get from environment
        model: Claude model to use (default: claude-3-sonnet-20240229)
        temperature: Temperature for response generation (0-1, default: 0)
        max_tokens: Maximum tokens in response (default: 4096)
        timeout: Request timeout in seconds (default: 60.0)
        **kwargs: Additional arguments for ChatAnthropic
        
    Returns:
        Configured ChatAnthropic client
        
    Raises:
        ValueError: If API key is empty, None, or invalid
        ConnectionError: If unable to connect to Anthropic API
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Validate API key
    if not api_key or not isinstance(api_key, str):
        raise ValueError(
            "Anthropic API key is required. Provide it as parameter or set ANTHROPIC_API_KEY environment variable."
        )
    
    if len(api_key.strip()) < 10:  # Basic validation
        raise ValueError("Invalid Anthropic API key format")
    
    # Validate model
    if model not in CLAUDE_MODELS.values():
        logger.warning(f"Model '{model}' not in known models list. Proceeding anyway.")
    
    # Validate parameters
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    
    if not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be between 1 and 8192")
    
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    
    # Prepare client configuration
    client_config = {
        "anthropic_api_key": api_key.strip(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        **kwargs
    }
    
    try:
        # Create and configure the client
        client = ChatAnthropic(**client_config)
        
        logger.info(f"Claude client created successfully with model: {model}")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to create Claude client: {str(e)}")
        raise ConnectionError(f"Unable to create Claude client: {str(e)}")


def get_claude_client_from_env(
    env_var: str = "ANTHROPIC_API_KEY",
    **kwargs
) -> ChatAnthropic:
    """
    Create a Claude client using API key from environment variable
    
    Args:
        env_var: Environment variable name (default: ANTHROPIC_API_KEY)
        **kwargs: Additional arguments for get_claude_client
        
    Returns:
        Configured ChatAnthropic client
        
    Raises:
        ValueError: If environment variable is not set
    """
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Environment variable '{env_var}' is not set")
    
    return get_claude_client(api_key=api_key, **kwargs)


def validate_api_key(api_key: str) -> bool:
    """
    Basic validation of API key format
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format checks
    api_key = api_key.strip()
    
    # Anthropic API keys typically start with 'sk-ant-' and are much longer
    if len(api_key) < 20:
        return False
    
    # Additional format validation could be added here
    return True


def get_available_models() -> Dict[str, str]:
    """
    Get dictionary of available Claude models
    
    Returns:
        Dictionary mapping model names to model IDs
    """
    return CLAUDE_MODELS.copy()


def create_client_with_config(config: Dict[str, Any]) -> ChatAnthropic:
    """
    Create Claude client from configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ChatAnthropic client
    """
    # Merge with defaults
    merged_config = {**DEFAULT_CONFIG, **config}
    
    return get_claude_client(**merged_config) 