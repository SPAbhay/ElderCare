from langchain_openai import ChatOpenAI 
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache 
# from langchain.cache import InMemoryCache # Fallback if Redis is not available - now from community
from langchain_community.cache import InMemoryCache 

from .redis_client import redis_client # Import your configured Redis client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- OpenRouter Configuration ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Ensure this is in your .env file
# Default model, can be overridden in get_llm or set via another env var
DEFAULT_OPENROUTER_MODEL = os.getenv("DEFAULT_OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free") # Using a generally available free model as default

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment variables. LLM calls will fail.")

# --- LLM Cache Configuration ---
if redis_client:
    try:
        llm_redis_cache = RedisCache(redis_client)
        set_llm_cache(llm_redis_cache)
        print(f"LangChain LLM response cache configured to use Redis at {redis_client.connection_pool.connection_kwargs.get('host')}:{redis_client.connection_pool.connection_kwargs.get('port')}")
    except Exception as e:
        print(f"Could not configure LangChain LLM response cache with Redis. Error: {e}")
        print("Attempting to use InMemoryCache for LLM responses.")
        set_llm_cache(InMemoryCache())
else:
    print("Redis client not available. Using InMemoryCache for LLM responses.")
    set_llm_cache(InMemoryCache())


# Function to get an LLM instance configured for OpenRouter
def get_llm(model_name: str = DEFAULT_OPENROUTER_MODEL, temperature: float = 0.7, request_timeout: int = 120):
    """
    Initializes and returns a ChatOpenAI instance configured for OpenRouter.
    """
    if not OPENROUTER_API_KEY:
        print("ERROR: Cannot initialize LLM. OPENROUTER_API_KEY is not set.")
        return None
    try:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_base=OPENROUTER_API_BASE,
            openai_api_key=OPENROUTER_API_KEY,
            request_timeout=request_timeout,
            # max_tokens= # Optional: set max tokens if needed
        )
        print(f"ChatOpenAI instance created for OpenRouter model: {model_name}")
        return llm
    except Exception as e:
        print(f"Error initializing ChatOpenAI model {model_name} for OpenRouter: {e}")
        return None

if __name__ == "__main__":
    print(f"Attempting to initialize LLM with OpenRouter model: {DEFAULT_OPENROUTER_MODEL}")
    llm_instance = get_llm() # Test with default model
    
    if llm_instance:
        print("LLM instance successfully created using OpenRouter.")
        try:
            print("Testing LLM invocation (this might take a moment)...")
            # A simple invocation test
            response = llm_instance.invoke("Why is the sky blue? Explain briefly.")
            print(f"LLM test response: {response.content[:150]}...") 
            print("LLM invocation test successful.")

            # Test caching (if Redis is up or InMemoryCache is working)
            print("Testing LLM caching (invoking the same prompt again)...")
            response_cached = llm_instance.invoke("Why is the sky blue? Explain briefly.") 
            print(f"LLM cached response: {response_cached.content[:150]}...")
            print("If the second call was faster, caching worked.")

        except Exception as e:
            print(f"Error during LLM invocation test: {e}")
    else:
        print("Failed to create LLM instance with OpenRouter.")

