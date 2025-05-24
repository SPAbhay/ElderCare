import redis
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Redis connection details from environment variables
# Default to localhost and standard port if not specified
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Initialize Redis client
# decode_responses=True means that it will decode responses from Redis (e.g., bytes to strings)
try:
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True, # Automatically decode responses from bytes to strings
        socket_connect_timeout=5 # Optional: timeout for connection
    )
    # Ping the server to ensure connection
    redis_client.ping()
    print(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Error: {e}")
    print("Please ensure Redis server is running and accessible.")
    print("Caching features will be unavailable.")
    redis_client = None # Set to None if connection fails
except Exception as e:
    print(f"An unexpected error occurred while connecting to Redis: {e}")
    redis_client = None

# Example of how you might use it (we'll integrate this later)
# def get_cached_data(key: str):
#     if redis_client:
#         return redis_client.get(key)
#     return None

# def set_cached_data(key: str, value: str, expiration_secs: int = 3600):
#     if redis_client:
#         redis_client.setex(key, expiration_secs, value)

if __name__ == "__main__":
    if redis_client:
        print("Testing Redis connection...")
        redis_client.set("mytestkey", "Hello Redis!")
        value = redis_client.get("mytestkey")
        print(f"Set and got value: {value}")
        redis_client.delete("mytestkey")
        print("Test key deleted.")
    else:
        print("Redis client not available for testing.")