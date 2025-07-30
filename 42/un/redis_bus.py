"""
Redis Event Bus for 42.un

Central event relay system for fast reactions and event persistence.
"""

import redis
import json
import threading
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from .events import Event, EventType


class RedisBus:
    """Redis-based event bus for 42.un system."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, max_connections: int = 10):
        """Initialize Redis bus."""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        
        # Redis connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=True
        )
        
        # Event handlers
        self.handlers: Dict[EventType, List[Callable]] = {}
        
        # Subscription thread
        self.subscription_thread = None
        self.running = False
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def publish_event(self, event: Event) -> bool:
        """Publish an event to Redis."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            
            # Publish to event channel
            channel = f"42:events:{event.event_type.value}"
            message = event.to_json()
            
            # Publish to specific channel
            redis_client.publish(channel, message)
            
            # Publish to general events channel
            redis_client.publish("42:events", message)
            
            # Store event for persistence
            self._store_event(event)
            
            logger.debug(f"Published event {event.event_id} to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
    
    def _store_event(self, event: Event) -> None:
        """Store event for persistence."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            
            # Store in sorted set by timestamp
            redis_client.zadd(
                "42:events:history",
                {event.to_json(): event.timestamp.timestamp()}
            )
            
            # Keep only last 10000 events
            redis_client.zremrangebyrank("42:events:history", 0, -10001)
            
        except Exception as e:
            logger.error(f"Failed to store event {event.event_id}: {e}")
    
    def subscribe_to_events(self, event_types: List[EventType], 
                          callback: Callable[[Event], None]) -> bool:
        """Subscribe to specific event types."""
        try:
            for event_type in event_types:
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                self.handlers[event_type].append(callback)
            
            logger.info(f"Subscribed to events: {[et.value for et in event_types]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")
            return False
    
    def start_subscription_thread(self) -> bool:
        """Start the subscription thread."""
        if self.running:
            logger.warning("Subscription thread already running")
            return False
        
        try:
            self.running = True
            self.subscription_thread = threading.Thread(
                target=self._subscription_worker,
                daemon=True
            )
            self.subscription_thread.start()
            logger.info("Started Redis subscription thread")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start subscription thread: {e}")
            self.running = False
            return False
    
    def stop_subscription_thread(self) -> None:
        """Stop the subscription thread."""
        self.running = False
        if self.subscription_thread:
            self.subscription_thread.join(timeout=5)
        logger.info("Stopped Redis subscription thread")
    
    def _subscription_worker(self) -> None:
        """Worker thread for handling Redis subscriptions."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            pubsub = redis_client.pubsub()
            
            # Subscribe to all event channels
            channels = ["42:events"] + [f"42:events:{et.value}" for et in self.handlers.keys()]
            pubsub.subscribe(*channels)
            
            logger.info(f"Subscribed to channels: {channels}")
            
            for message in pubsub.listen():
                if not self.running:
                    break
                
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        event = Event.from_dict(event_data)
                        
                        # Call handlers for this event type
                        if event.event_type in self.handlers:
                            for handler in self.handlers[event.event_type]:
                                try:
                                    handler(event)
                                except Exception as e:
                                    logger.error(f"Handler error for event {event.event_id}: {e}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process message: {e}")
                        
        except Exception as e:
            logger.error(f"Subscription worker error: {e}")
    
    def get_event_history(self, limit: int = 100, 
                         since: Optional[datetime] = None) -> List[Event]:
        """Get event history."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            
            # Get events from sorted set
            if since:
                min_score = since.timestamp()
                events = redis_client.zrangebyscore(
                    "42:events:history",
                    min_score,
                    "+inf",
                    start=0,
                    num=limit
                )
            else:
                events = redis_client.zrange(
                    "42:events:history",
                    -limit,
                    -1
                )
            
            # Parse events
            result = []
            for event_json in events:
                try:
                    event = Event.from_json(event_json)
                    result.append(event)
                except Exception as e:
                    logger.error(f"Failed to parse event: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            
            # Get total events
            total_events = redis_client.zcard("42:events:history")
            
            # Get events in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_events = redis_client.zcount(
                "42:events:history",
                one_hour_ago.timestamp(),
                "+inf"
            )
            
            # Get events by type
            events_by_type = {}
            for event_type in EventType:
                channel = f"42:events:{event_type.value}"
                count = redis_client.pubsub_numsub(channel)[0][1]
                events_by_type[event_type.value] = count
            
            return {
                "total_events": total_events,
                "recent_events": recent_events,
                "events_by_type": events_by_type,
                "handlers_registered": len(self.handlers)
            }
            
        except Exception as e:
            logger.error(f"Failed to get event stats: {e}")
            return {}
    
    def clear_history(self) -> bool:
        """Clear event history."""
        try:
            redis_client = redis.Redis(connection_pool=self.pool)
            redis_client.delete("42:events:history")
            logger.info("Cleared event history")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear event history: {e}")
            return False
    
    def close(self) -> None:
        """Close Redis connections."""
        self.stop_subscription_thread()
        self.pool.disconnect()
        logger.info("Closed Redis bus") 