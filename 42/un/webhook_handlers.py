"""
Webhook handlers for 42.un

This module implements webhook handlers for various sources,
enabling real-time event processing from external services.
"""

import hashlib
import hmac
import json
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .redis_bus import RedisBus
from .events import Event, EventType, create_github_repo_updated_event


class WebhookValidator:
    """Validate webhook signatures and payloads."""
    
    @staticmethod
    def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
        """Verify GitHub webhook signature."""
        if not signature or not secret:
            return False
        
        # GitHub uses sha256= prefix
        if not signature.startswith('sha256='):
            return False
        
        expected_signature = signature[7:]  # Remove 'sha256=' prefix
        
        # Create HMAC signature
        computed_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(computed_signature, expected_signature)
    
    @staticmethod
    def verify_generic_signature(payload: bytes, signature: str, secret: str, algorithm: str = 'sha256') -> bool:
        """Verify generic webhook signature."""
        if not signature or not secret:
            return False
        
        # Create HMAC signature
        if algorithm == 'sha256':
            hash_func = hashlib.sha256
        elif algorithm == 'sha1':
            hash_func = hashlib.sha1
        else:
            return False
        
        computed_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hash_func
        ).hexdigest()
        
        return hmac.compare_digest(computed_signature, signature)


class GitHubWebhookHandler:
    """Handle GitHub webhook events."""
    
    def __init__(self, redis_bus: RedisBus, secret: str):
        self.redis_bus = redis_bus
        self.secret = secret
        self.validator = WebhookValidator()
    
    async def handle_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Handle incoming GitHub webhook."""
        try:
            # Verify signature
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            if not self.validator.verify_github_signature(payload_bytes, signature, self.secret):
                logger.error("Invalid GitHub webhook signature")
                return False
            
            # Process based on event type
            event_type = payload.get("ref_type")
            if event_type == "push":
                await self._handle_push_event(payload)
            elif event_type == "pull_request":
                await self._handle_pr_event(payload)
            elif event_type == "create":
                await self._handle_create_event(payload)
            elif event_type == "delete":
                await self._handle_delete_event(payload)
            else:
                logger.info(f"Unhandled GitHub event type: {event_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling GitHub webhook: {e}")
            return False
    
    async def _handle_push_event(self, payload: Dict[str, Any]):
        """Handle push events."""
        repo_url = payload["repository"]["html_url"]
        commit_hash = payload["after"]
        branch = payload["ref"].replace("refs/heads/", "")
        author = payload["pusher"]["name"]
        message = payload.get("head_commit", {}).get("message", "")
        
        event = create_github_repo_updated_event(
            repo_url=repo_url,
            commit_hash=commit_hash,
            branch=branch,
            author=author,
            message=message
        )
        
        await self.redis_bus.publish_event(event)
        logger.info(f"GitHub push event: {repo_url} -> {commit_hash[:8]}")
    
    async def _handle_pr_event(self, payload: Dict[str, Any]):
        """Handle pull request events."""
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo_url = payload["repository"]["html_url"]
        
        if action in ["opened", "synchronize", "reopened"]:
            event = Event(
                event_type=EventType.GITHUB_REPO_UPDATED,
                data={
                    "repo_url": repo_url,
                    "pr_number": pr.get("number"),
                    "pr_title": pr.get("title"),
                    "pr_url": pr.get("html_url"),
                    "action": action,
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source="github_webhook"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"GitHub PR event: {repo_url} PR #{pr.get('number')}")
    
    async def _handle_create_event(self, payload: Dict[str, Any]):
        """Handle create events (branches, tags)."""
        ref_type = payload.get("ref_type")
        ref = payload.get("ref")
        repo_url = payload["repository"]["html_url"]
        
        event = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            data={
                "repo_url": repo_url,
                "ref_type": ref_type,
                "ref": ref,
                "action": "created",
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            source="github_webhook"
        )
        
        await self.redis_bus.publish_event(event)
        logger.info(f"GitHub create event: {repo_url} {ref_type} {ref}")
    
    async def _handle_delete_event(self, payload: Dict[str, Any]):
        """Handle delete events (branches, tags)."""
        ref_type = payload.get("ref_type")
        ref = payload.get("ref")
        repo_url = payload["repository"]["html_url"]
        
        event = Event(
            event_type=EventType.GITHUB_REPO_UPDATED,
            data={
                "repo_url": repo_url,
                "ref_type": ref_type,
                "ref": ref,
                "action": "deleted",
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            source="github_webhook"
        )
        
        await self.redis_bus.publish_event(event)
        logger.info(f"GitHub delete event: {repo_url} {ref_type} {ref}")


class GenericWebhookHandler:
    """Handle generic webhook events."""
    
    def __init__(self, redis_bus: RedisBus, secret: str, algorithm: str = 'sha256'):
        self.redis_bus = redis_bus
        self.secret = secret
        self.algorithm = algorithm
        self.validator = WebhookValidator()
    
    async def handle_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Handle incoming generic webhook."""
        try:
            # Verify signature
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            if not self.validator.verify_generic_signature(payload_bytes, signature, self.secret, self.algorithm):
                logger.error("Invalid webhook signature")
                return False
            
            # Process payload
            event_type = payload.get("event_type", "generic")
            source = payload.get("source", "unknown")
            
            event = Event(
                event_type=EventType.FILE_INGESTED,  # Reuse for generic events
                data={
                    "webhook_type": "generic",
                    "event_type": event_type,
                    "source": source,
                    "payload": payload,
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                source=f"{source}_webhook"
            )
            
            await self.redis_bus.publish_event(event)
            logger.info(f"Generic webhook event: {source} -> {event_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling generic webhook: {e}")
            return False


class WebhookManager:
    """Manage multiple webhook handlers."""
    
    def __init__(self, redis_bus: RedisBus, config: Dict[str, Any]):
        self.redis_bus = redis_bus
        self.config = config
        self.handlers = {}
        
        # Initialize handlers based on configuration
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup webhook handlers based on configuration."""
        # GitHub webhook handler
        github_config = self.config.get("github", {})
        if github_config.get("webhook_secret"):
            self.handlers["github"] = GitHubWebhookHandler(
                self.redis_bus,
                github_config["webhook_secret"]
            )
        
        # Generic webhook handlers
        generic_config = self.config.get("webhooks", {})
        for webhook_name, webhook_config in generic_config.items():
            if webhook_config.get("secret"):
                self.handlers[webhook_name] = GenericWebhookHandler(
                    self.redis_bus,
                    webhook_config["secret"],
                    webhook_config.get("algorithm", "sha256")
                )
    
    async def handle_webhook(self, webhook_type: str, payload: Dict[str, Any], signature: str) -> bool:
        """Route webhook to appropriate handler."""
        handler = self.handlers.get(webhook_type)
        if not handler:
            logger.error(f"No handler found for webhook type: {webhook_type}")
            return False
        
        return await handler.handle_webhook(payload, signature)
    
    def get_supported_webhooks(self) -> list:
        """Get list of supported webhook types."""
        return list(self.handlers.keys())
    
    def is_webhook_supported(self, webhook_type: str) -> bool:
        """Check if webhook type is supported."""
        return webhook_type in self.handlers 