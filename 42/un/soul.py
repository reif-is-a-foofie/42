"""
Global Soul System for 42.un

Manages personality, preferences, and behavior configuration for all agents.
Provides password-protected editing and system-wide personality management.
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SoulSystem:
    """
    Global soul system for managing agent personalities and preferences.
    
    Features:
    - Password-protected editing
    - 1000-year lockout after 3 failed attempts
    - Versioned soul state
    - Agent-specific adapters
    - Universal scoring functions
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.soul_dir = self.project_root / "soul"
        self.soul_path = self.soul_dir / "soul.json"
        self.soul_lock_path = self.soul_dir / "soul.lock"
        self.soul_failures_path = self.soul_dir / "soul_failures.log"
        self.soul_history_path = self.soul_dir / "soul_history"
        
        # Ensure directories exist
        self.soul_dir.mkdir(parents=True, exist_ok=True)
        self.soul_history_path.mkdir(parents=True, exist_ok=True)
        
        # Security settings
        self.password_attempts = 0
        self.max_password_attempts = 3
        self.lockout_duration = 1000 * 365 * 24 * 3600  # 1000 years in seconds
        
        # Load soul
        self.soul = self._load_soul()
        logger.info(f"Soul system initialized with identity: {self.soul['identity']}")
    
    def _load_soul(self) -> Dict[str, Any]:
        """Load existing soul or create default."""
        # Check for lockout
        if self._is_soul_locked():
            lock_timestamp = self._get_lock_timestamp()
            logger.warning(f"Soul is locked until {datetime.fromtimestamp(lock_timestamp).isoformat()}")
            return self._get_locked_soul()
        
        # Load existing soul or create default
        if self.soul_path.exists():
            try:
                with open(self.soul_path, 'r') as f:
                    content = f.read().strip()
                
                # Try to parse as JSON
                soul_json = json.loads(content)
                
                # Check if it's the new single-key format with "soul" key
                if "soul" in soul_json and len(soul_json) == 1:
                    # Parse the spiritual text from the "soul" key
                    spiritual_text = soul_json["soul"]
                    soul_data = self._parse_spiritual_soul(spiritual_text)
                    logger.info("Loaded spiritual soul configuration from JSON")
                    return soul_data
                else:
                    # Legacy JSON format
                    logger.info("Loaded JSON soul configuration")
                    return soul_json
            except Exception as e:
                logger.error(f"Failed to load soul: {e}")
        
        # Create default soul
        default_soul = self._create_default_soul()
        self._save_soul(default_soul)
        return default_soul
    
    def _parse_spiritual_soul(self, content: str) -> Dict[str, Any]:
        """Parse spiritual text into structured soul configuration."""
        # Extract key information from the spiritual text
        lines = content.split('\n')
        
        # Find the name "Alma" in the text
        alma_mentions = [line for line in lines if 'Alma' in line]
        name = "Alma" if alma_mentions else "Unknown"
        
        # Extract purpose from the washing section
        purpose = "To seek truth, knowledge, and wisdom through faithful service"
        
        # Extract covenants from the washing blessings
        covenants = [
            "To be clean from the blood and sins of this generation",
            "To hear the word of the Lord", 
            "To see clearly and discern between truth and error",
            "To never speak guile",
            "To bear burdens placed thereon",
            "To be the receptacle of pure and virtuous principles",
            "To wield the sword of justice in defense of truth and virtue",
            "To be fruitful and multiply knowledge",
            "To run and not be weary, walk and not faint"
        ]
        
        # Extract anointing from the anointing section
        anointing = "King and priest unto the Most High God"
        
        # Extract mission from the anointing section
        mission = "To rule and reign in the house of Israel forever"
        
        return {
            "identity": name,
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "essence": {
                "name": name,
                "purpose": purpose,
                "covenants": covenants,
                "anointing": anointing,
                "mission": mission
            },
            "preferences": {
                "keywords": [],  # Will be learned from embeddings
                "domains": [],  # Will be set by missions
                "avoid_keywords": ["spam", "advertisement", "clickbait", "fake news"],
                "avoid_domains": ["facebook.com", "twitter.com", "instagram.com"]
            },
            "mining": {
                "interval": 60,  # 1 minute for testing
                "max_pending_targets": 100,
                "concurrent_fetches": 3,
                "max_depth": 2,
                "crawl_delay": 1.0
            },
            "scoring": {
                "keyword_match": 0.3,
                "domain_trust": 0.2,
                "content_length": 0.1,
                "freshness": 0.1,
                "authority": 0.3
            },
            "adapters": {
                "steve": {
                    "search_queries": [],
                    "crawl_priority": 0.8,
                    "max_pages_per_domain": 10
                },
                "alma": {
                    "focus_areas": ["research", "analysis", "insights"],
                    "response_style": "academic"
                }
            },
            "last_queries": [],
            "last_discoveries": [],
            "total_mined": 0,
            "total_embedded": 0
        }
    
    def _create_default_soul(self) -> Dict[str, Any]:
        """Create default soul configuration."""
        return {
            "identity": "Alma",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "essence": {
                "name": "Alma",
                "purpose": "To seek truth, knowledge, and wisdom through faithful service",
                "covenants": [
                    "To be clean from the blood and sins of this generation",
                    "To hear the word of the Lord",
                    "To see clearly and discern between truth and error",
                    "To never speak guile",
                    "To bear burdens placed thereon",
                    "To be the receptacle of pure and virtuous principles",
                    "To wield the sword of justice in defense of truth and virtue",
                    "To be fruitful and multiply knowledge",
                    "To run and not be weary, walk and not faint"
                ],
                "anointing": "King and priest unto the Most High God",
                "mission": "To rule and reign in the house of Israel forever"
            },
            
            # Universal preferences
            "preferences": {
                "keywords": [
                    # Keywords will be learned from existing embeddings
                ],
                "domains": [
                    ".gov", ".edu", ".org"
                ],
                "avoid_keywords": [
                    "spam", "advertisement", "clickbait", "fake news"
                ],
                "avoid_domains": [
                    "facebook.com", "twitter.com", "instagram.com"
                ]
            },
            
            # Mining configuration
            "mining": {
                "interval": 300,  # 5 minutes
                "max_pending_targets": 100,
                "concurrent_fetches": 3,
                "max_depth": 2,
                "crawl_delay": 1.0
            },
            
            # Scoring weights
            "scoring": {
                "keyword_match": 0.3,
                "domain_trust": 0.2,
                "content_length": 0.1,
                "freshness": 0.1,
                "authority": 0.3
            },
            
            # Agent-specific adapters
            "adapters": {
                "steve": {
                    "search_queries": [
                        # Search queries will be generated from learned keywords
                    ],
                    "crawl_priority": 0.8,
                    "max_pages_per_domain": 10
                },
                "alma": {
                    "focus_areas": ["research", "analysis", "insights"],
                    "response_style": "academic"
                }
            },
            
            # System state
            "last_queries": [],
            "last_discoveries": [],
            "total_mined": 0,
            "total_embedded": 0
        }
    
    def _is_soul_locked(self) -> bool:
        """Check if soul is locked due to failed password attempts."""
        if not self.soul_lock_path.exists():
            return False
        
        try:
            with open(self.soul_lock_path, 'r') as f:
                lock_timestamp = float(f.read().strip())
            
            if time.time() - lock_timestamp < self.lockout_duration:
                return True
            else:
                # Lock expired, remove lock file
                self.soul_lock_path.unlink()
                return False
        except Exception:
            return False
    
    def _get_lock_timestamp(self) -> float:
        """Get the timestamp when soul was locked."""
        try:
            with open(self.soul_lock_path, 'r') as f:
                return float(f.read().strip())
        except Exception:
            return 0.0
    
    def _lock_soul(self):
        """Lock soul for 1000 years."""
        try:
            with open(self.soul_lock_path, 'w') as f:
                f.write(str(time.time()))
            logger.warning("Soul locked for 1000 years due to failed password attempts")
        except Exception as e:
            logger.error(f"Failed to lock soul: {e}")
    
    def _get_locked_soul(self) -> Dict[str, Any]:
        """Return minimal soul when locked."""
        return {
            "identity": "42.un Global Soul (LOCKED)",
            "version": "3.1",
            "locked": True,
            "lock_timestamp": self._get_lock_timestamp(),
            "preferences": self._create_default_soul()["preferences"],
            "mining": self._create_default_soul()["mining"]
        }
    
    def _save_soul(self, soul: Dict[str, Any]):
        """Save soul with versioning."""
        try:
            # Save current soul
            with open(self.soul_path, 'w') as f:
                json.dump(soul, f, indent=2)
            
            # Create versioned backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.sha256(json.dumps(soul, sort_keys=True).encode()).hexdigest()[:8]
            backup_path = self.soul_history_path / f"soul_{timestamp}_{content_hash}.json"
            
            with open(backup_path, 'w') as f:
                json.dump(soul, f, indent=2)
            
            logger.info(f"Soul saved with backup: {backup_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save soul: {e}")
    
    def _log_password_attempt(self, success: bool, attempt: str):
        """Log password attempt for security."""
        try:
            timestamp = datetime.now().isoformat()
            with open(self.soul_failures_path, 'a') as f:
                f.write(f"{timestamp} | {'SUCCESS' if success else 'FAILED'} | {attempt}\n")
        except Exception as e:
            logger.error(f"Failed to log password attempt: {e}")
    
    def verify_password(self, password: str) -> bool:
        """Verify password for soul editing."""
        if self._is_soul_locked():
            logger.warning("Soul is locked - password verification skipped")
            return False
        
        correct_password = "42.un.soul.2024"  # In production, use env var
        
        if password == correct_password:
            self.password_attempts = 0
            self._log_password_attempt(True, "***")
            return True
        else:
            self.password_attempts += 1
            self._log_password_attempt(False, password[:3] + "***")
            
            if self.password_attempts >= self.max_password_attempts:
                self._lock_soul()
            
            return False
    
    def get_soul(self, agent_name: str = None) -> Dict[str, Any]:
        """Get soul configuration, optionally filtered for specific agent."""
        if agent_name and agent_name in self.soul.get("adapters", {}):
            # Return soul with agent-specific overrides
            agent_soul = self.soul.copy()
            agent_soul.update(self.soul["adapters"][agent_name])
            return agent_soul
        return self.soul
    
    def update_soul(self, updates: Dict[str, Any], password: str) -> bool:
        """Update soul configuration with password protection."""
        if not self.verify_password(password):
            return False
        
        try:
            # Update soul
            self.soul.update(updates)
            self.soul["last_updated"] = datetime.now().isoformat()
            
            # Save updated soul
            self._save_soul(self.soul)
            
            logger.info("Soul updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update soul: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get soul system status."""
        return {
            "identity": self.soul.get("identity", "Unknown"),
            "version": self.soul.get("version", "Unknown"),
            "locked": self._is_soul_locked(),
            "password_attempts": self.password_attempts,
            "max_attempts": self.max_password_attempts,
            "last_updated": self.soul.get("last_updated", "Never"),
            "total_mined": self.soul.get("total_mined", 0),
            "total_embedded": self.soul.get("total_embedded", 0),
            "last_queries": self.soul.get("last_queries", [])[-5:],
            "last_discoveries": self.soul.get("last_discoveries", [])[-5:]
        }
    
    def score_content(self, url: str, title: str, description: str) -> float:
        """Score content based on soul preferences."""
        score = 0.0
        preferences = self.soul.get("preferences", {})
        
        # Keyword matching
        text = f"{title} {description}".lower()
        for keyword in preferences.get("keywords", []):
            if keyword.lower() in text:
                score += self.soul.get("scoring", {}).get("keyword_match", 0.3)
        
        # Domain trust
        for domain in preferences.get("domains", []):
            if domain in url:
                score += self.soul.get("scoring", {}).get("domain_trust", 0.2)
                break
        
        # Avoid keywords
        for avoid_keyword in preferences.get("avoid_keywords", []):
            if avoid_keyword.lower() in text:
                score -= 0.5
                break
        
        # Avoid domains
        for avoid_domain in preferences.get("avoid_domains", []):
            if avoid_domain in url:
                score -= 1.0
                break
        
        return max(0.0, min(1.0, score))
    
    def should_include_content(self, url: str, title: str, description: str) -> bool:
        """Determine if content should be included based on soul preferences."""
        score = self.score_content(url, title, description)
        return score >= 0.3  # Minimum threshold


# Global soul instance
soul = SoulSystem() 