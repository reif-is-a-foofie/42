"""
Mission Configuration for 42.un - Matthew 25:35 Intelligence Network

This module defines the curated intelligence sources for humanitarian response,
disaster relief, and crisis anticipation aligned with the mission of serving
those in need.
"""

import time
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

# GitHub Repositories - Deployable Tech Intelligence
GITHUB_MISSION_REPOS = {
    "disaster_mapping": [
        "OpenDroneMap/ODM",  # 3D maps from drone imagery
        "NASA-IMPACT/NASA-IMPACT",  # Earth observation analytics
        "hotosm/hotosm",  # OpenStreetMap for humanitarian disasters
    ],
    "weather_risk": [
        "meteostat/meteostat-python",  # Historical & live weather
        "openweathermap/api",  # Real-time weather data
    ],
    "humanitarian_tech": [
        "IFRCGo/IFRCGo",  # Red Cross disaster response
        "openfoodfoundation/openfoodnetwork",  # Food distribution
        "OpenWaterProject/openwaterproject",  # Water purification
    ],
    "ai_assistance": [
        "langchain-ai/langchain",  # Enable Alma field assistance
        "qdrant/qdrant",  # Vector intelligence core
        "tiangolo/fastapi",  # Deployable microservices
    ],
    "crisis_analytics": [
        "UNOCHA/hdx-python-library",  # Humanitarian data exchange
        "ReliefWeb/reliefweb-api",  # Disaster alerts
        "WHO/WHO-COVID-19",  # Disease outbreak tracking
    ]
}

# RSS Feeds - Real-World Crisis Intelligence
RSS_MISSION_FEEDS = {
    "natural_disasters": {
        "earthquakes": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.atom",
        "hurricanes": "https://www.nhc.noaa.gov/rss_examples.xml",
        "wildfires": "https://inciweb.nwcg.gov/feeds/rss/incidents/",
        "floods": "https://waterwatch.usgs.gov/rss/",
    },
    "humanitarian_crises": {
        "famine_warning": "https://fews.net/rss.xml",
        "disease_outbreaks": "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
        "relief_calls": "https://reliefweb.int/disasters/rss.xml",
    },
    "logistics_intelligence": {
        "defense_logistics": "https://www.defenseone.com/rss/all/",
        "emergency_contracts": "https://sam.gov/data-services",
        "crisis_tech": "https://feeds.feedburner.com/TechCrunch/startups",
    }
}

# API Endpoints - Real-Time Data Sources
API_MISSION_ENDPOINTS = {
    "weather": {
        "noaa_weather": {
            "url": "https://api.weather.gov/alerts/active",
            "method": "GET",
            "headers": {"Accept": "application/geo+json"},
            "description": "NOAA weather alerts for disaster preparation"
        },
        "usgs_earthquakes": {
            "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
            "method": "GET",
            "description": "Real-time earthquake data"
        }
    },
    "humanitarian": {
        "reliefweb": {
            "url": "https://api.reliefweb.int/v1/disasters",
            "method": "GET",
            "headers": {"Accept": "application/json"},
            "description": "Global disaster alerts"
        },
        "who_outbreaks": {
            "url": "https://www.who.int/csr/don/en/",
            "method": "GET",
            "description": "Disease outbreak intelligence"
        }
    }
}

# File System Monitoring - Local Intelligence
FILESYSTEM_MISSION_PATHS = {
    "deployment_configs": [
        "/opt/42/deployments",  # Basecamp configurations
        "/opt/42/assets",  # GFP asset inventory
        "/opt/42/intelligence",  # Local intelligence reports
    ],
    "field_data": [
        "/opt/42/field_reports",  # Field team reports
        "/opt/42/sensor_data",  # IoT sensor feeds
        "/opt/42/logistics",  # Supply chain data
    ]
}

# Mission Event Types
MISSION_EVENT_TYPES = {
    "CRISIS_DETECTED": "Natural disaster or humanitarian crisis detected",
    "DEPLOYMENT_READY": "Technology or assets ready for deployment",
    "INTELLIGENCE_UPDATE": "New intelligence or threat assessment",
    "RESOURCE_ALERT": "Critical resource or supply chain update",
    "FIELD_REPORT": "Field team report or situation update",
    "PREDICTION_TRIGGER": "Predictive model triggers response",
}

# Response Templates
RESPONSE_TEMPLATES = {
    "hurricane_alert": {
        "trigger": "hurricane",
        "response": "Hurricane {name} detected - {hours} hours to landfall. Pre-stage GFP assets in {counties}.",
        "actions": ["deploy_basecamp", "notify_teams", "update_inventory"]
    },
    "earthquake_response": {
        "trigger": "earthquake",
        "response": "Earthquake {magnitude} at {location}. Activate nearest response team.",
        "actions": ["dispatch_team", "assess_damage", "coordinate_relief"]
    },
    "famine_warning": {
        "trigger": "famine",
        "response": "Famine risk detected in {region}. Deploy food distribution network.",
        "actions": ["mobilize_food_aid", "coordinate_logistics", "monitor_situation"]
    },
    "disease_outbreak": {
        "trigger": "outbreak",
        "response": "Disease outbreak in {location}. Activate medical response protocol.",
        "actions": ["deploy_medical_team", "coordinate_healthcare", "track_spread"]
    }
}

# Mission Configuration
MISSION_CONFIG = {
    "scan_interval": 300,  # 5 minutes for crisis monitoring
    "max_errors": 3,
    "github": {
        "repos": [repo for category in GITHUB_MISSION_REPOS.values() for repo in category],
        "webhook_secret": "mission-critical-secret",
        "api_token": "github-token-for-mission-repos"
    },
    "rss": {
        "feeds": [feed for category in RSS_MISSION_FEEDS.values() for feed in category.values()]
    },
    "api": {
        "endpoints": [endpoint for category in API_MISSION_ENDPOINTS.values() for endpoint in category.values()]
    },
    "filesystem": {
        "watch_directories": [path for category in FILESYSTEM_MISSION_PATHS.values() for path in category],
        "ignore_patterns": ["*.tmp", "*.log", "*.bak"]
    },
    "webhooks": {
        "mission_webhook": {
            "secret": "mission-webhook-secret",
            "algorithm": "sha256"
        }
    }
}

# Mission Orchestrator - Calls Steve with objectives
class MissionOrchestrator:
    """Orchestrates missions by calling Steve with specific objectives."""
    
    def __init__(self, redis_bus, steve_instance):
        self.redis_bus = redis_bus
        self.steve = steve_instance
        self.active_missions = []
        
    def create_mission(self, mission_type: str, objective: str, 
                      keywords: List[str] = None, domains: List[str] = None,
                      priority: int = 5) -> str:
        """Create a mission and immediately assign it to Steve."""
        
        mission = {
            "id": f"mission_{int(time.time())}",
            "type": mission_type,
            "objective": objective,
            "keywords": keywords or [],
            "domains": domains or [],
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Add to active missions
        self.active_missions.append(mission)
        
        # Call Steve with this mission
        self._assign_mission_to_steve(mission)
        
        # Publish event
        from .events import Event, EventType
        try:
            event = Event(
                event_type=EventType.KNOWLEDGE_ADDED,
                source_id=mission["id"],
                data=mission
            )
            self.redis_bus.publish_event(event)
        except Exception as e:
            logger.warning(f"Failed to publish mission event: {e}")
        
        logger.info(f"Created and assigned mission: {mission['id']} - {objective}")
        return mission["id"]
    
    def _assign_mission_to_steve(self, mission: Dict[str, Any]):
        """Assign mission to Steve by updating its search parameters."""
        
        try:
            # Import the soul system
            from ..soul import soul
            
            # Prepare soul updates
            soul_updates = {
                "preferences": {
                    "keywords": mission.get("keywords", []),
                    "domains": mission.get("domains", [])
                },
                "current_mission": {
                    "id": mission["id"],
                    "objective": mission["objective"],
                    "type": mission["type"],
                    "priority": mission["priority"]
                }
            }
            
            # Update soul using the soul system (bypass password for internal updates)
            soul.soul.update(soul_updates)
            soul.soul["last_updated"] = datetime.now().isoformat()
            soul._save_soul(soul.soul)
            
            # Also update Steve's in-memory soul
            if "preferences" not in self.steve.soul:
                self.steve.soul["preferences"] = {}
            
            self.steve.soul["preferences"]["keywords"] = mission.get("keywords", [])
            self.steve.soul["preferences"]["domains"] = mission.get("domains", [])
            
            if "current_mission" not in self.steve.soul:
                self.steve.soul["current_mission"] = {}
            
            self.steve.soul["current_mission"] = {
                "id": mission["id"],
                "objective": mission["objective"],
                "type": mission["type"],
                "priority": mission["priority"]
            }
            
            logger.info(f"Assigned mission {mission['id']} to Steve")
            logger.info(f"Steve will now search for: {mission['objective']}")
            logger.info(f"Keywords: {mission.get('keywords', [])}")
            logger.info(f"Domains: {mission.get('domains', [])}")
            
        except Exception as e:
            logger.error(f"Failed to assign mission to Steve: {e}")
            # Fallback to just updating Steve's in-memory soul
            if "preferences" not in self.steve.soul:
                self.steve.soul["preferences"] = {}
            
            self.steve.soul["preferences"]["keywords"] = mission.get("keywords", [])
            self.steve.soul["preferences"]["domains"] = mission.get("domains", [])
            
            if "current_mission" not in self.steve.soul:
                self.steve.soul["current_mission"] = {}
            
            self.steve.soul["current_mission"] = {
                "id": mission["id"],
                "objective": mission["objective"],
                "type": mission["type"],
                "priority": mission["priority"]
            }
    
    def get_active_missions(self) -> List[Dict[str, Any]]:
        """Get all active missions."""
        return self.active_missions
    
    def clear_missions(self):
        """Clear all missions and reset Steve to default state."""
        self.active_missions = []
        
        # Reset Steve's soul to defaults
        if "preferences" in self.steve.soul:
            self.steve.soul["preferences"]["keywords"] = []
            self.steve.soul["preferences"]["domains"] = []
        
        if "current_mission" in self.steve.soul:
            del self.steve.soul["current_mission"]
        
        logger.info("Cleared all missions - Steve reset to default state") 