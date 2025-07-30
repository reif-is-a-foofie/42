"""
Mission Configuration for 42.un - Matthew 25:35 Intelligence Network

This module defines the curated intelligence sources for humanitarian response,
disaster relief, and crisis anticipation aligned with the mission of serving
those in need.
"""

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