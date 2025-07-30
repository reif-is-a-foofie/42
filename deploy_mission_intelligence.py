#!/usr/bin/env python3
"""
Deploy Matthew 25:35 Intelligence Network

Directly inject the curated intelligence sources into 42.un knowledge engine
and start real-time monitoring for humanitarian response.
"""

import asyncio
import json
from datetime import datetime, timezone

# Add 42 package to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "42"))

from un.knowledge_engine import (
    KnowledgeEngine, KnowledgeSource, KnowledgeEvent,
    SourceType, DomainType, TriggerType
)
from un.redis_bus import RedisBus


# Matthew 25:35 Intelligence Sources
MISSION_SOURCES = [
    # GitHub Repos - Deployable Tech
    {
        "id": "github_opendronemap",
        "name": "OpenDroneMap - Disaster Mapping",
        "type": "github",
        "domain": "geospatial",
        "url": "https://github.com/OpenDroneMap/ODM",
        "frequency": "1hour",
        "parser": "github",
        "description": "Generate 3D maps from drone imagery for disaster zones"
    },
    {
        "id": "github_nasa_impact", 
        "name": "NASA-IMPACT - Disaster Modeling",
        "type": "github",
        "domain": "geospatial",
        "url": "https://github.com/NASA-IMPACT",
        "frequency": "1hour",
        "parser": "github",
        "description": "Earth observation & climate impact analytics"
    },
    {
        "id": "github_hotosm",
        "name": "HOTOSM - Crisis Data Pipelines",
        "type": "github", 
        "domain": "humanitarian",
        "url": "https://github.com/hotosm",
        "frequency": "1hour",
        "parser": "github",
        "description": "OpenStreetMap for humanitarian disasters"
    },
    {
        "id": "github_meteostat",
        "name": "Meteostat - Weather Analytics",
        "type": "github",
        "domain": "weather",
        "url": "https://github.com/meteostat/meteostat-python",
        "frequency": "1hour",
        "parser": "github",
        "description": "Historical & live weather for prediction"
    },
    {
        "id": "github_ifrcgo",
        "name": "IFRCGo - Humanitarian Tech",
        "type": "github",
        "domain": "humanitarian",
        "url": "https://github.com/IFRCGo",
        "frequency": "1hour",
        "parser": "github",
        "description": "Red Cross disaster response platform"
    },
    {
        "id": "github_openfoodnetwork",
        "name": "OpenFoodNetwork - Food Logistics",
        "type": "github",
        "domain": "humanitarian",
        "url": "https://github.com/openfoodfoundation/openfoodnetwork",
        "frequency": "1hour",
        "parser": "github",
        "description": "Local food distribution networks"
    },
    {
        "id": "github_openwaterproject",
        "name": "OpenWaterProject - Water Tech",
        "type": "github",
        "domain": "humanitarian",
        "url": "https://github.com/OpenWaterProject",
        "frequency": "1hour",
        "parser": "github",
        "description": "Water filtration & sensor IoT"
    },
    {
        "id": "github_langchain",
        "name": "LangChain - AI Crisis Assistance",
        "type": "github",
        "domain": "technology",
        "url": "https://github.com/langchain-ai/langchain",
        "frequency": "1hour",
        "parser": "github",
        "description": "Enable Alma to answer field questions"
    },
    {
        "id": "github_qdrant",
        "name": "Qdrant - Vector Intelligence",
        "type": "github",
        "domain": "technology",
        "url": "https://github.com/qdrant/qdrant",
        "frequency": "1hour",
        "parser": "github",
        "description": "Core for searchable knowledge & ops memory"
    },
    {
        "id": "github_fastapi",
        "name": "FastAPI - Deployable Microservices",
        "type": "github",
        "domain": "technology",
        "url": "https://github.com/tiangolo/fastapi",
        "frequency": "1hour",
        "parser": "github",
        "description": "Backbone for deployable microservices"
    },
    
    # RSS Feeds - Real-World Intelligence
    {
        "id": "rss_usgs_earthquakes",
        "name": "USGS Earthquakes",
        "type": "rss",
        "domain": "geospatial",
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.atom",
        "frequency": "5min",
        "parser": "xml",
        "description": "Instant earthquake alerts for deployment"
    },
    {
        "id": "rss_noaa_hurricanes",
        "name": "NOAA/NHC Hurricanes",
        "type": "rss",
        "domain": "weather",
        "url": "https://www.nhc.noaa.gov/rss_examples.xml",
        "frequency": "5min",
        "parser": "xml",
        "description": "Prepare basecamps before landfall"
    },
    {
        "id": "rss_nifc_wildfires",
        "name": "NIFC Wildfires",
        "type": "rss",
        "domain": "crisis",
        "url": "https://inciweb.nwcg.gov/feeds/rss/incidents/",
        "frequency": "5min",
        "parser": "xml",
        "description": "Direct wildfire incident intel"
    },
    {
        "id": "rss_usgs_floods",
        "name": "USGS WaterWatch",
        "type": "rss",
        "domain": "weather",
        "url": "https://waterwatch.usgs.gov/rss/",
        "frequency": "5min",
        "parser": "xml",
        "description": "Early signal for flood relief"
    },
    {
        "id": "rss_fews_net",
        "name": "FEWS NET - Famine Warning",
        "type": "rss",
        "domain": "humanitarian",
        "url": "https://fews.net/rss.xml",
        "frequency": "5min",
        "parser": "xml",
        "description": "Detect regions at risk for food aid"
    },
    {
        "id": "rss_who_outbreaks",
        "name": "WHO Disease Outbreaks",
        "type": "rss",
        "domain": "medical",
        "url": "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
        "frequency": "5min",
        "parser": "xml",
        "description": "Early pandemic/epidemic signals"
    },
    {
        "id": "rss_reliefweb",
        "name": "ReliefWeb - Humanitarian Calls",
        "type": "rss",
        "domain": "humanitarian",
        "url": "https://reliefweb.int/disasters/rss.xml",
        "frequency": "5min",
        "parser": "xml",
        "description": "Global disaster alerts & relief calls"
    },
    {
        "id": "rss_defense_one",
        "name": "Defense One - Logistics",
        "type": "rss",
        "domain": "intelligence",
        "url": "https://www.defenseone.com/rss/all/",
        "frequency": "5min",
        "parser": "xml",
        "description": "Track gov mobilization patterns"
    },
    {
        "id": "rss_sam_gov",
        "name": "SAM.gov - Emergency Contracts",
        "type": "rss",
        "domain": "intelligence",
        "url": "https://sam.gov/data-services",
        "frequency": "5min",
        "parser": "xml",
        "description": "Identify deployable contract opportunities"
    },
    {
        "id": "rss_techcrunch_startups",
        "name": "TechCrunch Startups - Crisis Tech",
        "type": "rss",
        "domain": "technology",
        "url": "https://feeds.feedburner.com/TechCrunch/startups",
        "frequency": "5min",
        "parser": "xml",
        "description": "Scout deployable innovations"
    }
]

# Mission Triggers - Matthew 25:35 Response Actions
MISSION_TRIGGERS = [
    {
        "id": "trigger_earthquake",
        "name": "Earthquake Response",
        "trigger_type": "keyword",
        "trigger_value": "earthquake",
        "priority": 1,
        "actions": ["dispatch_team", "assess_damage", "coordinate_relief"],
        "description": "Activate nearest response team for earthquake"
    },
    {
        "id": "trigger_hurricane",
        "name": "Hurricane Response", 
        "trigger_type": "keyword",
        "trigger_value": "hurricane",
        "priority": 1,
        "actions": ["deploy_basecamp", "notify_teams", "update_inventory"],
        "description": "Pre-stage GFP assets before landfall"
    },
    {
        "id": "trigger_wildfire",
        "name": "Wildfire Response",
        "trigger_type": "keyword", 
        "trigger_value": "wildfire",
        "priority": 1,
        "actions": ["deploy_fire_team", "evacuation_support", "air_quality_monitor"],
        "description": "Deploy fire response and evacuation support"
    },
    {
        "id": "trigger_flood",
        "name": "Flood Response",
        "trigger_type": "keyword",
        "trigger_value": "flood",
        "priority": 1,
        "actions": ["deploy_water_rescue", "distribute_supplies", "coordinate_shelters"],
        "description": "Deploy water rescue and supply distribution"
    },
    {
        "id": "trigger_famine",
        "name": "Famine Response",
        "trigger_type": "keyword",
        "trigger_value": "famine",
        "priority": 1,
        "actions": ["mobilize_food_aid", "coordinate_logistics", "monitor_situation"],
        "description": "Deploy food distribution network"
    },
    {
        "id": "trigger_outbreak",
        "name": "Disease Outbreak Response",
        "trigger_type": "keyword",
        "trigger_value": "outbreak",
        "priority": 1,
        "actions": ["deploy_medical_team", "coordinate_healthcare", "track_spread"],
        "description": "Activate medical response protocol"
    },
    {
        "id": "trigger_disaster",
        "name": "General Disaster Response",
        "trigger_type": "keyword",
        "trigger_value": "disaster",
        "priority": 1,
        "actions": ["assess_situation", "deploy_teams", "coordinate_relief"],
        "description": "General disaster response protocol"
    },
    {
        "id": "trigger_emergency",
        "name": "Emergency Response",
        "trigger_type": "keyword",
        "trigger_value": "emergency",
        "priority": 1,
        "actions": ["activate_protocol", "deploy_assets", "notify_stakeholders"],
        "description": "Emergency response activation"
    },
    {
        "id": "trigger_crisis",
        "name": "Crisis Response",
        "trigger_type": "keyword",
        "trigger_value": "crisis",
        "priority": 1,
        "actions": ["crisis_assessment", "resource_mobilization", "stakeholder_coordination"],
        "description": "Crisis response and resource mobilization"
    },
    {
        "id": "trigger_evacuation",
        "name": "Evacuation Response",
        "trigger_type": "keyword",
        "trigger_value": "evacuation",
        "priority": 1,
        "actions": ["coordinate_evacuation", "provide_transport", "establish_shelters"],
        "description": "Evacuation coordination and shelter establishment"
    }
]


def create_mission_sources():
    """Create KnowledgeSource objects from mission data."""
    sources = []
    for source_data in MISSION_SOURCES:
        source = KnowledgeSource(
            id=source_data["id"],
            name=source_data["name"],
            type=SourceType(source_data["type"]),
            domain=DomainType(source_data["domain"]),
            url=source_data["url"],
            frequency=source_data["frequency"],
            parser=source_data["parser"],
            active=True,
            metadata={"description": source_data["description"]}
        )
        sources.append(source)
    return sources


def create_mission_triggers():
    """Create KnowledgeEvent objects from mission triggers."""
    events = []
    for trigger_data in MISSION_TRIGGERS:
        event = KnowledgeEvent(
            id=trigger_data["id"],
            name=trigger_data["name"],
            trigger_type=TriggerType(trigger_data["trigger_type"]),
            trigger_value=trigger_data["trigger_value"],
            priority=trigger_data["priority"],
            actions=trigger_data["actions"],
            metadata={"description": trigger_data["description"]}
        )
        events.append(event)
    return events


async def deploy_mission_intelligence():
    """Deploy the Matthew 25:35 intelligence network."""
    print("🚀 Deploying Matthew 25:35 Intelligence Network...")
    print("=" * 60)
    
    # Initialize Redis bus and knowledge engine
    redis_bus = RedisBus()
    engine = KnowledgeEngine(redis_bus)
    
    # Create and add mission sources
    print("📚 Loading mission intelligence sources...")
    sources = create_mission_sources()
    for source in sources:
        engine.add_source(source)
        print(f"  ✅ {source.name} ({source.type.value})")
    
    print(f"\n📊 Total sources loaded: {len(sources)}")
    
    # Create and add mission triggers
    print("\n🎯 Loading mission response triggers...")
    events = create_mission_triggers()
    for event in events:
        engine.add_event(event)
        print(f"  ✅ {event.name} ({event.trigger_type.value}: {event.trigger_value})")
    
    print(f"\n📊 Total triggers loaded: {len(events)}")
    
    # Save to database files
    print("\n💾 Saving to knowledge database...")
    save_sources_to_file(sources)
    save_events_to_file(events)
    print("  ✅ Sources and triggers saved to database")
    
    # Start monitoring
    print("\n⏰ Starting real-time monitoring...")
    print("  🔄 Monitoring interval: 5 minutes")
    print("  📡 Watching for: earthquakes, hurricanes, wildfires, floods, famine, outbreaks")
    print("  🎯 Ready for Matthew 25:35 response actions")
    print("\n" + "=" * 60)
    
    # Run monitoring cycle
    await engine.run_fetch_cycle()
    
    print("\n✅ Mission intelligence network deployed and running!")
    print("🌍 42.un is now monitoring for humanitarian response opportunities")


def save_sources_to_file(sources):
    """Save sources to JSON file."""
    data = [source.to_dict() for source in sources]
    with open("mission_sources.json", "w") as f:
        json.dump(data, f, indent=2)


def save_events_to_file(events):
    """Save events to JSON file."""
    data = [event.to_dict() for event in events]
    with open("mission_triggers.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(deploy_mission_intelligence()) 