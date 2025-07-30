#!/usr/bin/env python3
"""
Universal Intelligence Deployment

Deploy any knowledge sources into 42.un knowledge engine.
This is the abstract, universal approach - not hardcoded to specific missions.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone

# Add 42 package to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "42"))

from un.knowledge_engine import (
    KnowledgeEngine, KnowledgeSource, KnowledgeEvent,
    SourceType, DomainType, TriggerType
)
from un.redis_bus import RedisBus


async def deploy_from_files(sources_file: str = None, events_file: str = None):
    """Deploy intelligence network from JSON files."""
    print("🚀 Deploying Universal Intelligence Network...")
    print("=" * 60)
    
    # Initialize Redis bus and knowledge engine
    redis_bus = RedisBus()
    engine = KnowledgeEngine(redis_bus)
    
    # Load sources from file or create default
    if sources_file and os.path.exists(sources_file):
        print(f"📚 Loading sources from: {sources_file}")
        sources = load_sources_from_file(sources_file)
    else:
        print("📚 Creating default universal sources...")
        sources = create_default_sources()
    
    # Add sources to engine
    for source in sources:
        engine.add_source(source)
        print(f"  ✅ {source.name} ({source.type.value})")
    
    print(f"\n📊 Total sources loaded: {len(sources)}")
    
    # Load events from file or create default
    if events_file and os.path.exists(events_file):
        print(f"\n🎯 Loading events from: {events_file}")
        events = load_events_from_file(events_file)
    else:
        print("\n🎯 Creating default universal triggers...")
        events = create_default_events()
    
    # Add events to engine
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
    print("  📡 Universal intelligence gathering active")
    print("  🎯 Ready for any domain or mission")
    print("\n" + "=" * 60)
    
    # Run monitoring cycle
    await engine.run_fetch_cycle()
    
    print("\n✅ Universal intelligence network deployed and running!")
    print("🌍 42.un is now monitoring the world for any intelligence")


def create_default_sources():
    """Create default universal sources."""
    return [
        KnowledgeSource(
            id="universal_rss_news",
            name="Universal News Feed",
            type=SourceType.RSS,
            domain=DomainType.RESEARCH,
            url="https://feeds.bbci.co.uk/news/rss.xml",
            frequency="5min",
            parser="xml",
            active=True,
            metadata={"description": "Universal news monitoring"}
        ),
        KnowledgeSource(
            id="universal_api_weather",
            name="Universal Weather API",
            type=SourceType.API,
            domain=DomainType.WEATHER,
            url="https://api.weather.gov/alerts/active",
            frequency="5min",
            parser="json",
            active=True,
            metadata={
                "description": "Universal weather monitoring",
                "headers": {"Accept": "application/geo+json"}
            }
        )
    ]


def create_default_events():
    """Create default universal triggers."""
    return [
        KnowledgeEvent(
            id="universal_alert",
            name="Universal Alert",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="alert",
            priority=1,
            actions=["notify", "log", "analyze"],
            metadata={"description": "Universal alert detection"}
        ),
        KnowledgeEvent(
            id="universal_emergency",
            name="Universal Emergency",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="emergency",
            priority=1,
            actions=["escalate", "notify", "coordinate"],
            metadata={"description": "Universal emergency detection"}
        ),
        KnowledgeEvent(
            id="universal_crisis",
            name="Universal Crisis",
            trigger_type=TriggerType.KEYWORD,
            trigger_value="crisis",
            priority=1,
            actions=["assess", "mobilize", "coordinate"],
            metadata={"description": "Universal crisis detection"}
        )
    ]


def load_sources_from_file(filename: str) -> list:
    """Load sources from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    sources = []
    for item in data:
        source = KnowledgeSource.from_dict(item)
        sources.append(source)
    
    return sources


def load_events_from_file(filename: str) -> list:
    """Load events from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    events = []
    for item in data:
        event = KnowledgeEvent.from_dict(item)
        events.append(event)
    
    return events


def save_sources_to_file(sources):
    """Save sources to JSON file."""
    data = [source.to_dict() for source in sources]
    with open("universal_sources.json", "w") as f:
        json.dump(data, f, indent=2)


def save_events_to_file(events):
    """Save events to JSON file."""
    data = [event.to_dict() for event in events]
    with open("universal_events.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Universal Intelligence Network")
    parser.add_argument("--sources", help="Path to sources JSON file")
    parser.add_argument("--events", help="Path to events JSON file")
    
    args = parser.parse_args()
    
    asyncio.run(deploy_from_files(args.sources, args.events)) 