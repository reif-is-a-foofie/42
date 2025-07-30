"""
CLI commands for 42.un Knowledge Engine

Provides command-line interface for managing knowledge sources,
triggers, and monitoring the abstract knowledge engine.
"""

import asyncio
import typer
from typing import Optional, List
from loguru import logger

from .knowledge_engine import (
    KnowledgeEngine, KnowledgeSource, KnowledgeEvent,
    SourceType, DomainType, TriggerType
)
from .redis_bus import RedisBus
from .mission_config import MISSION_CONFIG


app = typer.Typer(help="42.un Knowledge Engine CLI")


@app.command()
def start_monitoring(
    interval: int = typer.Option(300, "--interval", "-i", help="Monitoring interval in seconds"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources JSON file"),
    events: Optional[str] = typer.Option(None, "--events", "-e", help="Path to events JSON file")
):
    """Start the knowledge engine monitoring."""
    typer.echo("🚀 Starting 42.un Knowledge Engine...")
    
    async def run_monitoring():
        # Initialize Redis bus
        redis_bus = RedisBus()
        
        # Initialize knowledge engine
        engine = KnowledgeEngine(redis_bus)
        
        # Load mission sources if no custom sources provided
        if not sources:
            typer.echo("📚 Loading mission sources...")
            await load_mission_sources(engine)
        else:
            typer.echo(f"📚 Loading sources from: {sources}")
            await load_sources_from_file(engine, sources)
        
        # Load mission events if no custom events provided
        if not events:
            typer.echo("🎯 Loading mission triggers...")
            await load_mission_events(engine)
        else:
            typer.echo(f"🎯 Loading events from: {events}")
            await load_events_from_file(engine, events)
        
        typer.echo(f"⏰ Starting monitoring with {len(engine.sources)} sources and {len(engine.events)} triggers")
        typer.echo(f"🔄 Interval: {interval} seconds")
        
        # Start monitoring
        await engine.start_monitoring(interval)
    
    asyncio.run(run_monitoring())


@app.command()
def add_source(
    name: str = typer.Argument(..., help="Source name"),
    url: str = typer.Argument(..., help="Source URL"),
    source_type: str = typer.Option("rss", "--type", "-t", help="Source type (rss, api, github)"),
    domain: str = typer.Option("research", "--domain", "-d", help="Knowledge domain"),
    frequency: str = typer.Option("5min", "--frequency", "-f", help="Update frequency"),
    active: bool = typer.Option(True, "--active/--inactive", help="Source active status")
):
    """Add a new knowledge source."""
    typer.echo(f"➕ Adding knowledge source: {name}")
    
    try:
        source = KnowledgeSource(
            id=f"src_{name.lower().replace(' ', '_')}",
            name=name,
            type=SourceType(source_type.lower()),
            domain=DomainType(domain.lower()),
            url=url,
            frequency=frequency,
            parser=source_type.lower(),
            active=active
        )
        
        # Save to file (placeholder - would integrate with database)
        save_source_to_file(source)
        typer.echo(f"✅ Added source: {source.id}")
        
    except Exception as e:
        typer.echo(f"❌ Error adding source: {e}")
        raise typer.Exit(1)


@app.command()
def add_trigger(
    name: str = typer.Argument(..., help="Trigger name"),
    trigger_type: str = typer.Argument(..., help="Trigger type (keyword, threshold, pattern)"),
    trigger_value: str = typer.Argument(..., help="Trigger value"),
    priority: int = typer.Option(1, "--priority", "-p", help="Trigger priority"),
    actions: Optional[List[str]] = typer.Option(None, "--actions", "-a", help="Actions to take")
):
    """Add a new knowledge trigger."""
    typer.echo(f"🎯 Adding knowledge trigger: {name}")
    
    try:
        event = KnowledgeEvent(
            id=f"evt_{name.lower().replace(' ', '_')}",
            name=name,
            trigger_type=TriggerType(trigger_type.lower()),
            trigger_value=trigger_value,
            priority=priority,
            actions=actions or []
        )
        
        # Save to file (placeholder - would integrate with database)
        save_event_to_file(event)
        typer.echo(f"✅ Added trigger: {event.id}")
        
    except Exception as e:
        typer.echo(f"❌ Error adding trigger: {e}")
        raise typer.Exit(1)


@app.command()
def list_sources():
    """List all knowledge sources."""
    typer.echo("📚 Knowledge Sources:")
    typer.echo("=" * 50)
    
    # Load from file (placeholder - would integrate with database)
    sources = load_sources_from_file()
    
    for source in sources:
        status = "🟢 Active" if source.active else "🔴 Inactive"
        typer.echo(f"{source.name} ({source.id})")
        typer.echo(f"  Type: {source.type.value}")
        typer.echo(f"  Domain: {source.domain.value}")
        typer.echo(f"  URL: {source.url}")
        typer.echo(f"  Status: {status}")
        typer.echo()


@app.command()
def list_triggers():
    """List all knowledge triggers."""
    typer.echo("🎯 Knowledge Triggers:")
    typer.echo("=" * 50)
    
    # Load from file (placeholder - would integrate with database)
    events = load_events_from_file()
    
    for event in events:
        typer.echo(f"{event.name} ({event.id})")
        typer.echo(f"  Type: {event.trigger_type.value}")
        typer.echo(f"  Value: {event.trigger_value}")
        typer.echo(f"  Priority: {event.priority}")
        typer.echo(f"  Actions: {', '.join(event.actions)}")
        typer.echo()


@app.command()
def test_source(
    source_id: str = typer.Argument(..., help="Source ID to test")
):
    """Test a knowledge source."""
    typer.echo(f"🧪 Testing source: {source_id}")
    
    async def test_source_async():
        # Load source
        sources = load_sources_from_file()
        source = next((s for s in sources if s.id == source_id), None)
        
        if not source:
            typer.echo(f"❌ Source not found: {source_id}")
            return
        
        # Initialize fetcher
        import aiohttp
        async with aiohttp.ClientSession() as session:
            if source.type == SourceType.RSS:
                from .knowledge_engine import RSSFetcher
                fetcher = RSSFetcher(session)
            elif source.type == SourceType.API:
                from .knowledge_engine import APIFetcher
                fetcher = APIFetcher(session)
            else:
                typer.echo(f"❌ No fetcher for type: {source.type}")
                return
            
            # Test fetch
            documents = await fetcher.fetch(source)
            typer.echo(f"✅ Fetched {len(documents)} documents")
            
            for i, doc in enumerate(documents[:3]):  # Show first 3
                typer.echo(f"  {i+1}. {doc.content[:100]}...")
    
    asyncio.run(test_source_async())


# Helper functions
async def load_mission_sources(engine: KnowledgeEngine):
    """Load mission sources from configuration."""
    from .mission_config import GITHUB_MISSION_REPOS, RSS_MISSION_FEEDS, API_MISSION_ENDPOINTS
    
    # Add RSS feeds
    for category, feeds in RSS_MISSION_FEEDS.items():
        for name, url in feeds.items():
            source = KnowledgeSource(
                id=f"mission_rss_{name}",
                name=f"Mission RSS: {name}",
                type=SourceType.RSS,
                domain=DomainType.CRISIS if category == "humanitarian_crises" else DomainType.GEOSPATIAL,
                url=url,
                frequency="5min",
                parser="xml",
                active=True
            )
            engine.add_source(source)
    
    # Add API endpoints
    for category, endpoints in API_MISSION_ENDPOINTS.items():
        for name, config in endpoints.items():
            source = KnowledgeSource(
                id=f"mission_api_{name}",
                name=f"Mission API: {name}",
                type=SourceType.API,
                domain=DomainType.WEATHER if category == "weather" else DomainType.HUMANITARIAN,
                url=config["url"],
                frequency="5min",
                parser="json",
                active=True,
                metadata={"headers": config.get("headers", {})}
            )
            engine.add_source(source)


async def load_mission_events(engine: KnowledgeEngine):
    """Load mission events from configuration."""
    from .mission_config import RESPONSE_TEMPLATES
    
    for template_id, template in RESPONSE_TEMPLATES.items():
        event = KnowledgeEvent(
            id=f"mission_{template_id}",
            name=f"Mission: {template_id.replace('_', ' ').title()}",
            trigger_type=TriggerType.KEYWORD,
            trigger_value=template["trigger"],
            priority=1,
            actions=template["actions"]
        )
        engine.add_event(event)


def save_source_to_file(source: KnowledgeSource):
    """Save source to file (placeholder)."""
    import json
    import os
    
    filename = "knowledge_sources.json"
    sources = []
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            sources = json.load(f)
    
    sources.append(source.to_dict())
    
    with open(filename, 'w') as f:
        json.dump(sources, f, indent=2)


def save_event_to_file(event: KnowledgeEvent):
    """Save event to file (placeholder)."""
    import json
    import os
    
    filename = "knowledge_events.json"
    events = []
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            events = json.load(f)
    
    events.append(event.to_dict())
    
    with open(filename, 'w') as f:
        json.dump(events, f, indent=2)


def load_sources_from_file(filename: str = "knowledge_sources.json") -> List[KnowledgeSource]:
    """Load sources from file (placeholder)."""
    import json
    import os
    
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return [KnowledgeSource.from_dict(item) for item in data]


def load_events_from_file(filename: str = "knowledge_events.json") -> List[KnowledgeEvent]:
    """Load events from file (placeholder)."""
    import json
    import os
    
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return [KnowledgeEvent.from_dict(item) for item in data]


async def load_sources_from_file(engine: KnowledgeEngine, filename: str):
    """Load sources from file into engine."""
    sources = load_sources_from_file(filename)
    for source in sources:
        engine.add_source(source)


async def load_events_from_file(engine: KnowledgeEngine, filename: str):
    """Load events from file into engine."""
    events = load_events_from_file(filename)
    for event in events:
        engine.add_event(event)


if __name__ == "__main__":
    app() 