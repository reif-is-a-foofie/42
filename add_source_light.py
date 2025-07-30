#!/usr/bin/env python3
"""Lightweight CLI for adding knowledge sources."""

import json
import os
import sys
import asyncio
import subprocess
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class SourceType(Enum):
    RSS = "rss"
    API = "api"
    GITHUB = "github"


class DomainType(Enum):
    WEATHER = "weather"
    MEDICAL = "medical"
    FINANCE = "finance"
    RESEARCH = "research"
    LOGISTICS = "logistics"
    GEOSPATIAL = "geospatial"
    HUMANITARIAN = "humanitarian"
    INTELLIGENCE = "intelligence"
    TECHNOLOGY = "technology"
    CRISIS = "crisis"


@dataclass
class KnowledgeSource:
    id: str
    name: str
    type: SourceType
    domain: DomainType
    url: str
    frequency: str = "5min"
    parser: str = ""
    auth_ref: Optional[str] = None
    vectorize: bool = True
    active: bool = True
    metadata: Optional[dict] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "domain": self.domain.value,
            "url": self.url,
            "frequency": self.frequency,
            "parser": self.parser,
            "auth_ref": self.auth_ref,
            "vectorize": self.vectorize,
            "active": self.active,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            name=data["name"],
            type=SourceType(data["type"]),
            domain=DomainType(data["domain"]),
            url=data["url"],
            frequency=data.get("frequency", "5min"),
            parser=data.get("parser", ""),
            auth_ref=data.get("auth_ref"),
            vectorize=data.get("vectorize", True),
            active=data.get("active", True),
            metadata=data.get("metadata", {})
        )


async def validate_rss_feed(url: str, timeout: int = 10) -> bool:
    """Validate an RSS feed without importing heavy components."""
    try:
        import aiohttp
        import feedparser
        
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"❌ HTTP {response.status}: {url}")
                    return False
                
                content = await response.text()
                
                # Parse with feedparser
                feed = feedparser.parse(content)
                
                if feed.bozo:
                    print(f"❌ Invalid RSS/XML: {url}")
                    return False
                
                if not feed.entries:
                    print(f"⚠️  No entries found: {url}")
                    return False
                
                print(f"✅ Valid RSS feed: {len(feed.entries)} entries")
                print(f"   Latest: {feed.entries[0].title[:80]}...")
                return True
                
    except asyncio.TimeoutError:
        print(f"❌ Timeout after {timeout}s: {url}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def add_source(name: str, url: str, source_type: str, domain: str, description: str = ""):
    """Add a knowledge source with validation."""
    print(f"Validating source: {name}")
    
    # Validate if RSS
    if source_type.lower() == "rss":
        validation_result = asyncio.run(validate_rss_feed(url))
        if not validation_result:
            print("Source not added - validation failed")
            return False
    else:
        print(f"⚠️  Validation not implemented for {source_type}")
        return False
    
    # Load existing sources
    sources = []
    if os.path.exists("universal_sources.json"):
        with open("universal_sources.json", "r") as f:
            sources_data = json.load(f)
            for item in sources_data:
                sources.append(KnowledgeSource.from_dict(item))
    
    # Create new source
    source_id = f"source_{name.lower().replace(' ', '_')}"
    new_source = KnowledgeSource(
        id=source_id,
        name=name,
        type=SourceType(source_type.lower()),
        domain=DomainType(domain.lower()),
        url=url,
        frequency="5min",
        parser=source_type.lower(),
        active=True,
        metadata={"description": description}
    )
    
    # Add to list
    sources.append(new_source)
    
    # Save back to file
    data = [source.to_dict() for source in sources]
    with open("universal_sources.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Added source: {name}")
    print(f"   ID: {source_id}")
    print(f"   Type: {source_type}")
    print(f"   Domain: {domain}")
    print(f"   URL: {url}")
    
    return True


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 add_source_light.py <name> <url> <type> <domain> [description]")
        sys.exit(1)
    
    name = sys.argv[1]
    url = sys.argv[2]
    source_type = sys.argv[3]
    domain = sys.argv[4]
    description = sys.argv[5] if len(sys.argv) > 5 else ""
    
    success = add_source(name, url, source_type, domain, description)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 