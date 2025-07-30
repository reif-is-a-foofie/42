#!/usr/bin/env python3
"""
Add Knowledge Source

Simple script to add any knowledge source to the universal intelligence database.
"""

import json
import sys
import os

# Add 42 package to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "42"))

from un.knowledge_engine import KnowledgeSource, SourceType, DomainType


def add_source(name: str, url: str, source_type: str, domain: str, description: str = ""):
    """Add a new knowledge source to the database."""
    
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
    
    print(f"✅ Added source: {name}")
    print(f"   ID: {source_id}")
    print(f"   Type: {source_type}")
    print(f"   Domain: {domain}")
    print(f"   URL: {url}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add Knowledge Source")
    parser.add_argument("name", help="Source name")
    parser.add_argument("url", help="Source URL")
    parser.add_argument("type", help="Source type (rss, api, github, etc.)")
    parser.add_argument("domain", help="Domain (weather, medical, finance, research, etc.)")
    parser.add_argument("--description", default="", help="Description")
    
    args = parser.parse_args()
    
    add_source(args.name, args.url, args.type, args.domain, args.description) 