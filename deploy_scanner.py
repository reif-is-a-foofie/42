#!/usr/bin/env python3
"""
Autonomous Scanner Production Deployment

Deploy the autonomous scanner with initial seed URLs and production configuration.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add 42 package to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "42"))

from un.redis_bus import RedisBus
from un.knowledge_engine import KnowledgeEngine
from un.autonomous_scanner import AutonomousScanner, CrawlTarget


# Production seed URLs - high-quality starting points
PRODUCTION_SEED_URLS = [
    {
        "url": "https://dynomight.net",
        "priority": 0.9,
        "description": "Science and existential angst blog"
    },
    {
        "url": "https://arxiv.org",
        "priority": 0.8,
        "description": "Research papers and academic content"
    },
    {
        "url": "https://github.com",
        "priority": 0.7,
        "description": "Open source projects and documentation"
    },
    {
        "url": "https://news.ycombinator.com",
        "priority": 0.8,
        "description": "Tech news and discussions"
    },
    {
        "url": "https://reddit.com/r/science",
        "priority": 0.6,
        "description": "Science discussions and discoveries"
    },
    {
        "url": "https://stackoverflow.com",
        "priority": 0.7,
        "description": "Technical knowledge and solutions"
    },
    {
        "url": "https://wikipedia.org",
        "priority": 0.6,
        "description": "General knowledge and references"
    },
    {
        "url": "https://medium.com",
        "priority": 0.5,
        "description": "Articles and insights"
    }
]

# Production configuration
PRODUCTION_CONFIG = {
    "max_depth": 3,
    "max_pages_per_domain": 10,
    "crawl_delay": 2.0,  # Be respectful to servers
    "discovery_interval": 300,  # 5 minutes
    "user_agent": "42-AutonomousScanner/1.0 (https://github.com/reif-is-a-foofie/42)",
    "max_concurrent_crawls": 3,
    "respect_robots_txt": True,
    "rate_limit_per_domain": 1.0  # seconds between requests
}


async def deploy_scanner_production():
    """Deploy the autonomous scanner in production mode."""
    print("🚀 Deploying Autonomous Scanner in Production Mode")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    redis_bus = RedisBus()
    knowledge_engine = KnowledgeEngine(redis_bus)
    scanner = AutonomousScanner(redis_bus, knowledge_engine, PRODUCTION_CONFIG)
    
    # Add seed URLs to pending targets
    print("🌱 Adding seed URLs...")
    for seed in PRODUCTION_SEED_URLS:
        target = CrawlTarget(
            url=seed["url"],
            priority=seed["priority"],
            crawl_depth=2,
            source_type="seed_url",
            discovered_from="production_deployment"
        )
        scanner.pending_targets.append(target)
        print(f"  ✅ {seed['url']} (priority: {seed['priority']})")
    
    print(f"\n📊 Total seed URLs: {len(PRODUCTION_SEED_URLS)}")
    print(f"🔧 Configuration: {json.dumps(PRODUCTION_CONFIG, indent=2)}")
    
    # Start the scanner
    print("\n🚀 Starting autonomous scanner...")
    print("  🔄 Discovery loop will run every 5 minutes")
    print("  🕷️  Crawling with rate limiting and respect for robots.txt")
    print("  🧠 Learning from knowledge base to guide discovery")
    print("  📡 Publishing events to Redis bus")
    print("  💾 Storing discovered sources in knowledge engine")
    
    try:
        await scanner.start()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping scanner...")
        await scanner.stop()
        print("✅ Scanner stopped gracefully")
    except Exception as e:
        print(f"\n❌ Scanner error: {e}")
        await scanner.stop()
        raise


async def add_custom_seed_urls(urls: List[str]):
    """Add custom seed URLs to the scanner."""
    print("🌱 Adding custom seed URLs...")
    
    redis_bus = RedisBus()
    knowledge_engine = KnowledgeEngine(redis_bus)
    scanner = AutonomousScanner(redis_bus, knowledge_engine, PRODUCTION_CONFIG)
    
    for url in urls:
        target = CrawlTarget(
            url=url,
            priority=0.8,  # High priority for custom URLs
            crawl_depth=2,
            source_type="custom_seed",
            discovered_from="user_input"
        )
        scanner.pending_targets.append(target)
        print(f"  ✅ Added: {url}")
    
    print(f"📊 Added {len(urls)} custom seed URLs")


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Autonomous Scanner")
    parser.add_argument("--custom-urls", nargs="+", help="Custom seed URLs to add")
    parser.add_argument("--add-only", action="store_true", help="Only add URLs, don't start scanner")
    
    args = parser.parse_args()
    
    if args.custom_urls:
        asyncio.run(add_custom_seed_urls(args.custom_urls))
        if not args.add_only:
            print("\n🚀 Starting scanner with custom URLs...")
            asyncio.run(deploy_scanner_production())
    else:
        asyncio.run(deploy_scanner_production())


if __name__ == "__main__":
    main() 