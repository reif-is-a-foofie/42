#!/usr/bin/env python3
"""Lightweight RSS validation script."""

import asyncio
import aiohttp
import feedparser
import sys
from urllib.parse import urlparse


async def validate_rss_feed(url: str, timeout: int = 10) -> bool:
    """Validate an RSS feed without importing heavy components."""
    try:
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 validate_rss.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    result = asyncio.run(validate_rss_feed(url))
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main() 