"""
GitHub Fetcher for 42.un Knowledge Engine

Monitors GitHub repositories for changes, releases, and updates
relevant to humanitarian response and crisis management.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any
import aiohttp
from loguru import logger

from .knowledge_engine import KnowledgeFetcher, KnowledgeSource, KnowledgeDocument, DomainType


class GitHubFetcher(KnowledgeFetcher):
    """Fetch from GitHub repositories."""
    
    def __init__(self, session: aiohttp.ClientSession):
        super().__init__(session)
        self.api_base = "https://api.github.com"
    
    async def fetch(self, source: KnowledgeSource) -> List[KnowledgeDocument]:
        """Fetch GitHub repository information."""
        try:
            # Extract repo owner and name from URL
            repo_path = source.url.replace("https://github.com/", "")
            owner, repo_name = repo_path.split("/")
            
            # Get repository info
            repo_info = await self._get_repo_info(owner, repo_name)
            if not repo_info:
                return []
            
            # Get recent commits
            commits = await self._get_recent_commits(owner, repo_name)
            
            # Get recent releases
            releases = await self._get_recent_releases(owner, repo_name)
            
            # Create documents
            documents = []
            
            # Repository info document
            if repo_info:
                doc = KnowledgeDocument(
                    source_id=source.id,
                    content=f"Repository: {repo_info['name']} - {repo_info.get('description', 'No description')}",
                    metadata={
                        'url': source.url,
                        'stars': repo_info.get('stargazers_count', 0),
                        'forks': repo_info.get('forks_count', 0),
                        'language': repo_info.get('language', 'Unknown'),
                        'last_updated': repo_info.get('updated_at', ''),
                        'domain': source.domain.value,
                        'repo_info': repo_info
                    }
                )
                documents.append(doc)
            
            # Recent commits document
            if commits:
                commit_content = []
                for commit in commits[:5]:  # Last 5 commits
                    commit_content.append(f"Commit: {commit['sha'][:8]} - {commit['commit']['message']}")
                
                doc = KnowledgeDocument(
                    source_id=source.id,
                    content="Recent commits:\n" + "\n".join(commit_content),
                    metadata={
                        'url': source.url,
                        'commit_count': len(commits),
                        'domain': source.domain.value,
                        'commits': commits[:5]
                    }
                )
                documents.append(doc)
            
            # Recent releases document
            if releases:
                release_content = []
                for release in releases[:3]:  # Last 3 releases
                    release_content.append(f"Release: {release['tag_name']} - {release['name']}")
                
                doc = KnowledgeDocument(
                    source_id=source.id,
                    content="Recent releases:\n" + "\n".join(release_content),
                    metadata={
                        'url': source.url,
                        'release_count': len(releases),
                        'domain': source.domain.value,
                        'releases': releases[:3]
                    }
                )
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents from GitHub: {source.name}")
            return documents
                
        except Exception as e:
            logger.error(f"Error fetching GitHub {source.name}: {e}")
            return []
    
    async def _get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information."""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}"
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': '42-un-knowledge-engine'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get repo info: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting repo info: {e}")
            return None
    
    async def _get_recent_commits(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get recent commits."""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/commits"
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': '42-un-knowledge-engine'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get commits: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting commits: {e}")
            return []
    
    async def _get_recent_releases(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get recent releases."""
        try:
            url = f"{self.api_base}/repos/{owner}/{repo}/releases"
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': '42-un-knowledge-engine'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get releases: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting releases: {e}")
            return [] 