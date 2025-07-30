"""Chunker for splitting files into chunks."""

import ast
import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

from .interfaces import Chunk


class Chunker:
    """Splits files into chunks for embedding."""
    
    def __init__(self):
        """Initialize the chunker."""
        self.supported_extensions = {
            '.py', '.md', '.txt', '.js', '.ts', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.html', '.css', '.scss', '.sh',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.sql', '.r', '.m', '.scala', '.kt', '.swift', '.dart'
        }
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Split a file into chunks."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return []
            
            if path.suffix not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {path.suffix}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if path.suffix == '.py':
                return self._chunk_python(content, file_path)
            elif path.suffix == '.md':
                return self._chunk_markdown(content, file_path)
            else:
                return self._chunk_generic(content, file_path)
                
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            return []
    
    def _chunk_python(self, content: str, file_path: str) -> List[Chunk]:
        """Split Python file by functions and classes."""
        chunks = []
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Get the source lines for this node
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    # Extract the text for this node
                    lines = content.split('\n')
                    node_lines = lines[start_line - 1:end_line]
                    node_text = '\n'.join(node_lines)
                    
                    chunks.append(Chunk(
                        text=node_text,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={'type': type(node).__name__, 'name': node.name}
                    ))
            
            # If no functions/classes found, create a single chunk
            if not chunks:
                chunks.append(Chunk(
                    text=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(content.split('\n')),
                    metadata={'type': 'module'}
                ))
                
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fall back to generic chunking
            return self._chunk_generic(content, file_path)
        
        return chunks
    
    def _chunk_markdown(self, content: str, file_path: str) -> List[Chunk]:
        """Split Markdown file by headers."""
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_start = 1
        
        for i, line in enumerate(lines, 1):
            if line.startswith('#'):
                # Save previous chunk if it exists
                if current_chunk_lines:
                    chunks.append(Chunk(
                        text='\n'.join(current_chunk_lines),
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i - 1,
                        metadata={'type': 'markdown_section'}
                    ))
                
                # Start new chunk
                current_chunk_lines = [line]
                current_start = i
            else:
                current_chunk_lines.append(line)
        
        # Add final chunk
        if current_chunk_lines:
            chunks.append(Chunk(
                text='\n'.join(current_chunk_lines),
                file_path=file_path,
                start_line=current_start,
                end_line=len(lines),
                metadata={'type': 'markdown_section'}
            ))
        
        return chunks
    
    def _chunk_generic(self, content: str, file_path: str) -> List[Chunk]:
        """Generic chunking for other file types."""
        lines = content.split('\n')
        chunk_size = 50  # Lines per chunk
        
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunks.append(Chunk(
                text='\n'.join(chunk_lines),
                file_path=file_path,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                metadata={'type': 'generic'}
            ))
        
        return chunks
    
    def chunk_directory(self, directory_path: str) -> List[Chunk]:
        """Chunk all supported files in a directory recursively."""
        chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory_path}")
            return chunks
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                file_chunks = self.chunk_file(str(file_path))
                chunks.extend(file_chunks)
                logger.info(f"Chunked {file_path}: {len(file_chunks)} chunks")
        
        return chunks 