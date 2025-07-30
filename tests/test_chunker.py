"""Tests for the chunker module."""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the 42 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "42"))

from chunker import Chunker


class TestChunker:
    """Test chunker functionality."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance for testing."""
        return Chunker()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_chunker_initialization(self, chunker):
        """Test that chunker can be initialized."""
        assert chunker is not None
        assert hasattr(chunker, 'supported_extensions')
        assert isinstance(chunker.supported_extensions, set)
        assert len(chunker.supported_extensions) > 0
    
    def test_chunk_python_file(self, chunker, temp_dir):
        """Test chunking a Python file."""
        python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b

# Main execution
if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")
'''
        
        python_file = Path(temp_dir) / "test.py"
        python_file.write_text(python_code)
        
        chunks = chunker.chunk_file(str(python_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Should have chunks for function and class
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("def hello_world" in text for text in chunk_texts)
        assert any("class Calculator" in text for text in chunk_texts)
        
        # Check metadata
        for chunk in chunks:
            assert "file_path" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert chunk["file_path"] == str(python_file)
    
    def test_chunk_markdown_file(self, chunker, temp_dir):
        """Test chunking a Markdown file."""
        markdown_content = '''
# Introduction

This is an introduction to the project.

## Features

### Feature 1
This is the first feature.

### Feature 2
This is the second feature.

## Installation

```bash
pip install package
```

## Usage

Here's how to use the package:

1. Import the module
2. Create an instance
3. Call methods

## Contributing

Please read the contributing guidelines.
'''
        
        md_file = Path(temp_dir) / "README.md"
        md_file.write_text(markdown_content)
        
        chunks = chunker.chunk_file(str(md_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Should have chunks for different sections
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("Introduction" in text for text in chunk_texts)
        assert any("Features" in text for text in chunk_texts)
        assert any("Installation" in text for text in chunk_texts)
    
    def test_chunk_text_file(self, chunker, temp_dir):
        """Test chunking a plain text file."""
        text_content = '''
This is a plain text file.

It contains multiple paragraphs.

Each paragraph should be a separate chunk.

This is the final paragraph.
'''
        
        text_file = Path(temp_dir) / "document.txt"
        text_file.write_text(text_content)
        
        chunks = chunker.chunk_file(str(text_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Should have chunks for different paragraphs
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("plain text file" in text for text in chunk_texts)
        assert any("multiple paragraphs" in text for text in chunk_texts)
    
    def test_chunk_unsupported_file(self, chunker, temp_dir):
        """Test chunking an unsupported file type."""
        # Create a binary file
        binary_file = Path(temp_dir) / "image.jpg"
        binary_file.write_bytes(b'\xff\xd8\xff\xe0')  # JPEG header
        
        chunks = chunker.chunk_file(str(binary_file))
        
        # Should return empty list for unsupported files
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    def test_chunk_empty_file(self, chunker, temp_dir):
        """Test chunking an empty file."""
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.write_text("")
        
        chunks = chunker.chunk_file(str(empty_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    def test_chunk_large_file(self, chunker, temp_dir):
        """Test chunking a large file."""
        # Create a large text file
        large_content = "This is a large file. " * 10000
        large_file = Path(temp_dir) / "large.txt"
        large_file.write_text(large_content)
        
        chunks = chunker.chunk_file(str(large_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that chunks are reasonable size
        for chunk in chunks:
            assert len(chunk["text"]) <= 1000  # Reasonable chunk size
    
    def test_chunk_file_with_metadata(self, chunker, temp_dir):
        """Test that chunks include proper metadata."""
        content = '''
def test_function():
    """Test function."""
    return "test"

class TestClass:
    """Test class."""
    pass
'''
        
        test_file = Path(temp_dir) / "test_metadata.py"
        test_file.write_text(content)
        
        chunks = chunker.chunk_file(str(test_file))
        
        for chunk in chunks:
            assert "file_path" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert "text" in chunk
            assert chunk["file_path"] == str(test_file)
            assert isinstance(chunk["start_line"], int)
            assert isinstance(chunk["end_line"], int)
            assert chunk["start_line"] <= chunk["end_line"]
    
    def test_chunk_file_not_found(self, chunker):
        """Test chunking a non-existent file."""
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file("nonexistent_file.txt")
    
    def test_chunk_file_permission_error(self, chunker, temp_dir):
        """Test chunking a file with permission issues."""
        # Create a file and remove read permissions
        test_file = Path(temp_dir) / "permission_test.txt"
        test_file.write_text("Test content")
        test_file.chmod(0o000)  # No permissions
        
        try:
            chunks = chunker.chunk_file(str(test_file))
            # Should handle permission error gracefully
            assert isinstance(chunks, list)
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)
    
    def test_chunk_file_with_special_characters(self, chunker, temp_dir):
        """Test chunking a file with special characters in content."""
        special_content = '''
def special_chars():
    """Function with special characters: éñüß"""
    return "Special: éñüß"
    
# Comments with special chars: éñüß
'''
        
        special_file = Path(temp_dir) / "special_chars.py"
        special_file.write_text(special_content)
        
        chunks = chunker.chunk_file(str(special_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that special characters are preserved
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("éñüß" in text for text in chunk_texts)
    
    def test_chunk_file_with_encoding(self, chunker, temp_dir):
        """Test chunking a file with different encoding."""
        # Test UTF-8 encoding
        utf8_content = "Hello, 世界! 🌍"
        utf8_file = Path(temp_dir) / "utf8.txt"
        utf8_file.write_text(utf8_content, encoding='utf-8')
        
        chunks = chunker.chunk_file(str(utf8_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that UTF-8 content is preserved
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("世界" in text for text in chunk_texts)
    
    def test_chunk_file_size_limit(self, chunker, temp_dir):
        """Test chunking with size limits."""
        # Create a file that exceeds size limit
        large_content = "Large content. " * 100000
        large_file = Path(temp_dir) / "too_large.txt"
        large_file.write_text(large_content)
        
        # Test with size limit
        chunks = chunker.chunk_file(str(large_file), max_file_size=1024)
        
        assert isinstance(chunks, list)
        # Should either skip the file or return limited chunks
        assert len(chunks) == 0 or len(chunks) < 10
    
    def test_chunk_file_with_line_numbers(self, chunker, temp_dir):
        """Test that line numbers are accurate."""
        content = '''
# Line 1
def function1():
    # Line 3
    return 1

# Line 6
def function2():
    # Line 8
    return 2
'''
        
        test_file = Path(temp_dir) / "line_numbers.py"
        test_file.write_text(content)
        
        chunks = chunker.chunk_file(str(test_file))
        
        for chunk in chunks:
            assert chunk["start_line"] >= 1
            assert chunk["end_line"] >= chunk["start_line"]
            
            # Check that line numbers correspond to content
            lines = content.split('\n')
            chunk_lines = lines[chunk["start_line"]-1:chunk["end_line"]]
            chunk_content = '\n'.join(chunk_lines)
            assert chunk_content.strip() in chunk["text"]
    
    def test_chunk_file_with_comments(self, chunker, temp_dir):
        """Test chunking files with comments."""
        python_with_comments = '''
# This is a comment
def main():
    """Docstring comment."""
    # Inline comment
    x = 1  # End of line comment
    return x

# Another comment
class Test:
    """Class docstring."""
    pass
'''
        
        comment_file = Path(temp_dir) / "comments.py"
        comment_file.write_text(python_with_comments)
        
        chunks = chunker.chunk_file(str(comment_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that comments are preserved
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("This is a comment" in text for text in chunk_texts)
        assert any("Inline comment" in text for text in chunk_texts)
    
    def test_chunk_file_with_imports(self, chunker, temp_dir):
        """Test chunking files with imports."""
        python_with_imports = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def main():
    """Main function."""
    return "Hello"

if __name__ == "__main__":
    main()
'''
        
        import_file = Path(temp_dir) / "imports.py"
        import_file.write_text(python_with_imports)
        
        chunks = chunker.chunk_file(str(import_file))
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check that imports are preserved
        chunk_texts = [chunk["text"] for chunk in chunks]
        assert any("import os" in text for text in chunk_texts)
        assert any("from pathlib import Path" in text for text in chunk_texts) 