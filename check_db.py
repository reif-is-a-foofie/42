#!/usr/bin/env python3
"""Check vector database contents."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 42.vector_store import VectorStore

def main():
    try:
        vs = VectorStore('42_knowledge')
        total_points = vs.get_total_points()
        print(f"Total points in 42_knowledge collection: {total_points}")
        
        if total_points > 0:
            # Get some sample points
            points = vs.get_all_vectors()
            print(f"Sample points: {len(points)}")
            for i, point in enumerate(points[:3]):
                print(f"Point {i}: {point.get('text', '')[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 