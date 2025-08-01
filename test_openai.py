#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Use the correct import syntax
import importlib.util
spec = importlib.util.spec_from_file_location("moroni", "42/moroni/moroni.py")
moroni_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moroni_module)
Moroni = moroni_module.Moroni

def test_openai():
    try:
        moroni = Moroni()
        print(f"OpenAI available: {moroni.ai_clients['openai']['available']}")
        print(f"Primary provider: {moroni.primary_provider}")
        print(f"Fallback provider: {moroni.fallback_provider}")
        
        # Test a simple call
        response = moroni._call_ai("Hello", "test")
        print(f"Response: {response[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_openai() 