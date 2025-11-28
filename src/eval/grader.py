#!/usr/bin/env python3
"""
Grader module for evaluation.

This module provides convenient access to Claude grader and OpenAI embedder
for evaluation tasks.
"""

# Re-export from api_clients for convenience
from src.utils.api_clients import (
    ClaudeGrader,
    OpenAIEmbedder,
    create_claude_grader,
    create_openai_embedder
)

__all__ = [
    'ClaudeGrader',
    'OpenAIEmbedder',
    'create_claude_grader',
    'create_openai_embedder'
]
