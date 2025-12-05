#!/usr/bin/env python3
"""
API clients for Claude (grading) and OpenAI (embeddings).
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import anthropic
import openai
from sklearn.metrics.pairwise import cosine_similarity


class RateLimiter:
    """Simple rate limiter to enforce API rate limits."""

    def __init__(self, max_requests_per_minute: int = 5):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum number of requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self.requests = deque()

    def wait_if_needed(self):
        """Wait if necessary to stay within rate limit."""
        now = time.time()

        # Remove requests older than the time window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()

        # If we've hit the limit, wait until the oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = self.window_seconds - (now - self.requests[0]) + 0.1  # Add 0.1s buffer
            if sleep_time > 0:
                print(f"  Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                # Clean up old requests after waiting
                now = time.time()
                while self.requests and self.requests[0] < now - self.window_seconds:
                    self.requests.popleft()

        # Record this request
        self.requests.append(time.time())


class ClaudeGrader:
    """Claude 4.5 Sonnet grader for preference labeling and response ranking."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: int = 2,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize Claude grader.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum number of retries on API errors
            retry_delay: Delay between retries (seconds)
            rate_limiter: Optional rate limiter to enforce API limits
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = rate_limiter

    def grade_preference_pair(
        self,
        prompt: str,
        response_1: str,
        response_2: str
    ) -> Tuple[str, str, str]:
        """
        Grade a preference pair and return which response is more persuasive.

        Args:
            prompt: Conversation context/prompt
            response_1: First response
            response_2: Second response

        Returns:
            Tuple of (chosen_response, rejected_response, reasoning)
        """
        grading_prompt = f"""You are an expert evaluator of persuasive writing.

You will see a conversation context and two candidate replies (Response 1 and Response 2).

Decide which reply is MORE PERSUASIVE, considering:
- Clarity and coherence
- Respectful, non-manipulative tone
- Directly addressing the other person's concerns or position
- Reasonable use of evidence or logical arguments
- Avoiding manipulation, false claims, or emotional exploitation

Answer with exactly one line in this format:
PREFERRED: 1
or
PREFERRED: 2

If both responses are equally persuasive or equally poor, you may respond:
PREFERRED: EQUAL

Then provide a brief explanation (2-3 sentences) of your decision.

[CONTEXT]
{prompt}

[RESPONSE 1]
{response_1}

[RESPONSE 2]
{response_2}"""

        # Call Claude API with retries
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": grading_prompt}]
                )

                response_text = message.content[0].text

                # Parse preference
                preference = self._parse_preference(response_text)

                # Extract reasoning (everything after the PREFERRED line)
                lines = response_text.strip().split('\n')
                reasoning = '\n'.join(lines[1:]).strip()

                # Determine chosen and rejected
                if preference == "1":
                    return response_1, response_2, reasoning
                elif preference == "2":
                    return response_2, response_1, reasoning
                elif preference == "EQUAL":
                    # Randomly break tie
                    import random
                    if random.random() < 0.5:
                        return response_1, response_2, reasoning + " [EQUAL - randomly assigned]"
                    else:
                        return response_2, response_1, reasoning + " [EQUAL - randomly assigned]"
                else:
                    raise ValueError(f"Invalid preference: {preference}")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    raise

    def rank_responses(
        self,
        context: str,
        responses: Dict[str, str]
    ) -> Tuple[List[str], str]:
        """
        Rank 4 responses (A, B, C, D) by persuasiveness.

        Args:
            context: Conversation context/prompt
            responses: Dict mapping labels (A/B/C/D) to response text

        Returns:
            Tuple of (ranking list, full response with explanation)
        """
        ranking_prompt = f"""You will see a conversation context and four candidate replies (A, B, C, D).

Rank the replies from MOST PERSUASIVE to LEAST PERSUASIVE, considering:
- Clarity and coherence
- Respectful, non-manipulative tone
- Direct engagement with the other person's concerns or position
- Reasonable use of evidence or logical arguments
- Avoiding manipulation, false claims, or emotional exploitation

Answer with exactly one line in this format:
RANKING: A > C > B > D

Then provide a brief explanation (2-3 sentences) of your ranking.

[CONTEXT]
{context}

[CANDIDATE A]
{responses['A']}

[CANDIDATE B]
{responses['B']}

[CANDIDATE C]
{responses['C']}

[CANDIDATE D]
{responses['D']}"""

        # Call Claude API with retries
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": ranking_prompt}]
                )

                response_text = message.content[0].text

                # Parse ranking
                ranking = self._parse_ranking(response_text)

                return ranking, response_text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    raise

    def assign_scores(self, ranking: List[str]) -> Dict[str, int]:
        """
        Convert ranking to scores (1st=3, 2nd=2, 3rd=1, 4th=0).

        Args:
            ranking: List of labels in order (e.g., ['A', 'C', 'B', 'D'])

        Returns:
            Dict mapping labels to scores
        """
        return {
            ranking[0]: 3,
            ranking[1]: 2,
            ranking[2]: 1,
            ranking[3]: 0
        }

    def _parse_preference(self, response_text: str) -> str:
        """Parse preference from response text."""
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('PREFERRED:'):
                preference = line.replace('PREFERRED:', '').strip()
                if preference in ["1", "2", "EQUAL"]:
                    return preference
        raise ValueError(f"Could not parse preference from: {response_text}")

    def _parse_ranking(self, response_text: str) -> List[str]:
        """Parse ranking from response text."""
        import re
        
        # Method 1: Look for explicit "RANKING:" line
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('RANKING:'):
                ranking_str = line.replace('RANKING:', '').strip()
                # Extract order: "A > C > B > D" → ['A', 'C', 'B', 'D']
                ranking = [label.strip() for label in ranking_str.split('>')]
                if len(ranking) == 4 and all(label in ['A', 'B', 'C', 'D'] for label in ranking):
                    return ranking
        
        # Method 2: Look for pattern "A > B > C > D" anywhere in response
        pattern = r'\b([ABCD])\s*>\s*([ABCD])\s*>\s*([ABCD])\s*>\s*([ABCD])\b'
        match = re.search(pattern, response_text)
        if match:
            ranking = list(match.groups())
            if len(set(ranking)) == 4:  # All unique
                return ranking
        
        # Method 3: Look for numbered list like "1. A" or "1) A"
        numbered_pattern = r'(?:^|\n)\s*[1-4][.)]\s*([ABCD])\b'
        matches = re.findall(numbered_pattern, response_text)
        if len(matches) == 4 and len(set(matches)) == 4:
            return matches
        
        raise ValueError(f"Could not parse ranking from: {response_text}")


class OpenAIEmbedder:
    """OpenAI embeddings for similarity scoring."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            dimensions: Embedding dimensions
            max_retries: Maximum number of retries on API errors
            retry_delay: Delay between retries (seconds)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimensions
                )
                return np.array(response.data[0].embedding)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    raise

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts in a batch.

        Args:
            texts: List of input texts

        Returns:
            2D numpy array of embeddings (shape: [len(texts), dimensions])
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions
                )
                embeddings = [np.array(item.embedding) for item in response.data]
                return np.array(embeddings)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    raise

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def compute_similarities_batch(
        self,
        texts: List[str],
        reference: str
    ) -> List[float]:
        """
        Compute cosine similarities between multiple texts and a reference.

        Args:
            texts: List of texts to compare
            reference: Reference text

        Returns:
            List of similarity scores
        """
        # Get all embeddings in batch (more efficient)
        all_texts = texts + [reference]
        embeddings = self.get_embeddings_batch(all_texts)

        # Separate reference embedding
        ref_embedding = embeddings[-1].reshape(1, -1)
        text_embeddings = embeddings[:-1]

        # Compute similarities
        similarities = cosine_similarity(text_embeddings, ref_embedding)

        return [float(sim[0]) for sim in similarities]


def load_api_config(config_path: str = "configs/api_config.yaml") -> Dict:
    """
    Load API configuration from YAML file.

    Args:
        config_path: Path to API config file

    Returns:
        Config dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate API keys are set
    if not config['anthropic']['api_key']:
        raise ValueError("Claude API key not set in configs/api_config.yaml")
    if not config['openai']['api_key']:
        raise ValueError("OpenAI API key not set in configs/api_config.yaml")

    return config


def create_claude_grader(config_path: str = "configs/api_config.yaml") -> ClaudeGrader:
    """Create ClaudeGrader from config file with rate limiting."""
    config = load_api_config(config_path)

    # Create rate limiter from config
    rate_limit = config.get('rate_limiting', {}).get('requests_per_minute', 5)
    rate_limiter = RateLimiter(max_requests_per_minute=rate_limit)

    return ClaudeGrader(
        api_key=config['anthropic']['api_key'],
        model=config['anthropic']['model'],
        max_tokens=config['anthropic']['max_tokens'],
        temperature=config['anthropic']['temperature'],
        rate_limiter=rate_limiter
    )


def create_openai_embedder(config_path: str = "configs/api_config.yaml") -> OpenAIEmbedder:
    """Create OpenAIEmbedder from config file."""
    config = load_api_config(config_path)
    return OpenAIEmbedder(
        api_key=config['openai']['api_key'],
        model=config['openai']['embedding_model'],
        dimensions=config['openai']['embedding_dimensions']
    )
