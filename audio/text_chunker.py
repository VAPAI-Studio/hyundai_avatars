"""
Module for chunking text into meaningful segments for streaming TTS.
"""

import re
import logging
from typing import List, Generator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_index: int
    end_index: int
    chunk_type: str  # "sentence", "phrase", "pause", "semantic"
    confidence: float = 1.0

class TextChunker:
    def __init__(self, chunk_strategy: str = "semantic"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_strategy: Strategy for chunking ("semantic", "sentence", "phrase", "pause")
        """
        self.chunk_strategy = chunk_strategy
        
        # Sentence ending patterns
        self.sentence_endings = r'[.!?]+'
        
        # Natural pause patterns
        self.pause_patterns = r'[,;:]'
        
        # Semantic break patterns (conjunctions, transitions)
        self.semantic_breaks = r'\b(pero|sin embargo|además|por otro lado|en primer lugar|en segundo lugar|finalmente|por último|por tanto|por consiguiente|es decir|o sea|específicamente|en particular|por ejemplo|como|cuando|donde|quien|que|cuyo|cual|si|aunque|mientras|después|antes|durante|desde|hasta|entre|contra|hacia|según|mediante|sin|con|por|para|de|a|en|sobre|bajo|tras|ante|bajo|cabe|con|contra|de|desde|durante|en|entre|hacia|hasta|mediante|para|por|según|sin|sobre|tras|durante|mientras|después|antes|cuando|donde|como|quien|que|cuyo|cual|si|aunque|pero|sin embargo|además|por otro lado|en primer lugar|en segundo lugar|finalmente|por último|por tanto|por consiguiente|es decir|o sea|específicamente|en particular|por ejemplo)\b'
        
        # Minimum chunk size (in characters)
        self.min_chunk_size = 20
        
        # Maximum chunk size (in characters)
        self.max_chunk_size = 200
        
    def chunk_text(self, text: str) -> Generator[TextChunk, None, None]:
        """
        Chunk text based on the selected strategy.
        
        Args:
            text: The text to chunk
            
        Yields:
            TextChunk objects
        """
        if self.chunk_strategy == "semantic":
            yield from self._chunk_semantic(text)
        elif self.chunk_strategy == "sentence":
            yield from self._chunk_sentences(text)
        elif self.chunk_strategy == "phrase":
            yield from self._chunk_phrases(text)
        elif self.chunk_strategy == "pause":
            yield from self._chunk_pauses(text)
        else:
            # Default to semantic chunking
            yield from self._chunk_semantic(text)
    
    def _chunk_semantic(self, text: str) -> Generator[TextChunk, None, None]:
        """Chunk text based on semantic boundaries."""
        if not text.strip():
            return
            
        # Find all potential break points
        break_points = []
        
        # Add sentence endings
        for match in re.finditer(self.sentence_endings, text):
            break_points.append((match.end(), "sentence", 1.0))
            
        # Add semantic breaks
        for match in re.finditer(self.semantic_breaks, text, re.IGNORECASE):
            # Only consider breaks that are followed by a space or punctuation
            if match.end() < len(text) and text[match.end()] in ' \n\t':
                break_points.append((match.end(), "semantic", 0.8))
                
        # Add pause patterns
        for match in re.finditer(self.pause_patterns, text):
            break_points.append((match.end(), "pause", 0.6))
            
        # Sort break points by position
        break_points.sort(key=lambda x: x[0])
        
        # Create chunks based on break points
        start = 0
        current_chunk = ""
        
        for pos, break_type, confidence in break_points:
            # Add text up to this break point
            current_chunk += text[start:pos]
            
            # Check if chunk is large enough
            if len(current_chunk.strip()) >= self.min_chunk_size:
                # Yield the chunk
                chunk_text = current_chunk.strip()
                if chunk_text:
                    yield TextChunk(
                        text=chunk_text,
                        start_index=start,
                        end_index=pos,
                        chunk_type=break_type,
                        confidence=confidence
                    )
                current_chunk = ""
                start = pos
            elif len(current_chunk.strip()) > self.max_chunk_size:
                # Force a break if chunk is too large
                chunk_text = current_chunk.strip()
                if chunk_text:
                    yield TextChunk(
                        text=chunk_text,
                        start_index=start,
                        end_index=pos,
                        chunk_type="forced",
                        confidence=0.5
                    )
                current_chunk = ""
                start = pos
        
        # Handle remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                yield TextChunk(
                    text=remaining,
                    start_index=start,
                    end_index=len(text),
                    chunk_type="final",
                    confidence=1.0
                )
    
    def _chunk_sentences(self, text: str) -> Generator[TextChunk, None, None]:
        """Chunk text into sentences."""
        if not text.strip():
            return
            
        # Split by sentence endings
        sentences = re.split(f'({self.sentence_endings})', text)
        
        current_sentence = ""
        start_index = 0
        
        for i, part in enumerate(sentences):
            current_sentence += part
            
            # Check if this part ends a sentence
            if re.search(self.sentence_endings, part):
                sentence_text = current_sentence.strip()
                if sentence_text and len(sentence_text) >= self.min_chunk_size:
                    yield TextChunk(
                        text=sentence_text,
                        start_index=start_index,
                        end_index=start_index + len(current_sentence),
                        chunk_type="sentence",
                        confidence=1.0
                    )
                current_sentence = ""
                start_index += len(current_sentence)
        
        # Handle any remaining text
        if current_sentence.strip():
            remaining = current_sentence.strip()
            if len(remaining) >= self.min_chunk_size:
                yield TextChunk(
                    text=remaining,
                    start_index=start_index,
                    end_index=len(text),
                    chunk_type="sentence",
                    confidence=1.0
                )
    
    def _chunk_phrases(self, text: str) -> Generator[TextChunk, None, None]:
        """Chunk text into phrases based on punctuation."""
        if not text.strip():
            return
            
        # Split by common phrase separators
        phrases = re.split(r'([,;:])', text)
        
        current_phrase = ""
        start_index = 0
        
        for i, part in enumerate(phrases):
            current_phrase += part
            
            # Check if this part is a phrase separator
            if re.search(r'[,;:]', part):
                phrase_text = current_phrase.strip()
                if phrase_text and len(phrase_text) >= self.min_chunk_size:
                    yield TextChunk(
                        text=phrase_text,
                        start_index=start_index,
                        end_index=start_index + len(current_phrase),
                        chunk_type="phrase",
                        confidence=0.8
                    )
                current_phrase = ""
                start_index += len(current_phrase)
        
        # Handle any remaining text
        if current_phrase.strip():
            remaining = current_phrase.strip()
            if len(remaining) >= self.min_chunk_size:
                yield TextChunk(
                    text=remaining,
                    start_index=start_index,
                    end_index=len(text),
                    chunk_type="phrase",
                    confidence=0.8
                )
    
    def _chunk_pauses(self, text: str) -> Generator[TextChunk, None, None]:
        """Chunk text based on natural pause patterns."""
        if not text.strip():
            return
            
        # Find all pause points
        pause_points = []
        for match in re.finditer(self.pause_patterns, text):
            pause_points.append(match.end())
        
        if not pause_points:
            # If no pauses found, treat as single chunk
            if len(text.strip()) >= self.min_chunk_size:
                yield TextChunk(
                    text=text.strip(),
                    start_index=0,
                    end_index=len(text),
                    chunk_type="pause",
                    confidence=1.0
                )
            return
        
        # Create chunks based on pause points
        start = 0
        for pause_pos in pause_points:
            chunk_text = text[start:pause_pos].strip()
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                yield TextChunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=pause_pos,
                    chunk_type="pause",
                    confidence=0.7
                )
            start = pause_pos
        
        # Handle remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining and len(remaining) >= self.min_chunk_size:
                yield TextChunk(
                    text=remaining,
                    start_index=start,
                    end_index=len(text),
                    chunk_type="pause",
                    confidence=0.7
                )
    
    def set_chunk_limits(self, min_size: int = 20, max_size: int = 200):
        """Set minimum and maximum chunk sizes."""
        self.min_chunk_size = min_size
        self.max_chunk_size = max_size
    
    def get_chunk_strategy(self) -> str:
        """Get the current chunking strategy."""
        return self.chunk_strategy
    
    def set_chunk_strategy(self, strategy: str):
        """Set the chunking strategy."""
        valid_strategies = ["semantic", "sentence", "phrase", "pause"]
        if strategy in valid_strategies:
            self.chunk_strategy = strategy
        else:
            logger.warning(f"Invalid chunk strategy: {strategy}. Using 'semantic' instead.")
            self.chunk_strategy = "semantic" 