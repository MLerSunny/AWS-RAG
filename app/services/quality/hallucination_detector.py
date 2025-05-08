"""
Hallucination detection and factual consistency checking.
"""
import re
from typing import Dict, List, Tuple, Optional, Set, cast, Any
from functools import lru_cache
from collections import Counter
import math
from ...utils.logger import setup_logger

# Conditional import for nltk
try:
    import nltk
    have_nltk = True

    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger = setup_logger(__name__)
            logger.warning(f"Failed to download punkt tokenizer: {str(e)}")

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger = setup_logger(__name__)
            logger.warning(f"Failed to download stopwords: {str(e)}")
        
    try:
        STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    except Exception as e:
        logger = setup_logger(__name__)
        logger.warning(f"Failed to load stopwords: {str(e)}")
        STOPWORDS = set()
except ImportError:
    have_nltk = False
    nltk = None
    STOPWORDS = set()

# Optional numpy import for vectorization
try:
    import numpy as np
    have_numpy = True
except ImportError:
    have_numpy = False
    np = None

logger = setup_logger(__name__)

class HallucinationDetector:
    """Detector for hallucinations in model responses."""
    
    def __init__(self, similarity_threshold: float = 0.5, use_vectorization: bool = True):
        """
        Initialize the hallucination detector.
        
        Args:
            similarity_threshold (float): Threshold for determining fact support
            use_vectorization (bool): Whether to use vectorized similarity calculation
        """
        if not have_nltk:
            logger.warning("NLTK is not installed. Please install with 'pip install nltk'")
        
        if use_vectorization and not have_numpy:
            logger.warning("NumPy is not installed but vectorization was requested. "
                          "Falling back to token overlap. Install with 'pip install numpy'")
            use_vectorization = False
            
        self.similarity_threshold = similarity_threshold
        self.use_vectorization = use_vectorization and have_numpy
        logger.info(f"Initialized hallucination detector with threshold {similarity_threshold}, "
                   f"vectorization: {self.use_vectorization}")
    
    def check_factual_consistency(self, answer: str, sources: List[str]) -> Dict[str, Any]:
        """
        Check if the answer is consistent with the provided sources.
        
        Args:
            answer (str): The model's answer
            sources (List[str]): List of source texts to check against
            
        Returns:
            Dict: Analysis of factual consistency
        """
        # Handle empty inputs gracefully
        if not answer or not sources:
            return {
                "claims": [],
                "confidence": 1.0,
                "is_hallucination": False,
                "explanation": "No answer or sources provided for analysis."
            }
            
        # Check if NLTK is available
        if not have_nltk:
            return {
                "claims": [],
                "confidence": 0.0,
                "is_hallucination": False,
                "explanation": "NLTK is not installed. Cannot perform hallucination detection."
            }
            
        # Extract claims from the answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return {
                "claims": [],
                "confidence": 1.0,
                "is_hallucination": False,
                "explanation": "No factual claims detected in the answer."
            }
        
        # Check each claim against sources
        results = []
        for claim in claims:
            source_support = self._find_support_in_sources(claim, sources)
            results.append({
                "claim": claim,
                "supported": source_support[0],
                "source": source_support[1],
                "confidence": source_support[2]
            })
            
        # Calculate overall confidence
        supported_claims = sum(1 for r in results if r["supported"])
        if len(results) == 0:
            confidence = 1.0
        else:
            confidence = supported_claims / len(results)
        
        # Determine if it's a hallucination
        is_hallucination = confidence < 0.5
        
        # Generate explanation
        if is_hallucination:
            unsupported_claims = [r["claim"] for r in results if not r["supported"]]
            explanation = f"The response contains {len(unsupported_claims)} unsupported claims that couldn't be verified in the source material."
        else:
            explanation = f"{supported_claims} out of {len(results)} claims are supported by the sources."
        
        return {
            "claims": results,
            "confidence": confidence,
            "is_hallucination": is_hallucination,
            "explanation": explanation
        }
    
    @lru_cache(maxsize=128)
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[str]: Extracted factual claims
        """
        if not have_nltk or nltk is None:
            return []
        
        # Defensive check for text
        if not isinstance(text, str) or not text.strip():
            return []
            
        # Safe way to call sent_tokenize
        try:
            sentences = nltk.sent_tokenize(text)
            
            # Filter sentences that are likely to be factual claims
            claims = [s for s in sentences if self._is_factual_claim(s)]
            
            logger.info(f"Extracted {len(claims)} claims from text")
            return claims
        except Exception as e:
            logger.error(f"Error extracting claims from text: {str(e)}")
            return []
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence is likely a factual claim.
        
        Args:
            sentence (str): The sentence to analyze
            
        Returns:
            bool: Whether the sentence is likely a factual claim
        """
        # Defensive check
        if not isinstance(sentence, str) or not sentence.strip():
            return False
            
        # Skip questions, short sentences, or non-statements
        if sentence.endswith("?"):
            return False
            
        if len(sentence.split()) < 4:
            return False
        
        # Skip sentences with first-person opinions
        opinion_patterns = [
            r"\bI think\b", r"\bI believe\b", r"\bIn my opinion\b",
            r"\bI feel\b", r"\bI would\b", r"\bI'd\b", r"\bI guess\b"
        ]
        try:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in opinion_patterns):
                return False
        except Exception as e:
            logger.warning(f"Error in regex search: {str(e)}")
            return False
        
        # Check for indicators of factual claims
        try:
            has_number = bool(re.search(r'\d', sentence))
            has_entity = bool(re.search(r'[A-Z][a-z]+', sentence))
            has_date = bool(re.search(r'\b(in|on|during|after|before)\s+\d{4}\b', sentence))
            
            # Check for factual language patterns
            factual_patterns = [
                r"\bis\b", r"\bwas\b", r"\bwere\b", r"\bhave\b", r"\bhas\b",
                r"\bconsists of\b", r"\bcomprises\b", r"\bcontains\b"
            ]
            has_factual_pattern = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_patterns)
            
            # Sentence is a claim if it has any of these indicators
            return has_number or has_date or (has_entity and has_factual_pattern)
        except Exception as e:
            logger.warning(f"Error checking factual claim: {str(e)}")
            return False
    
    def _tokenize_and_clean(self, text: str) -> Set[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text (str): The text to tokenize
            
        Returns:
            Set[str]: Set of unique tokens
        """
        if not have_nltk:
            return set()
            
        # Handle empty or non-string input
        if not isinstance(text, str) or not text.strip():
            return set()
            
        try:
            # Convert to lowercase, tokenize, and remove stopwords and punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
            return {t for t in tokens if t not in STOPWORDS and len(t) > 2}
        except Exception as e:
            logger.warning(f"Error in tokenization: {str(e)}")
            return set()
    
    def _get_term_frequency(self, tokens: Set[str], text: str) -> Dict[str, float]:
        """
        Calculate term frequency for tokens in a text.
        
        Args:
            tokens (Set[str]): Set of tokens to consider
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Term frequency for each token
        """
        try:
            # Find all words in lowercase
            all_words = re.findall(r'\b\w+\b', text.lower())
            
            # Count occurrences
            word_count = Counter(all_words)
            total_words = len(all_words)
            
            # Calculate term frequency
            if total_words == 0:
                return {token: 0.0 for token in tokens}
                
            return {token: word_count.get(token, 0) / total_words for token in tokens}
        except Exception as e:
            logger.warning(f"Error in term frequency calculation: {str(e)}")
            return {token: 0.0 for token in tokens}
    
    def _compute_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two term frequency vectors.
        
        Args:
            vec1 (Dict[str, float]): First term frequency vector
            vec2 (Dict[str, float]): Second term frequency vector
            
        Returns:
            float: Cosine similarity score
        """
        # Use numpy for vectorized calculation if available
        if self.use_vectorization and have_numpy and np is not None:
            try:
                # Get all unique terms
                all_terms = set(vec1.keys()) | set(vec2.keys())
                
                # Create vectors
                v1 = np.array([vec1.get(term, 0.0) for term in all_terms])
                v2 = np.array([vec2.get(term, 0.0) for term in all_terms])
                
                # Calculate cosine similarity
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                return float(np.dot(v1, v2) / (norm1 * norm2))
            except Exception as e:
                logger.warning(f"Error in numpy cosine calculation: {str(e)}")
                # Fall back to non-vectorized calculation
        
        # Manual calculation as fallback
        try:
            # Find common terms
            common_terms = set(vec1.keys()) & set(vec2.keys())
            
            if not common_terms:
                return 0.0
                
            # Calculate numerator (dot product)
            dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
            
            # Calculate denominators (magnitudes)
            magnitude1 = math.sqrt(sum(vec1[term] ** 2 for term in vec1))
            magnitude2 = math.sqrt(sum(vec2[term] ** 2 for term in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
                
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.warning(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0
    
    def _find_support_in_sources(self, claim: str, sources: List[str]) -> Tuple[bool, str, float]:
        """
        Find support for a claim in the provided sources using vectorized similarity
        calculation when available, falling back to token overlap otherwise.
        
        Args:
            claim (str): The claim to check
            sources (List[str]): Source texts to check against
            
        Returns:
            Tuple[bool, str, float]: (is_supported, supporting_source, confidence)
        """
        if not have_nltk:
            return (False, "", 0.0)
            
        # Defensive check
        if not isinstance(claim, str) or not claim.strip() or not sources:
            return (False, "", 0.0)
            
        claim_tokens = self._tokenize_and_clean(claim)
        
        if not claim_tokens:
            return (False, "", 0.0)
        
        # Use vectorized similarity if enabled
        if self.use_vectorization:
            # Calculate TF vector for claim
            claim_tf = self._get_term_frequency(claim_tokens, claim)
            
            best_match_score = 0.0
            best_match_source = ""
            
            for source in sources:
                if not isinstance(source, str) or not source.strip():
                    continue
                    
                source_tokens = self._tokenize_and_clean(source)
                
                if not source_tokens:
                    continue
                
                # Calculate TF vector for source
                source_tf = self._get_term_frequency(source_tokens, source)
                
                # Calculate cosine similarity
                similarity = self._compute_cosine_similarity(claim_tf, source_tf)
                
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_source = source
            
            is_supported = best_match_score >= self.similarity_threshold
            return (is_supported, best_match_source, best_match_score)
        
        # Fall back to token overlap method
        best_match_score = 0.0
        best_match_source = ""
        
        # Calculate TF-IDF style weights
        claim_token_count = len(claim_tokens)
        sources_with_scores = []
        
        for source in sources:
            if not isinstance(source, str) or not source.strip():
                continue
                
            source_tokens = self._tokenize_and_clean(source)
            
            if not source_tokens:
                continue
            
            # Calculate overlap (simple but effective)
            overlap_tokens = claim_tokens.intersection(source_tokens)
            overlap_count = len(overlap_tokens)
            
            if overlap_count == 0:
                continue
                
            # Calculate recall (how much of the claim is covered)
            recall = overlap_count / claim_token_count
            
            # Calculate precision (how specific the match is)
            precision = overlap_count / len(source_tokens)
            
            # F1 score gives balanced measure
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            sources_with_scores.append((source, f1_score))
        
        # Sort sources by score
        sources_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match if available
        if sources_with_scores:
            best_match_source, best_match_score = sources_with_scores[0]
            is_supported = best_match_score >= self.similarity_threshold
            return (is_supported, best_match_source, best_match_score)
        
        return (False, "", 0.0)
    
    def enhance_response(self, answer: str, sources: List[str]) -> str:
        """
        Enhance a response by adding source attributions to reduce hallucination perception.
        
        Args:
            answer (str): The original answer
            sources (List[str]): Source texts
            
        Returns:
            str: Enhanced answer with attributions
        """
        # Check if NLTK is available
        if not have_nltk:
            return answer
        
        # Handle empty inputs gracefully
        if not answer or not sources:
            return answer
            
        # Check factual consistency
        consistency_check = self.check_factual_consistency(answer, sources)
        
        # If there are no claims or all claims are supported, return the original answer
        if not consistency_check["claims"] or not consistency_check["is_hallucination"]:
            return answer
        
        # Calculate confidence score
        confidence = consistency_check["confidence"]
        
        # Add appropriate disclaimers based on confidence level
        if confidence < 0.3:
            disclaimer = "\n\nNote: Much of this response may not be directly supported by the provided sources."
        elif confidence < 0.5:
            disclaimer = "\n\nNote: Some parts of this response may not be directly supported by the provided sources."
        else:
            return answer
            
        # Add disclaimer to the answer
        return answer + disclaimer 