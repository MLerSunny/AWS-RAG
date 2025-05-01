"""
Hallucination detection and factual consistency checking.
"""
import re
import nltk
from typing import Dict, List, Tuple, Optional
from ...utils.logger import setup_logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = setup_logger(__name__)

class HallucinationDetector:
    """Detector for hallucinations in model responses."""
    
    def __init__(self):
        """Initialize the hallucination detector."""
        logger.info("Initialized hallucination detector")
        
    def check_factual_consistency(self, answer: str, sources: List[str]) -> Dict:
        """
        Check if the answer is consistent with the provided sources.
        
        Args:
            answer (str): The model's answer
            sources (List[str]): List of source texts to check against
            
        Returns:
            Dict: Analysis of factual consistency
        """
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
        
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[str]: Extracted factual claims
        """
        sentences = nltk.sent_tokenize(text)
        
        # Filter sentences that are likely to be factual claims
        # This is a simplified implementation - in production you would
        # use a more sophisticated approach, potentially with a classifier
        claims = [s for s in sentences if self._is_factual_claim(s)]
        
        logger.info(f"Extracted {len(claims)} claims from text")
        return claims
        
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence is likely a factual claim.
        
        Args:
            sentence (str): The sentence to analyze
            
        Returns:
            bool: Whether the sentence is likely a factual claim
        """
        # Simple heuristic approach:
        # 1. Look for numbers, dates, specific entities
        # 2. Avoid sentences that are questions, commands, or personal opinions
        
        # Skip questions
        if sentence.endswith("?"):
            return False
            
        # Skip short sentences
        if len(sentence.split()) < 4:
            return False
            
        # Skip sentences with first-person opinions
        opinion_patterns = [
            r"\bI think\b", r"\bI believe\b", r"\bIn my opinion\b",
            r"\bI feel\b", r"\bI would\b", r"\bI'd\b"
        ]
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in opinion_patterns):
            return False
        
        # Look for indicators of factual claims
        has_number = bool(re.search(r'\d', sentence))
        has_entity = bool(re.search(r'[A-Z][a-z]+', sentence))
        has_date = bool(re.search(r'\b(in|on|during|after|before)\s+\d{4}\b', sentence))
        
        # Look for factual language patterns
        factual_patterns = [
            r"\bis\b", r"\bwas\b", r"\bwere\b", r"\bhave\b", r"\bhas\b",
            r"\bconsists of\b", r"\bcomprises\b", r"\bcontains\b"
        ]
        has_factual_pattern = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_patterns)
        
        # Sentence is a claim if it has any of these indicators
        return has_number or has_date or (has_entity and has_factual_pattern)
        
    def _find_support_in_sources(self, claim: str, sources: List[str]) -> Tuple[bool, str, float]:
        """
        Find support for a claim in the provided sources.
        
        Args:
            claim (str): The claim to check
            sources (List[str]): Source texts to check against
            
        Returns:
            Tuple[bool, str, float]: (is_supported, supporting_source, confidence)
        """
        # This is a simplified implementation using text similarity
        # In a production system, you would use a more sophisticated approach,
        # such as entailment models or semantic search
        
        # Tokenize claim and remove stop words
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        
        best_match_score = 0
        best_match_source = ""
        
        for source in sources:
            # Calculate a simple overlap score
            source_words = set(re.findall(r'\b\w+\b', source.lower()))
            
            if not source_words:
                continue
                
            # Jaccard similarity
            overlap = len(claim_words.intersection(source_words))
            union = len(claim_words.union(source_words))
            
            similarity = overlap / len(claim_words) if claim_words else 0
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_source = source
        
        # Consider it supported if the similarity is above a threshold
        threshold = 0.5
        is_supported = best_match_score >= threshold
        
        return (is_supported, best_match_source, best_match_score)
    
    def enhance_response(self, answer: str, sources: List[str]) -> str:
        """
        Enhance a response by adding source attributions to reduce hallucination perception.
        
        Args:
            answer (str): The original answer
            sources (List[str]): Source texts
            
        Returns:
            str: Enhanced answer with attributions
        """
        # Check factual consistency
        consistency_check = self.check_factual_consistency(answer, sources)
        
        # If there are no claims or all claims are supported, return the original answer
        if not consistency_check["claims"] or not consistency_check["is_hallucination"]:
            return answer
            
        # Identify unsupported claims
        unsupported_claims = [claim["claim"] for claim in consistency_check["claims"] if not claim["supported"]]
        
        # Add a disclaimer for potentially unsupported information
        disclaimer = "\n\nNote: Some information in this response may not be directly supported by the provided sources."
        
        # Create enhanced response
        enhanced_answer = answer + disclaimer
        
        return enhanced_answer 