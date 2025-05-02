import json
import argparse
from typing import Dict, List
from collections import defaultdict
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def analyze_feedback(feedback_file: str) -> Dict:
    """Analyze user feedback data."""
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        # Initialize metrics
        metrics = {
            "total_queries": len(feedback_data),
            "positive_feedback": 0,
            "negative_feedback": 0,
            "common_queries": defaultdict(int),
            "response_times": []
        }
        
        # Process feedback
        for entry in feedback_data:
            # Count feedback
            if entry.get("feedback") == "positive":
                metrics["positive_feedback"] += 1
            elif entry.get("feedback") == "negative":
                metrics["negative_feedback"] += 1
            
            # Track common queries
            query = entry.get("query", "").lower()
            metrics["common_queries"][query] += 1
            
            # Track response times
            if "response_time" in entry:
                metrics["response_times"].append(entry["response_time"])
        
        # Calculate averages
        if metrics["response_times"]:
            metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
        
        # Get top queries
        metrics["top_queries"] = sorted(
            metrics["common_queries"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing feedback: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Analyze user feedback data")
    parser.add_argument("feedback_file", help="JSON file containing feedback data")
    parser.add_argument("--output", help="Output file for analysis results")
    args = parser.parse_args()
    
    # Analyze feedback
    metrics = analyze_feedback(args.feedback_file)
    
    # Print results
    print("\nFeedback Analysis Results:")
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Positive Feedback: {metrics['positive_feedback']}")
    print(f"Negative Feedback: {metrics['negative_feedback']}")
    print(f"Average Response Time: {metrics.get('avg_response_time', 'N/A')}s")
    
    print("\nTop 10 Queries:")
    for query, count in metrics["top_queries"]:
        print(f"- {query}: {count} times")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Analysis results saved to {args.output}")

if __name__ == "__main__":
    main() 