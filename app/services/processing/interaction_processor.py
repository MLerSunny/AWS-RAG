import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Sequence, Union
from datetime import datetime, timedelta
from app.services.storage.dynamodb_service import DynamoDBService
from app.utils.logger import setup_logger
from app.utils.dynamodb_types import UserInteraction, DynamoDBTypeConverter

logger = setup_logger(__name__)

def anonymize_user_id(user_id: Optional[str]) -> str:
    if not user_id:
        return "anonymous"
    return hashlib.sha256(user_id.encode()).hexdigest()

def fetch_interactions(
    days: int = 7,
    min_feedback: Optional[bool] = True,
    table_prefix: str = "genai_"
) -> List[UserInteraction]:
    """
    Fetch recent user interactions from DynamoDB.
    
    Args:
        days: Number of days of history to fetch
        min_feedback: If True, only fetch interactions with feedback
        table_prefix: Prefix for DynamoDB table names
        
    Returns:
        List of UserInteraction instances
    """
    db_service = DynamoDBService(table_prefix=table_prefix)
    table_name = f"{table_prefix}user_interactions"
    table = db_service._get_table(table_name)
    since = datetime.utcnow() - timedelta(days=days)
    since_ts = since.timestamp()
    
    # Scan for recent items
    response = table.scan()
    items = response.get("Items", [])
    interactions = []
    
    for item in items:
        try:
            # Convert DynamoDB item to UserInteraction
            interaction = UserInteraction.from_dynamodb(item)
            
            # Apply filters
            if interaction.timestamp < since_ts:
                continue
            if min_feedback and interaction.is_helpful is None:
                continue
                
            interactions.append(interaction)
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}")
            continue
            
    return interactions

def process_interactions_to_jsonl(
    interactions: Sequence[Union[UserInteraction, Dict[str, Any]]],
    output_file: str
) -> bool:
    """
    Process interactions and save to JSONL file.
    
    Args:
        interactions: List of UserInteraction instances or raw interaction dicts
        output_file: Path to output JSONL file
        
    Returns:
        bool: True if processing was successful
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for interaction in interactions:
                # Convert UserInteraction to dict if needed
                if isinstance(interaction, UserInteraction):
                    data = interaction.dict()
                else:
                    data = interaction
                
                # Anonymize user ID if present
                if 'user_id' in data and data['user_id']:
                    data['user_id'] = anonymize_user_id(data['user_id'])
                
                # Write to JSONL
                f.write(json.dumps(data) + '\n')
        return True
    except Exception as e:
        logger.error(f"Error processing interactions to JSONL: {str(e)}")
        return False

def run_processing_pipeline(
    output_path: str = "processed_interactions.jsonl",
    days: int = 7,
    min_feedback: Optional[bool] = True,
    table_prefix: str = "genai_"
):
    """
    Main entry point to fetch, process, and output fine-tuning data.
    """
    logger.info("Starting interaction processing pipeline...")
    interactions = fetch_interactions(days=days, min_feedback=min_feedback, table_prefix=table_prefix)
    count = process_interactions_to_jsonl(interactions, output_path)
    logger.info(f"Processing pipeline complete. {count} records written.") 