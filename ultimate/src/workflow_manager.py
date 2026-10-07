#!/usr/bin/env python3
"""
Workflow Stage Manager

Handles workflow stage updates for document processing pipeline.
Integrates with external API to update file processing stages.
"""

import os
import time
import logging
import zipapp
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for workflow stage updates."""
    api_url: str
    access_key: str
    stor_api_key: str
    enabled: bool = True
    timeout: int = 10
    retry_attempts: int = 3

class WorkflowStageManager:
    """Manages workflow stage updates during document processing."""
    
    # Define workflow stages
    WORKFLOW_STAGES = {
        'DOWNLOADING': 'DOWNLOADING',
        'PROCESSING': 'PROCESSING', 
        'IMAGE_PROCESSING': 'IMAGE_PROCESSING',
        'OCR_EXTRACTION': 'OCR_EXTRACTION',
        'TEXT_NORMALIZATION': 'TEXT_NORMALIZATION',
        'VISION_ANALYSIS': 'VISION_ANALYSIS',
        'FINALIZATION': 'FINALIZATION',
        'COMPLETED': 'COMPLETED',
        'FAILED': 'FAILED'
    }
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize workflow stage manager."""
        if config is None:
            # Load from environment variables
            config = WorkflowConfig(
                api_url=os.getenv('WORKFLOW_API_URL', 'http://localhost:3333'),
                access_key=os.getenv('WORKFLOW_ACCESS_KEY', ''),
                stor_api_key=os.getenv('WORKFLOW_STOR_API_KEY', ''),
                enabled=os.getenv('WORKFLOW_ENABLED', 'true').lower() == 'true'
            )
        
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'access-key': self.config.access_key,
            'stor-api-key': self.config.stor_api_key,
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Workflow manager initialized - Enabled: {self.config.enabled}")
    
    def update_workflow_stage(self, file_id: str, stage: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update workflow stage for a file.
        
        Args:
            file_id: Unique identifier for the file
            stage: Workflow stage to set
            metadata: Optional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            logger.debug(f"Workflow updates disabled - would update {file_id} to {stage}")
            return True
        
        if not file_id:
            logger.error("File ID is required for workflow stage update")
            return False
        
        # Check if file_id is a valid MongoDB ObjectId format (24 hex characters)
        # If not, skip workflow update to avoid API errors
        import re
        object_id_pattern = re.compile(r'^[0-9a-fA-F]{24}$')
        if not object_id_pattern.match(file_id):
            logger.debug(f"Skipping workflow update for non-ObjectId file_id: {file_id} (external API requires ObjectId format)")
            return True  # Return True to not block processing
        
        if stage not in self.WORKFLOW_STAGES.values():
            logger.error(f"Invalid workflow stage: {stage}")
            return False
        
        # Prepare request data
        data = {
            "fileId": file_id,
            "workflowStage": stage
        }
        
        # Add metadata if provided
        if metadata:
            data["metadata"] = metadata
        
        # Make API request with retries
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.put(
                    f"{self.config.api_url}/api/file/update-workflow-stage",
                    json=data,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"Workflow stage updated: {file_id} -> {stage}")
                    return True
                else:
                    logger.warning(f"Workflow update failed (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Workflow update request failed (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        logger.error(f"Failed to update workflow stage after {self.config.retry_attempts} attempts")
        return False
    
    def update_downloading(self, file_id: str) -> bool:
        """Update stage to DOWNLOADING."""
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['DOWNLOADING'])
    
    def update_processing(self, file_id: str) -> bool:
        """Update stage to PROCESSING."""
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['PROCESSING'])
    
    def update_image_processing(self, file_id: str) -> bool:
        """Update stage to IMAGE_PROCESSING."""
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['IMAGE_PROCESSING'])
    
    def update_ocr_extraction(self, file_id: str, confidence: float = None) -> bool:
        """Update stage to OCR_EXTRACTION."""
        metadata = {}
        if confidence is not None:
            metadata['confidence'] = confidence
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['OCR_EXTRACTION'], metadata)
    
    def update_text_normalization(self, file_id: str) -> bool:
        """Update stage to TEXT_NORMALIZATION."""
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['TEXT_NORMALIZATION'])
    
    def update_vision_analysis(self, file_id: str, objects_count: int = None) -> bool:
        """Update stage to VISION_ANALYSIS."""
        metadata = {}
        if objects_count is not None:
            metadata['objects_detected'] = objects_count
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['VISION_ANALYSIS'], metadata)
    
    def update_finalization(self, file_id: str) -> bool:
        """Update stage to FINALIZATION."""
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['FINALIZATION'])
    
    def update_completed(self, file_id: str, processing_time: float = None, text_length: int = None) -> bool:
        """Update stage to COMPLETED."""
        metadata = {}
        if processing_time is not None:
            metadata['processing_time'] = processing_time
        if text_length is not None:
            metadata['text_length'] = text_length
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['COMPLETED'], metadata)
    
    def update_failed(self, file_id: str, error_message: str = None) -> bool:
        """Update stage to FAILED."""
        metadata = {}
        if error_message:
            metadata['error'] = error_message
        return self.update_workflow_stage(file_id, self.WORKFLOW_STAGES['FAILED'], metadata)

# Global workflow manager instance
workflow_manager = None

def get_workflow_manager() -> WorkflowStageManager:
    """Get or create global workflow manager instance."""
    global workflow_manager
    if workflow_manager is None:
        workflow_manager = WorkflowStageManager()
    return workflow_manager

def update_workflow_stage(file_id: str, stage: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to update workflow stage."""
    manager = get_workflow_manager()
    return manager.update_workflow_stage(file_id, stage, metadata)