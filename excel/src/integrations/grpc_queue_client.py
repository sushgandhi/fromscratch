"""
gRPC Queue Client for Excel Agent Integration
Handles communication with mediator queue system
"""

import grpc
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types for queue communication"""
    INTERMEDIATE_RESULT = "intermediate_result"
    FINAL_RESULT = "final_result"
    ERROR = "error"
    STATUS_UPDATE = "status_update"

@dataclass
class QueueMessage:
    """Standard message format for queue communication"""
    message_id: str
    session_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class GRPCQueueClient:
    """Client for communicating with gRPC mediator queue"""
    
    def __init__(self, 
                 queue_endpoint: str,
                 queue_name: str = "excel_agent_queue",
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """
        Initialize gRPC queue client
        
        Args:
            queue_endpoint: gRPC server endpoint (e.g., "localhost:50051")
            queue_name: Name of the queue to use
            max_retries: Maximum retry attempts for failed sends
            timeout: Request timeout in seconds
        """
        self.queue_endpoint = queue_endpoint
        self.queue_name = queue_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.channel = None
        self.stub = None
        
    async def connect(self):
        """Establish connection to gRPC queue server"""
        try:
            self.channel = grpc.aio.insecure_channel(self.queue_endpoint)
            # Note: You'd need to generate the actual gRPC stub from .proto files
            # self.stub = queue_pb2_grpc.QueueServiceStub(self.channel)
            logger.info(f"Connected to gRPC queue at {self.queue_endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to gRPC queue: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to gRPC queue server"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from gRPC queue")
    
    async def send_message(self, message: QueueMessage) -> bool:
        """
        Send message to queue with retry logic
        
        Args:
            message: Message to send
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                # Convert message to gRPC format
                grpc_message = self._to_grpc_message(message)
                
                # Send message (this would use the actual gRPC stub)
                # response = await self.stub.SendMessage(grpc_message, timeout=self.timeout)
                
                logger.info(f"Message sent successfully: {message.message_id}")
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to send message after {self.max_retries} attempts")
                    return False
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def send_intermediate_result(self, 
                                     session_id: str,
                                     operation_id: str,
                                     operation_name: str,
                                     result_data: Any,
                                     metadata: Optional[Dict] = None) -> bool:
        """Send intermediate operation result to queue"""
        
        message = QueueMessage(
            message_id=f"{session_id}_{operation_id}",
            session_id=session_id,
            message_type=MessageType.INTERMEDIATE_RESULT,
            payload={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "result_data": result_data,
                "success": True
            },
            metadata=metadata
        )
        
        return await self.send_message(message)
    
    async def send_final_result(self,
                              session_id: str,
                              final_result: Dict[str, Any]) -> bool:
        """Send final Excel analysis result to queue"""
        
        message = QueueMessage(
            message_id=f"{session_id}_final",
            session_id=session_id,
            message_type=MessageType.FINAL_RESULT,
            payload=final_result,
            metadata={
                "component": "excel_agent",
                "result_type": "analysis_complete"
            }
        )
        
        return await self.send_message(message)
    
    async def send_error(self,
                        session_id: str,
                        error_message: str,
                        error_context: Optional[Dict] = None) -> bool:
        """Send error message to queue"""
        
        message = QueueMessage(
            message_id=f"{session_id}_error",
            session_id=session_id,
            message_type=MessageType.ERROR,
            payload={
                "error_message": error_message,
                "error_context": error_context or {},
                "component": "excel_agent"
            }
        )
        
        return await self.send_message(message)
    
    async def send_status_update(self,
                               session_id: str,
                               status: str,
                               progress: Optional[float] = None) -> bool:
        """Send status update to queue"""
        
        message = QueueMessage(
            message_id=f"{session_id}_status",
            session_id=session_id,
            message_type=MessageType.STATUS_UPDATE,
            payload={
                "status": status,
                "progress": progress,
                "component": "excel_agent"
            }
        )
        
        return await self.send_message(message)
    
    def _to_grpc_message(self, message: QueueMessage) -> Dict[str, Any]:
        """Convert QueueMessage to gRPC message format"""
        return {
            "message_id": message.message_id,
            "session_id": message.session_id,
            "message_type": message.message_type.value,
            "payload": json.dumps(message.payload),
            "metadata": json.dumps(message.metadata or {}),
            "timestamp": message.timestamp
        }

class QueueConfig:
    """Configuration for queue integration"""
    
    def __init__(self, 
                 queue_endpoint: str,
                 queue_name: str = "excel_agent_queue",
                 enable_intermediate_results: bool = True,
                 enable_status_updates: bool = True,
                 batch_size: int = 1):
        self.queue_endpoint = queue_endpoint
        self.queue_name = queue_name
        self.enable_intermediate_results = enable_intermediate_results
        self.enable_status_updates = enable_status_updates
        self.batch_size = batch_size 