"""
Agent Registry Schema for AgentKG

This module defines the schemas and data models for the Agent Registry layer
of AgentKG. It provides the structure for registering external crews and agents
without storing their implementation details.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class AgentCapability(str, Enum):
    """Enumeration of common agent capabilities"""
    TEXT_ANALYSIS = "text_analysis"
    IMAGE_ANALYSIS = "image_analysis"
    DATA_PROCESSING = "data_processing"
    CONTENT_CREATION = "content_creation"
    CONTENT_MODERATION = "content_moderation"
    DECISION_MAKING = "decision_making"
    PLANNING = "planning"
    INFORMATION_RETRIEVAL = "information_retrieval"
    FRAUD_DETECTION = "fraud_detection"
    SAFETY_COMPLIANCE = "safety_compliance"
    POLICY_ENFORCEMENT = "policy_enforcement"
    CUSTOMER_SERVICE = "customer_service"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class AgentType(str, Enum):
    """Enumeration of agent types"""
    LLM = "llm"  # Large Language Model
    SPECIALIZED = "specialized"  # Specialized for a specific task
    HYBRID = "hybrid"  # Combination of different agents
    HUMAN_IN_LOOP = "human_in_loop"  # Requires human supervision/input


class ApiAuth(str, Enum):
    """Enumeration of API authentication methods"""
    API_KEY = "api_key"
    OAUTH = "oauth"
    JWT = "jwt"
    NONE = "none"


class AgentMetadata(BaseModel):
    """Metadata about an agent"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose and capabilities")
    version: str = Field(..., description="Version of the agent")
    type: AgentType = Field(..., description="Type of agent")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    
    # External repository information
    repository_url: Optional[HttpUrl] = Field(None, description="URL to the agent's code repository")
    repository_path: Optional[str] = Field(None, description="Path within the repository where the agent is defined")
    
    # Optional fields
    model_dependencies: Optional[List[str]] = Field(None, description="List of model dependencies")
    tool_dependencies: Optional[List[str]] = Field(None, description="List of tool dependencies")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class CrewMetadata(BaseModel):
    """Metadata about a crew"""
    crew_id: str = Field(..., description="Unique identifier for the crew")
    name: str = Field(..., description="Name of the crew")
    description: str = Field(..., description="Description of the crew's purpose")
    version: str = Field(..., description="Version of the crew")
    agent_ids: List[str] = Field(..., description="List of agent IDs that compose this crew")
    
    # External repository information
    repository_url: Optional[HttpUrl] = Field(None, description="URL to the crew's code repository")
    repository_path: Optional[str] = Field(None, description="Path within the repository where the crew is defined")
    
    # Process mapping
    process_ids: List[str] = Field(..., description="List of process IDs this crew can handle")
    
    # API information for invoking the crew
    api_endpoint: Optional[HttpUrl] = Field(None, description="API endpoint for invoking this crew")
    api_auth_type: Optional[ApiAuth] = Field(None, description="Authentication type for the API")
    api_docs_url: Optional[HttpUrl] = Field(None, description="URL to the API documentation")
    
    # Optional fields
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class AgentRegistration(BaseModel):
    """Request model for registering an agent"""
    metadata: AgentMetadata = Field(..., description="Metadata about the agent")
    capabilities_detail: Optional[Dict[str, Any]] = Field(None, description="Detailed information about capabilities")


class CrewRegistration(BaseModel):
    """Request model for registering a crew"""
    metadata: CrewMetadata = Field(..., description="Metadata about the crew")
    agent_registrations: Optional[List[AgentRegistration]] = Field(None, description="Optional registrations for new agents")


class RegistrationResponse(BaseModel):
    """Response model for registration requests"""
    success: bool = Field(..., description="Whether the registration was successful")
    id: Optional[str] = Field(None, description="ID of the registered entity if successful")
    message: Optional[str] = Field(None, description="Message about the registration")
    errors: Optional[List[str]] = Field(None, description="List of errors if registration failed")


class DiscoveryQuery(BaseModel):
    """Query model for discovering agents or crews"""
    process_id: Optional[str] = Field(None, description="Process ID to find agents/crews for")
    capabilities: Optional[List[str]] = Field(None, description="Capabilities to search for")
    query_text: Optional[str] = Field(None, description="Free text query for semantic search")
    agent_type: Optional[AgentType] = Field(None, description="Type of agent to search for")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return")


class InvocationRequest(BaseModel):
    """Request model for invoking a crew or agent"""
    id: str = Field(..., description="ID of the crew or agent to invoke")
    is_crew: bool = Field(..., description="Whether the ID is for a crew (True) or agent (False)")
    input_data: Dict[str, Any] = Field(..., description="Input data for the invocation")
    async_execution: Optional[bool] = Field(False, description="Whether to execute asynchronously")
    callback_url: Optional[HttpUrl] = Field(None, description="URL to call back with results if async")