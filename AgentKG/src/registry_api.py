"""
Agent Registry API for AgentKG

This module implements a FastAPI-based API for registering and discovering
agents and crews in the AgentKG knowledge graph.
"""

import os
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
import datetime
import uuid
import json

from agent_registry_schema import (
    AgentRegistration, 
    CrewRegistration, 
    RegistrationResponse,
    DiscoveryQuery,
    InvocationRequest,
    AgentMetadata,
    CrewMetadata
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AgentKG Registry API",
    description="API for registering and discovering agents and crews in the AgentKG knowledge graph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Neo4jConnector:
    """
    A connector class for Neo4j database operations
    """
    
    def __init__(self):
        """Initialize the Neo4j connection using environment variables"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def execute_query(self, query, params=None):
        """
        Execute a Cypher query and return results
        
        Args:
            query (str): The Cypher query to execute
            params (dict, optional): Parameters for the query
            
        Returns:
            list: Query results
        """
        if params is None:
            params = {}
            
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return [record for record in result]


# Create database connector instance
neo4j_connector = Neo4jConnector()


# Dependency for handling API key authentication
async def api_key_auth(x_api_key: str = Header(None)):
    """Validate API key"""
    expected_api_key = os.getenv("AGENTKG_API_KEY")
    if not expected_api_key:
        # If no API key is set, don't validate (for development)
        return True
    
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


@app.on_event("shutdown")
def shutdown_event():
    """Close Neo4j connection on shutdown"""
    neo4j_connector.close()


@app.post("/agents/register", response_model=RegistrationResponse, tags=["Agents"])
async def register_agent(
    registration: AgentRegistration, 
    authorized: bool = Depends(api_key_auth)
):
    """
    Register an agent in the AgentKG knowledge graph
    
    This endpoint allows external services to register agent metadata.
    It doesn't store the actual agent implementation, just the metadata.
    """
    try:
        # Generate timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract metadata
        metadata = registration.metadata
        
        # Generate a UUID if agent_id is not provided or is empty
        if not metadata.agent_id:
            metadata.agent_id = f"AGENT-{str(uuid.uuid4())}"
        
        # Check if agent already exists
        check_query = """
        MATCH (a:Agent {agentId: $agentId})
        RETURN a
        """
        
        check_result = neo4j_connector.execute_query(check_query, {"agentId": metadata.agent_id})
        
        if check_result:
            # Update existing agent
            update_query = """
            MATCH (a:Agent {agentId: $agentId})
            SET 
                a.name = $name,
                a.description = $description,
                a.version = $version,
                a.type = $type,
                a.capabilities = $capabilities,
                a.repositoryUrl = $repositoryUrl,
                a.repositoryPath = $repositoryPath,
                a.modelDependencies = $modelDependencies,
                a.toolDependencies = $toolDependencies,
                a.updatedAt = $updatedAt
            RETURN a.agentId as agentId
            """
            
            result = neo4j_connector.execute_query(update_query, {
                "agentId": metadata.agent_id,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "type": metadata.type.value,
                "capabilities": metadata.capabilities,
                "repositoryUrl": str(metadata.repository_url) if metadata.repository_url else None,
                "repositoryPath": metadata.repository_path,
                "modelDependencies": metadata.model_dependencies,
                "toolDependencies": metadata.tool_dependencies,
                "updatedAt": timestamp
            })
            
            return RegistrationResponse(
                success=True,
                id=metadata.agent_id,
                message=f"Agent {metadata.name} (ID: {metadata.agent_id}) updated successfully"
            )
            
        else:
            # Create new agent
            create_query = """
            CREATE (a:Agent {
                agentId: $agentId,
                name: $name,
                description: $description,
                version: $version,
                type: $type,
                capabilities: $capabilities,
                repositoryUrl: $repositoryUrl,
                repositoryPath: $repositoryPath,
                modelDependencies: $modelDependencies,
                toolDependencies: $toolDependencies,
                createdAt: $createdAt,
                updatedAt: $createdAt
            })
            RETURN a.agentId as agentId
            """
            
            result = neo4j_connector.execute_query(create_query, {
                "agentId": metadata.agent_id,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "type": metadata.type.value,
                "capabilities": metadata.capabilities,
                "repositoryUrl": str(metadata.repository_url) if metadata.repository_url else None,
                "repositoryPath": metadata.repository_path,
                "modelDependencies": metadata.model_dependencies,
                "toolDependencies": metadata.tool_dependencies,
                "createdAt": timestamp
            })
            
            # Store capabilities details if provided
            if registration.capabilities_detail:
                # Store capabilities detail as a property
                capabilities_query = """
                MATCH (a:Agent {agentId: $agentId})
                SET a.capabilitiesDetail = $capabilitiesDetail
                """
                
                neo4j_connector.execute_query(capabilities_query, {
                    "agentId": metadata.agent_id,
                    "capabilitiesDetail": json.dumps(registration.capabilities_detail)
                })
            
            return RegistrationResponse(
                success=True,
                id=metadata.agent_id,
                message=f"Agent {metadata.name} (ID: {metadata.agent_id}) registered successfully"
            )
        
    except Exception as e:
        return RegistrationResponse(
            success=False,
            errors=[str(e)]
        )


@app.post("/crews/register", response_model=RegistrationResponse, tags=["Crews"])
async def register_crew(
    registration: CrewRegistration, 
    authorized: bool = Depends(api_key_auth)
):
    """
    Register a crew in the AgentKG knowledge graph
    
    This endpoint allows external services to register crew metadata.
    It doesn't store the actual crew implementation, just the metadata
    and relationships to processes and agents.
    """
    try:
        # Generate timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract metadata
        metadata = registration.metadata
        
        # Generate a UUID if crew_id is not provided or is empty
        if not metadata.crew_id:
            metadata.crew_id = f"CREW-{str(uuid.uuid4())}"
        
        # Check if crew already exists
        check_query = """
        MATCH (c:Crew {crewId: $crewId})
        RETURN c
        """
        
        check_result = neo4j_connector.execute_query(check_query, {"crewId": metadata.crew_id})
        
        # Register new agents if provided
        if registration.agent_registrations:
            for agent_reg in registration.agent_registrations:
                # Register agent using the agent registration endpoint
                agent_result = await register_agent(agent_reg)
                if not agent_result.success:
                    return RegistrationResponse(
                        success=False,
                        errors=[f"Failed to register agent {agent_reg.metadata.name}: {agent_result.errors}"]
                    )
        
        if check_result:
            # Update existing crew
            update_query = """
            MATCH (c:Crew {crewId: $crewId})
            SET 
                c.name = $name,
                c.description = $description,
                c.version = $version,
                c.repositoryUrl = $repositoryUrl,
                c.repositoryPath = $repositoryPath,
                c.processIds = $processIds,
                c.apiEndpoint = $apiEndpoint,
                c.apiAuthType = $apiAuthType,
                c.apiDocsUrl = $apiDocsUrl,
                c.updatedAt = $updatedAt
            RETURN c.crewId as crewId
            """
            
            result = neo4j_connector.execute_query(update_query, {
                "crewId": metadata.crew_id,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "repositoryUrl": str(metadata.repository_url) if metadata.repository_url else None,
                "repositoryPath": metadata.repository_path,
                "processIds": metadata.process_ids,
                "apiEndpoint": str(metadata.api_endpoint) if metadata.api_endpoint else None,
                "apiAuthType": metadata.api_auth_type.value if metadata.api_auth_type else None,
                "apiDocsUrl": str(metadata.api_docs_url) if metadata.api_docs_url else None,
                "updatedAt": timestamp
            })
            
            # Clear existing agent relationships
            clear_agents_query = """
            MATCH (c:Crew {crewId: $crewId})<-[r:MEMBER_OF]-(a:Agent)
            DELETE r
            """
            
            neo4j_connector.execute_query(clear_agents_query, {"crewId": metadata.crew_id})
            
            # Clear existing process relationships
            clear_processes_query = """
            MATCH (c:Crew {crewId: $crewId})-[r:HANDLES]->(p:Process)
            DELETE r
            """
            
            neo4j_connector.execute_query(clear_processes_query, {"crewId": metadata.crew_id})
            
        else:
            # Create new crew
            create_query = """
            CREATE (c:Crew {
                crewId: $crewId,
                name: $name,
                description: $description,
                version: $version,
                repositoryUrl: $repositoryUrl,
                repositoryPath: $repositoryPath,
                processIds: $processIds,
                apiEndpoint: $apiEndpoint,
                apiAuthType: $apiAuthType,
                apiDocsUrl: $apiDocsUrl,
                createdAt: $createdAt,
                updatedAt: $createdAt
            })
            RETURN c.crewId as crewId
            """
            
            result = neo4j_connector.execute_query(create_query, {
                "crewId": metadata.crew_id,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "repositoryUrl": str(metadata.repository_url) if metadata.repository_url else None,
                "repositoryPath": metadata.repository_path,
                "processIds": metadata.process_ids,
                "apiEndpoint": str(metadata.api_endpoint) if metadata.api_endpoint else None,
                "apiAuthType": metadata.api_auth_type.value if metadata.api_auth_type else None,
                "apiDocsUrl": str(metadata.api_docs_url) if metadata.api_docs_url else None,
                "createdAt": timestamp
            })
        
        # Connect agents to crew
        for agent_id in metadata.agent_ids:
            connect_agent_query = """
            MATCH (c:Crew {crewId: $crewId})
            MATCH (a:Agent {agentId: $agentId})
            MERGE (a)-[:MEMBER_OF]->(c)
            """
            
            neo4j_connector.execute_query(connect_agent_query, {
                "crewId": metadata.crew_id,
                "agentId": agent_id
            })
        
        # Connect crew to processes
        for process_id in metadata.process_ids:
            connect_process_query = """
            MATCH (c:Crew {crewId: $crewId})
            MATCH (p:Process {processId: $processId})
            MERGE (c)-[:HANDLES]->(p)
            """
            
            neo4j_connector.execute_query(connect_process_query, {
                "crewId": metadata.crew_id,
                "processId": process_id
            })
        
        return RegistrationResponse(
            success=True,
            id=metadata.crew_id,
            message=f"Crew {metadata.name} (ID: {metadata.crew_id}) registered successfully with {len(metadata.agent_ids)} agents and {len(metadata.process_ids)} processes"
        )
        
    except Exception as e:
        return RegistrationResponse(
            success=False,
            errors=[str(e)]
        )


@app.post("/discovery/query", tags=["Discovery"])
async def discover_agents_crews(
    query: DiscoveryQuery,
    authorized: bool = Depends(api_key_auth)
):
    """
    Discover agents and crews based on process IDs, capabilities, or free text
    
    This endpoint allows external services to find appropriate agents and crews
    for specific tasks or processes.
    """
    try:
        results = {
            "agents": [],
            "crews": []
        }
        
        # Query based on process ID
        if query.process_id:
            # Find crews that handle this process
            crew_query = """
            MATCH (c:Crew)-[:HANDLES]->(p:Process {processId: $processId})
            RETURN 
                c.crewId as crewId,
                c.name as name,
                c.description as description,
                c.version as version,
                c.repositoryUrl as repositoryUrl,
                c.processIds as processIds,
                c.apiEndpoint as apiEndpoint,
                c.apiAuthType as apiAuthType
            LIMIT $limit
            """
            
            crew_results = neo4j_connector.execute_query(crew_query, {
                "processId": query.process_id,
                "limit": query.max_results
            })
            
            for crew in crew_results:
                # Get agent members
                agent_query = """
                MATCH (a:Agent)-[:MEMBER_OF]->(c:Crew {crewId: $crewId})
                RETURN collect(a.agentId) as agentIds
                """
                
                agent_results = neo4j_connector.execute_query(agent_query, {"crewId": crew["crewId"]})
                
                results["crews"].append({
                    "crew_id": crew["crewId"],
                    "name": crew["name"],
                    "description": crew["description"],
                    "version": crew["version"],
                    "repository_url": crew["repositoryUrl"],
                    "process_ids": crew["processIds"],
                    "api_endpoint": crew["apiEndpoint"],
                    "api_auth_type": crew["apiAuthType"],
                    "agent_ids": agent_results[0]["agentIds"] if agent_results else []
                })
            
        # Query based on capabilities
        if query.capabilities:
            # For each capability, find agents that have it
            for capability in query.capabilities:
                agent_query = """
                MATCH (a:Agent)
                WHERE $capability IN a.capabilities
                RETURN 
                    a.agentId as agentId,
                    a.name as name,
                    a.description as description,
                    a.version as version,
                    a.type as type,
                    a.capabilities as capabilities,
                    a.repositoryUrl as repositoryUrl
                LIMIT $limit
                """
                
                agent_results = neo4j_connector.execute_query(agent_query, {
                    "capability": capability,
                    "limit": query.max_results
                })
                
                for agent in agent_results:
                    # Check if agent is already in results
                    if not any(a["agent_id"] == agent["agentId"] for a in results["agents"]):
                        results["agents"].append({
                            "agent_id": agent["agentId"],
                            "name": agent["name"],
                            "description": agent["description"],
                            "version": agent["version"],
                            "type": agent["type"],
                            "capabilities": agent["capabilities"],
                            "repository_url": agent["repositoryUrl"]
                        })
        
        # Query based on free text (using Neo4j's full-text search if available)
        if query.query_text:
            # Search for agents matching the text
            agent_query = """
            MATCH (a:Agent)
            WHERE a.name CONTAINS $queryText OR a.description CONTAINS $queryText
            RETURN 
                a.agentId as agentId,
                a.name as name,
                a.description as description,
                a.version as version,
                a.type as type,
                a.capabilities as capabilities,
                a.repositoryUrl as repositoryUrl
            LIMIT $limit
            """
            
            agent_results = neo4j_connector.execute_query(agent_query, {
                "queryText": query.query_text,
                "limit": query.max_results
            })
            
            for agent in agent_results:
                # Check if agent is already in results
                if not any(a["agent_id"] == agent["agentId"] for a in results["agents"]):
                    results["agents"].append({
                        "agent_id": agent["agentId"],
                        "name": agent["name"],
                        "description": agent["description"],
                        "version": agent["version"],
                        "type": agent["type"],
                        "capabilities": agent["capabilities"],
                        "repository_url": agent["repositoryUrl"]
                    })
            
            # Search for crews matching the text
            crew_query = """
            MATCH (c:Crew)
            WHERE c.name CONTAINS $queryText OR c.description CONTAINS $queryText
            RETURN 
                c.crewId as crewId,
                c.name as name,
                c.description as description,
                c.version as version,
                c.repositoryUrl as repositoryUrl,
                c.processIds as processIds,
                c.apiEndpoint as apiEndpoint,
                c.apiAuthType as apiAuthType
            LIMIT $limit
            """
            
            crew_results = neo4j_connector.execute_query(crew_query, {
                "queryText": query.query_text,
                "limit": query.max_results
            })
            
            for crew in crew_results:
                # Check if crew is already in results
                if not any(c["crew_id"] == crew["crewId"] for c in results["crews"]):
                    # Get agent members
                    agent_query = """
                    MATCH (a:Agent)-[:MEMBER_OF]->(c:Crew {crewId: $crewId})
                    RETURN collect(a.agentId) as agentIds
                    """
                    
                    agent_results = neo4j_connector.execute_query(agent_query, {"crewId": crew["crewId"]})
                    
                    results["crews"].append({
                        "crew_id": crew["crewId"],
                        "name": crew["name"],
                        "description": crew["description"],
                        "version": crew["version"],
                        "repository_url": crew["repositoryUrl"],
                        "process_ids": crew["processIds"],
                        "api_endpoint": crew["apiEndpoint"],
                        "api_auth_type": crew["apiAuthType"],
                        "agent_ids": agent_results[0]["agentIds"] if agent_results else []
                    })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invoke", tags=["Execution"])
async def invoke_agent_crew(
    request: InvocationRequest,
    authorized: bool = Depends(api_key_auth)
):
    """
    Invoke a registered agent or crew via its API endpoint
    
    This endpoint allows external services to invoke registered agents or crews.
    It acts as a proxy to the actual implementation's API endpoint.
    """
    try:
        # Get the entity (crew or agent)
        if request.is_crew:
            entity_query = """
            MATCH (c:Crew {crewId: $id})
            RETURN 
                c.apiEndpoint as apiEndpoint,
                c.apiAuthType as apiAuthType,
                c.name as name
            """
            
            entity_results = neo4j_connector.execute_query(entity_query, {"id": request.id})
            
            if not entity_results:
                raise HTTPException(status_code=404, detail=f"Crew with ID {request.id} not found")
            
            entity = entity_results[0]
            
        else:
            entity_query = """
            MATCH (a:Agent {agentId: $id})
            RETURN 
                a.apiEndpoint as apiEndpoint,
                a.apiAuthType as apiAuthType,
                a.name as name
            """
            
            entity_results = neo4j_connector.execute_query(entity_query, {"id": request.id})
            
            if not entity_results:
                raise HTTPException(status_code=404, detail=f"Agent with ID {request.id} not found")
            
            entity = entity_results[0]
        
        # Check if API endpoint is available
        if not entity["apiEndpoint"]:
            raise HTTPException(
                status_code=400, 
                detail=f"No API endpoint available for {request.id}"
            )
        
        # In a real implementation, you would make an HTTP request to the entity's API endpoint
        # For now, return a mock response
        return {
            "status": "success",
            "message": f"Invocation request for {entity['name']} forwarded to {entity['apiEndpoint']}",
            "request_id": str(uuid.uuid4()),
            "async": request.async_execution
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "time": datetime.datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)