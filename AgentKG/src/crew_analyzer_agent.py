"""
Crew Analyzer Agent for AgentKG

This module provides a GraphRAG-powered agent that can analyze crew repositories
(code, documentation, diagrams) and extract metadata for registration in AgentKG.
"""

import os
import re
import sys
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from pydantic import BaseModel, Field, HttpUrl

# Load AgentKG schema definitions
from agent_registry_schema import (
    AgentRegistration, 
    CrewRegistration, 
    AgentMetadata,
    CrewMetadata,
    AgentType,
    ApiAuth
)

# Load environment variables
load_dotenv()


class RepositoryAnalysisResult(BaseModel):
    """Results of repository analysis"""
    crew_metadata: Optional[CrewMetadata] = Field(None, description="Metadata about the crew")
    agent_metadata: List[AgentMetadata] = Field(default_factory=list, description="Metadata about agents in the crew")
    process_ids: List[str] = Field(default_factory=list, description="Process IDs this crew can handle")
    capabilities: List[str] = Field(default_factory=list, description="Capabilities detected in the repository")
    confidence_score: float = Field(0.0, description="Confidence score of the analysis (0.0-1.0)")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during analysis")


class CrewAnalyzerAgent:
    """
    An agent that analyzes crew repositories to extract metadata
    for registration in the AgentKG knowledge graph.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the CrewAnalyzerAgent
        
        Args:
            api_base_url: Base URL for the AgentKG Registry API
        """
        self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings()
        self.api_base_url = api_base_url.rstrip("/")
        
        # Init knowledge graph connection
        self.api_key = os.getenv("AGENTKG_API_KEY", "")
        
        # Load process hierarchy for mapping
        self.processes = self._load_process_hierarchy()
    
    def _load_process_hierarchy(self) -> Dict[str, Dict]:
        """
        Load the process hierarchy from the knowledge graph
        
        Returns:
            Dictionary mapping process IDs to process details
        """
        # In production, this would query the knowledge graph
        # For demo purposes, we'll use a local cache
        try:
            with open("process_hierarchy_cache.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty dict if file doesn't exist or is invalid
            return {}
    
    def _create_vector_store(self, repo_path: str) -> Optional[FAISS]:
        """
        Create a vector store from repository files
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            FAISS vector store or None if creation failed
        """
        try:
            # Determine files to load (Python files, markdown, etc.)
            loader = DirectoryLoader(
                repo_path,
                glob="**/*.{py,md,txt,json}",
                exclude=["**/venv/**", "**/.git/**", "**/__pycache__/**"],
                loader_cls=TextLoader,
                show_progress=True
            )
            
            documents = loader.load()
            
            if not documents:
                return None
                
            # Split files into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, self.embeddings)
            
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def _extract_agents_from_code(self, repo_path: str) -> List[Dict]:
        """
        Extract agents from code using regex and heuristics
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of dictionaries with agent information
        """
        agents = []
        agent_pattern = re.compile(r"Agent\s*\(\s*(?:role\s*=\s*[\"']([^\"']+)[\"']|[\"']([^\"']+)[\"'])")
        goal_pattern = re.compile(r"goal\s*=\s*[\"']([^\"']+)[\"']")
        backstory_pattern = re.compile(r"backstory\s*=\s*[\"'\"]([^\"'\"]+)[\"'\"]")
        
        # Walk through Python files
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # Find agent definitions
                            for match in agent_pattern.finditer(content):
                                agent_name = match.group(1) or match.group(2)
                                if agent_name:
                                    # Find the goal
                                    goal_match = goal_pattern.search(content[match.start():])
                                    goal = goal_match.group(1) if goal_match else ""
                                    
                                    # Find the backstory
                                    backstory_match = backstory_pattern.search(content[match.start():])
                                    backstory = backstory_match.group(1) if backstory_match else ""
                                    
                                    agents.append({
                                        "name": agent_name,
                                        "goal": goal,
                                        "backstory": backstory,
                                        "file_path": file_path
                                    })
                                    
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        return agents
    
    def _extract_crew_from_code(self, repo_path: str) -> Dict:
        """
        Extract crew from code using regex and heuristics
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary with crew information
        """
        crew_pattern = re.compile(r"Crew\s*\(\s*(?:name\s*=\s*[\"']([^\"']+)[\"']|[\"']([^\"']+)[\"'])")
        agents_pattern = re.compile(r"agents\s*=\s*\[(.*?)\]", re.DOTALL)
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # Find crew definitions
                            crew_match = crew_pattern.search(content)
                            if crew_match:
                                crew_name = crew_match.group(1) or crew_match.group(2)
                                
                                # Find agents list
                                agents_match = agents_pattern.search(content[crew_match.start():])
                                agents_text = agents_match.group(1) if agents_match else ""
                                
                                return {
                                    "name": crew_name,
                                    "file_path": file_path,
                                    "agents_text": agents_text
                                }
                                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        return {}
    
    def _match_processes(self, description: str, repo_analysis: Dict) -> List[str]:
        """
        Match a crew/agent description to process IDs in the knowledge graph
        
        Args:
            description: Description to match
            repo_analysis: Additional repository analysis information
            
        Returns:
            List of matching process IDs
        """
        # In production, this would use GraphRAG to query the knowledge graph
        # For demo purposes, we'll use the LLM directly
        
        try:
            # Prepare system prompt for process matching
            system_prompt = """You are an expert in mapping agent capabilities to business processes.
Given a description of an agent or crew, you need to identify which business processes it can handle.

Below is a list of available business processes in our system:
- RETAIL-CATALOG-001: Catalog Management - End-to-end process for managing the retail product catalog
- RETAIL-CATALOG-001-001: Item Setup - Process for setting up new items in the catalog
- RETAIL-CATALOG-001-002: Item Maintenance - Process for maintaining and updating existing items
- RETAIL-CATALOG-001-003: Catalog Optimization - Process for optimizing the catalog structure
- RETAIL-CATALOG-001-004: Catalog Data Quality - Process for ensuring catalog data quality
- RETAIL-CATALOG-001-005: Catalog Compliance - Process for ensuring catalog compliance with regulations

- RETAIL-COMPLIANCE-001: Trust and Safety Compliance - Ensuring retail operations comply with regulations
- RETAIL-COMPLIANCE-001-001: Product Safety Compliance - Ensuring products meet safety standards
- RETAIL-COMPLIANCE-001-002: Content Moderation - Reviewing and moderating product content
- RETAIL-COMPLIANCE-001-003: Regulatory Compliance - Ensuring compliance with regulations
- RETAIL-COMPLIANCE-001-004: Fraud Prevention - Detecting and preventing fraudulent activities
- RETAIL-COMPLIANCE-001-005: Policy Enforcement - Enforcing company policies

Respond with a JSON array containing ONLY the process IDs that this agent/crew can handle.
Only include process IDs with high confidence of relevance."""

            # Use OpenAI to match
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Description: {description}\n\nCapabilities: {repo_analysis.get('capabilities', [])}"}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            response_json = json.loads(response_text)
            
            # Extract process IDs
            process_ids = response_json.get("process_ids", [])
            if not isinstance(process_ids, list):
                process_ids = []
                
            return process_ids
            
        except Exception as e:
            print(f"Error matching processes: {e}")
            return []
    
    def _analyze_repository_with_llm(self, repo_path: str, vector_store) -> Dict:
        """
        Analyze repository contents using LLM with RAG support
        
        Args:
            repo_path: Path to the repository
            vector_store: FAISS vector store of repository contents
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract basic info with code parsing
            extracted_agents = self._extract_agents_from_code(repo_path)
            extracted_crew = self._extract_crew_from_code(repo_path)
            
            # Prepare repository summary by searching vector store
            repo_summary = []
            
            # Get crew definition context
            if extracted_crew:
                file_path = extracted_crew.get("file_path", "")
                if file_path:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            repo_summary.append(f"FILE: {os.path.basename(file_path)}\n{f.read()[:4000]}")
                    except Exception:
                        pass
            
            # Get README context
            readme_paths = [
                os.path.join(repo_path, "README.md"),
                os.path.join(repo_path, "Readme.md"),
                os.path.join(repo_path, "readme.md")
            ]
            
            for path in readme_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            repo_summary.append(f"README.md:\n{f.read()[:3000]}")
                    except Exception:
                        pass
                    break
            
            # Add agent definitions as context
            for i, agent in enumerate(extracted_agents[:5]):  # Limit to first 5 agents
                file_path = agent.get("file_path", "")
                if file_path:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Find the agent definition
                            agent_name = agent.get("name", "")
                            if agent_name:
                                # Find a chunk of text around the agent definition
                                start = content.find(agent_name)
                                if start != -1:
                                    chunk = content[max(0, start-200):min(len(content), start+1000)]
                                    repo_summary.append(f"AGENT {i+1}: {agent_name}\n{chunk}")
                    except Exception:
                        pass
            
            # Prepare system prompt for repository analysis
            system_prompt = """You are an AI expert in analyzing agent and crew repositories for AI orchestration.
Your task is to analyze code and documentation to extract metadata about agents and crews.

Focus on:
1. Identifying crew name, purpose, and capabilities
2. Identifying individual agents, their roles, and capabilities
3. Mapping these to business process domains
4. Extracting API endpoints if available

Return a JSON object with the following structure:
{
    "crew": {
        "name": "string", // Name of the crew
        "description": "string", // Description of what the crew does
        "version": "string", // Version if available, otherwise "0.1.0"
        "api_endpoint": "string" // API endpoint if available
    },
    "agents": [
        {
            "name": "string", // Name of the agent
            "description": "string", // Description of the agent's purpose
            "type": "llm", // Type: llm, specialized, hybrid, or human_in_loop
            "capabilities": ["string", "string"] // List of capabilities this agent has
        }
    ],
    "capabilities": ["string", "string"], // Overall capabilities of the repository
    "confidence_score": 0.0 // Confidence in the analysis from 0.0 to 1.0
}"""

            # Use OpenAI to analyze
            repo_context = "\n\n".join(repo_summary)
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Repository contents:\n{repo_context}\n\nPlease analyze this repository and extract metadata about the crew and agents."}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            analysis = json.loads(response_text)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing repository with LLM: {e}")
            return {
                "crew": {"name": "", "description": "", "version": "0.1.0"},
                "agents": [],
                "capabilities": [],
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def analyze_repository(self, repo_path: str) -> RepositoryAnalysisResult:
        """
        Analyze a repository to extract crew and agent metadata
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            RepositoryAnalysisResult with the analysis results
        """
        try:
            # Verify repository path
            repo_path = os.path.abspath(repo_path)
            if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                return RepositoryAnalysisResult(
                    errors=[f"Repository path {repo_path} does not exist or is not a directory"]
                )
            
            # Create vector store from repository files
            vector_store = self._create_vector_store(repo_path)
            
            # Analyze repository with LLM
            analysis = self._analyze_repository_with_llm(repo_path, vector_store)
            
            # Extract crew metadata
            crew_data = analysis.get("crew", {})
            crew_name = crew_data.get("name", "")
            crew_description = crew_data.get("description", "")
            
            # Match to process IDs
            process_ids = self._match_processes(crew_description, analysis)
            
            # Extract agent metadata
            agents_data = analysis.get("agents", [])
            agent_metadata_list = []
            
            for i, agent_data in enumerate(agents_data):
                agent_name = agent_data.get("name", "")
                agent_description = agent_data.get("description", "")
                agent_type = agent_data.get("type", "llm")
                agent_capabilities = agent_data.get("capabilities", [])
                
                # Create agent metadata
                agent_metadata = AgentMetadata(
                    agent_id=f"AGENT-{crew_name.replace(' ', '-')}-{i+1}".upper(),
                    name=agent_name,
                    description=agent_description,
                    version="0.1.0",
                    type=AgentType(agent_type),
                    capabilities=agent_capabilities,
                    repository_url=None,
                    repository_path=repo_path,
                )
                
                agent_metadata_list.append(agent_metadata)
            
            # Create crew metadata
            crew_metadata = CrewMetadata(
                crew_id=f"CREW-{crew_name.replace(' ', '-')}".upper(),
                name=crew_name,
                description=crew_description,
                version=crew_data.get("version", "0.1.0"),
                agent_ids=[agent.agent_id for agent in agent_metadata_list],
                repository_url=None,
                repository_path=repo_path,
                process_ids=process_ids,
                api_endpoint=crew_data.get("api_endpoint"),
                api_auth_type=ApiAuth.API_KEY if crew_data.get("api_endpoint") else None,
                api_docs_url=None
            )
            
            # Create result
            result = RepositoryAnalysisResult(
                crew_metadata=crew_metadata,
                agent_metadata=agent_metadata_list,
                process_ids=process_ids,
                capabilities=analysis.get("capabilities", []),
                confidence_score=analysis.get("confidence_score", 0.0)
            )
            
            return result
            
        except Exception as e:
            return RepositoryAnalysisResult(
                errors=[f"Error analyzing repository: {str(e)}"]
            )
    
    def register_crew(self, analysis_result: RepositoryAnalysisResult) -> bool:
        """
        Register a crew and its agents in the AgentKG knowledge graph
        
        Args:
            analysis_result: Results of repository analysis
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            if not analysis_result.crew_metadata:
                print("No crew metadata available")
                return False
                
            # Prepare agent registrations
            agent_registrations = []
            for agent in analysis_result.agent_metadata:
                agent_reg = AgentRegistration(
                    metadata=agent,
                    capabilities_detail=None
                )
                agent_registrations.append(agent_reg)
            
            # Prepare crew registration
            crew_reg = CrewRegistration(
                metadata=analysis_result.crew_metadata,
                agent_registrations=agent_registrations
            )
            
            # Register with the API
            headers = {}
            if self.api_key:
                headers["X-Api-Key"] = self.api_key
                
            response = requests.post(
                f"{self.api_base_url}/crews/register",
                json=crew_reg.dict(),
                headers=headers
            )
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"Crew registration successful: {response_data.get('message')}")
                return True
            else:
                print(f"Crew registration failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error registering crew: {e}")
            return False


def main():
    """Main function to demonstrate the CrewAnalyzerAgent"""
    if len(sys.argv) < 2:
        print("Usage: python crew_analyzer_agent.py <repository_path>")
        return
        
    repo_path = sys.argv[1]
    analyzer = CrewAnalyzerAgent()
    
    print(f"Analyzing repository: {repo_path}")
    analysis = analyzer.analyze_repository(repo_path)
    
    if analysis.errors:
        print("Analysis errors:")
        for error in analysis.errors:
            print(f"- {error}")
        return
        
    print("Analysis results:")
    print(f"Crew: {analysis.crew_metadata.name if analysis.crew_metadata else 'Unknown'}")
    print(f"Agents: {len(analysis.agent_metadata)}")
    print(f"Process IDs: {analysis.process_ids}")
    print(f"Capabilities: {analysis.capabilities}")
    print(f"Confidence score: {analysis.confidence_score}")
    
    if analysis.crew_metadata:
        register = input("Register this crew in AgentKG? (y/n): ")
        if register.lower() == "y":
            success = analyzer.register_crew(analysis)
            if success:
                print("Registration successful!")
            else:
                print("Registration failed.")


if __name__ == "__main__":
    main()