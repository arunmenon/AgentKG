"""
Multimodal Crew Analyzer for AgentKG

This module provides an enhanced, multimodal agent that can analyze crew repositories
using both code analysis and visual understanding. It can:
1. Clone git repositories directly
2. Analyze code structure and patterns
3. Interpret diagrams and screenshots to understand agent relationships
4. Map crews to appropriate processes in the knowledge graph
5. Register discovered crews and agents with the AgentKG registry
"""

import os
import re
import sys
import json
import git
import tempfile
import requests
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import io
import mimetypes
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from pydantic import BaseModel, Field, HttpUrl

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load AgentKG schema definitions
from AgentKG.src.agent_registry_schema import (
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
    diagrams: List[Dict[str, Any]] = Field(default_factory=list, description="Diagrams found in the repository")


class MultimodalCrewAnalyzer:
    """
    An enhanced, multimodal agent that analyzes crew repositories 
    using both code analysis and visual understanding.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the MultimodalCrewAnalyzer
        
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
        # In a real implementation, this would query the knowledge graph API
        try:
            # First try to query the API
            headers = {}
            if self.api_key:
                headers["X-Api-Key"] = self.api_key
                
            response = requests.get(
                f"{self.api_base_url}/processes",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            
            # Fall back to local cache if API call fails
            with open(os.path.join(os.path.dirname(__file__), "process_hierarchy_cache.json"), "r") as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading process hierarchy: {e}")
            # Return a minimal process hierarchy as fallback
            return {
                "RETAIL-CATALOG-001": {
                    "name": "Catalog Management",
                    "description": "End-to-end process for managing the retail product catalog"
                },
                "RETAIL-CATALOG-001-001": {
                    "name": "Item Setup",
                    "description": "Process for setting up new items in the catalog"
                },
                "RETAIL-CATALOG-001-002": {
                    "name": "Item Maintenance",
                    "description": "Process for maintaining and updating existing items"
                },
                "RETAIL-CATALOG-001-003": {
                    "name": "Catalog Optimization",
                    "description": "Process for optimizing the catalog structure"
                },
                "RETAIL-CATALOG-001-004": {
                    "name": "Catalog Data Quality",
                    "description": "Process for ensuring catalog data quality"
                },
                "RETAIL-CATALOG-001-005": {
                    "name": "Catalog Compliance",
                    "description": "Process for ensuring catalog compliance with regulations"
                },
                "RETAIL-COMPLIANCE-001": {
                    "name": "Trust and Safety Compliance",
                    "description": "Ensuring retail operations comply with regulations"
                },
                "RETAIL-COMPLIANCE-001-001": {
                    "name": "Product Safety Compliance",
                    "description": "Ensuring products meet safety standards"
                },
                "RETAIL-COMPLIANCE-001-002": {
                    "name": "Content Moderation",
                    "description": "Reviewing and moderating product content"
                },
                "RETAIL-COMPLIANCE-001-003": {
                    "name": "Regulatory Compliance",
                    "description": "Ensuring compliance with regulations"
                },
                "RETAIL-COMPLIANCE-001-004": {
                    "name": "Fraud Prevention",
                    "description": "Detecting and preventing fraudulent activities"
                },
                "RETAIL-COMPLIANCE-001-005": {
                    "name": "Policy Enforcement",
                    "description": "Enforcing company policies"
                }
            }
    
    def clone_repository(self, repo_url: str) -> str:
        """
        Clone a git repository to a temporary directory
        
        Args:
            repo_url: URL of the git repository to clone
            
        Returns:
            Path to the cloned repository
        """
        # Check if the URL is valid
        try:
            result = urlparse(repo_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError(f"Invalid repository URL: {repo_url}")
        except Exception as e:
            raise ValueError(f"Invalid repository URL: {repo_url}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="agentkg_repo_")
        
        try:
            # Clone the repository
            git.Repo.clone_from(repo_url, temp_dir)
            return temp_dir
        except Exception as e:
            raise Exception(f"Error cloning repository: {e}")
    
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
                glob="**/*.{py,md,txt,json,yaml,yml}",
                exclude=["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/node_modules/**"],
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
        # Pattern for CrewAI agent definition
        agent_pattern = re.compile(r"Agent\s*\(\s*(?:role\s*=\s*[\"']([^\"']+)[\"']|[\"']([^\"']+)[\"'])")
        goal_pattern = re.compile(r"goal\s*=\s*[\"']([^\"']+)[\"']")
        backstory_pattern = re.compile(r"backstory\s*=\s*[\"'\"]([^\"'\"]+)[\"'\"]")
        
        # Alternative patterns for other frameworks (LangChain, etc.)
        langchain_agent_pattern = re.compile(r"initialize_agent\s*\(.*,\s*[\"']([^\"']+)[\"']")
        
        # Walk through Python files
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".js", ".ts")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # Find CrewAI agent definitions
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
                                        "file_path": file_path,
                                        "framework": "crewai"
                                    })
                            
                            # Find LangChain agent definitions
                            for match in langchain_agent_pattern.finditer(content):
                                agent_name = match.group(1)
                                if agent_name:
                                    agents.append({
                                        "name": agent_name,
                                        "goal": "",
                                        "backstory": "",
                                        "file_path": file_path,
                                        "framework": "langchain"
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
        # CrewAI patterns
        crew_pattern = re.compile(r"Crew\s*\(\s*(?:name\s*=\s*[\"']([^\"']+)[\"']|[\"']([^\"']+)[\"'])")
        agents_pattern = re.compile(r"agents\s*=\s*\[(.*?)\]", re.DOTALL)
        
        # Alternative patterns for other frameworks
        langchain_sequential_pattern = re.compile(r"SequentialChain\s*\(")
        langchain_router_pattern = re.compile(r"RouterChain\s*\(")
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".js", ".ts")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                            # Find CrewAI crew definitions
                            crew_match = crew_pattern.search(content)
                            if crew_match:
                                crew_name = crew_match.group(1) or crew_match.group(2) or "Unnamed Crew"
                                
                                # Find agents list
                                agents_match = agents_pattern.search(content[crew_match.start():])
                                agents_text = agents_match.group(1) if agents_match else ""
                                
                                return {
                                    "name": crew_name,
                                    "file_path": file_path,
                                    "agents_text": agents_text,
                                    "framework": "crewai"
                                }
                            
                            # Find LangChain sequences (alternative to crews)
                            if langchain_sequential_pattern.search(content) or langchain_router_pattern.search(content):
                                return {
                                    "name": "LangChain Agent Sequence",
                                    "file_path": file_path,
                                    "agents_text": "",
                                    "framework": "langchain"
                                }
                                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        
        return {}
    
    def _find_diagrams(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Find diagram files (images) in the repository
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of dictionaries with diagram information
        """
        diagrams = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".drawio")):
                    file_path = os.path.join(root, file)
                    try:
                        # Check if it's a diagram (heuristic: filename contains 'diagram', 'flow', 'architecture')
                        filename_lower = file.lower()
                        is_likely_diagram = any(keyword in filename_lower for keyword in
                                               ["diagram", "flow", "architecture", "model", "structure", "crew", "agent"])
                        
                        # Analyze the image
                        relative_path = os.path.relpath(file_path, repo_path)
                        mime_type, _ = mimetypes.guess_type(file_path)
                        
                        if mime_type and mime_type.startswith("image/"):
                            diagrams.append({
                                "path": relative_path,
                                "is_likely_diagram": is_likely_diagram,
                                "mime_type": mime_type,
                                "file_path": file_path
                            })
                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")
        
        return diagrams
    
    def _analyze_diagram(self, diagram_path: str) -> Dict[str, Any]:
        """
        Analyze a diagram using multimodal capabilities
        
        Args:
            diagram_path: Path to the diagram file
            
        Returns:
            Dictionary with diagram analysis results
        """
        try:
            # Read the image
            with open(diagram_path, "rb") as f:
                image_data = f.read()
                
            # Convert to base64 for API
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Use OpenAI's vision capabilities to analyze the diagram
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in analyzing software architecture diagrams, particularly those related to agent-based systems and workflows. 
                        Examine this diagram and extract:
                        1. Overall system architecture
                        2. Individual agents or components and their roles
                        3. Relationships between components
                        4. Data flow
                        5. Any text explaining the purpose or capabilities
                        
                        Focus specifically on identifying:
                        - Agent names and purposes
                        - Workflow sequences
                        - Integration points
                        - Technologies mentioned
                        
                        Respond with a JSON object containing:
                        {
                            "diagram_type": "flow/architecture/component/etc",
                            "agent_components": [{"name": "...", "role": "...", "capabilities": [...]}],
                            "relationships": [{"from": "...", "to": "...", "type": "..."}],
                            "overall_purpose": "...",
                            "technologies": [...],
                            "confidence": 0.0-1.0 (how confident you are in this analysis)
                        }"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this diagram and extract information about agents, workflows, and system architecture:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            analysis_json = json.loads(response.choices[0].message.content)
            analysis_json["file_path"] = diagram_path
            
            return analysis_json
            
        except Exception as e:
            print(f"Error analyzing diagram: {e}")
            return {
                "error": str(e),
                "file_path": diagram_path,
                "confidence": 0.0
            }
    
    def _match_processes(self, description: str, repo_analysis: Dict) -> List[str]:
        """
        Match a crew/agent description to process IDs in the knowledge graph
        
        Args:
            description: Description to match
            repo_analysis: Additional repository analysis information
            
        Returns:
            List of matching process IDs
        """
        try:
            # Prepare system prompt for process matching using available processes
            process_descriptions = []
            for process_id, process_info in self.processes.items():
                process_desc = f"- {process_id}: {process_info.get('name', '')} - {process_info.get('description', '')}"
                process_descriptions.append(process_desc)
            
            process_list = "\n".join(process_descriptions)
            
            system_prompt = f"""You are an expert in mapping agent capabilities to business processes.
Given a description of an agent or crew, you need to identify which business processes it can handle.

Below is a list of available business processes in our system:
{process_list}

Respond with a JSON object containing only the process IDs that this agent/crew can handle.
Only include process IDs with high confidence of relevance.

Response format:
{{
    "process_ids": ["PROCESS-ID-1", "PROCESS-ID-2"]
}}"""

            # Use OpenAI to match
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Description: {description}\n\nCapabilities: {repo_analysis.get('capabilities', [])}\n\nTechnologies: {repo_analysis.get('technologies', [])}"}
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
    
    def _analyze_repository_with_llm(self, repo_path: str, vector_store, diagrams: List[Dict]) -> Dict:
        """
        Analyze repository contents using LLM with RAG support
        
        Args:
            repo_path: Path to the repository
            vector_store: FAISS vector store of repository contents
            diagrams: List of diagrams found in the repository
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract basic info with code parsing
            extracted_agents = self._extract_agents_from_code(repo_path)
            extracted_crew = self._extract_crew_from_code(repo_path)
            
            # Analyze diagrams if available
            diagram_analyses = []
            for diagram in diagrams:
                if diagram.get("is_likely_diagram", False):
                    analysis = self._analyze_diagram(diagram.get("file_path"))
                    if analysis:
                        diagram_analyses.append(analysis)
            
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
            
            # Add diagram analysis summaries
            for i, analysis in enumerate(diagram_analyses):
                summary = f"DIAGRAM {i+1}:\n"
                summary += f"Type: {analysis.get('diagram_type', 'Unknown')}\n"
                summary += f"Purpose: {analysis.get('overall_purpose', 'Unknown')}\n"
                
                # Add agent components
                agents = analysis.get("agent_components", [])
                if agents:
                    summary += "Components:\n"
                    for agent in agents:
                        summary += f"- {agent.get('name', 'Unnamed')}: {agent.get('role', 'Unknown')}\n"
                
                repo_summary.append(summary)
            
            # Prepare system prompt for repository analysis
            system_prompt = """You are an AI expert in analyzing agent and crew repositories for AI orchestration.
Your task is to analyze code and documentation to extract metadata about agents and crews.

Focus on:
1. Identifying crew name, purpose, and capabilities
2. Identifying individual agents, their roles, and capabilities
3. Understanding the orchestration pattern and workflow
4. Identifying technologies and frameworks used
5. Extracting API endpoints if available

Return a JSON object with the following structure:
{
    "crew": {
        "name": "string", // Name of the crew
        "description": "string", // Description of what the crew does
        "version": "string", // Version if available, otherwise "0.1.0"
        "api_endpoint": "string", // API endpoint if available
        "orchestration_pattern": "string" // sequential, parallel, or event-driven
    },
    "agents": [
        {
            "name": "string", // Name of the agent
            "description": "string", // Description of the agent's purpose
            "type": "llm", // Type: llm, specialized, hybrid, or human_in_loop
            "capabilities": ["string"], // List of capabilities this agent has
            "role": "string" // Role in the overall workflow
        }
    ],
    "capabilities": ["string"], // Overall capabilities of the repository
    "technologies": ["string"], // Technologies and frameworks used
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
            
            # Add diagram analyses to the result
            analysis["diagrams"] = diagram_analyses
            analysis["framework"] = extracted_crew.get("framework", "unknown")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing repository with LLM: {e}")
            return {
                "crew": {"name": "", "description": "", "version": "0.1.0"},
                "agents": [],
                "capabilities": [],
                "technologies": [],
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def analyze_repository_path(self, repo_path: str) -> RepositoryAnalysisResult:
        """
        Analyze a local repository to extract crew and agent metadata
        
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
            
            # Find diagrams in the repository
            diagrams = self._find_diagrams(repo_path)
            
            # Create vector store from repository files
            vector_store = self._create_vector_store(repo_path)
            
            # Analyze repository with LLM and diagrams
            analysis = self._analyze_repository_with_llm(repo_path, vector_store, diagrams)
            
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
                agent_type_str = agent_data.get("type", "llm")
                agent_capabilities = agent_data.get("capabilities", [])
                
                # Map string type to enum
                if agent_type_str == "llm":
                    agent_type = AgentType.LLM
                elif agent_type_str == "human_in_loop":
                    agent_type = AgentType.HUMAN_IN_LOOP
                elif agent_type_str == "hybrid":
                    agent_type = AgentType.HYBRID
                else:
                    agent_type = AgentType.SPECIALIZED
                
                # Create agent metadata
                agent_metadata = AgentMetadata(
                    agent_id=f"AGENT-{crew_name.replace(' ', '-')}-{i+1}".upper() if crew_name else f"AGENT-DISCOVERED-{i+1}",
                    name=agent_name,
                    description=agent_description,
                    version="0.1.0",
                    type=agent_type,
                    capabilities=agent_capabilities,
                    repository_url=None,
                    repository_path=repo_path,
                )
                
                agent_metadata_list.append(agent_metadata)
            
            # Create crew metadata
            crew_id = f"CREW-{crew_name.replace(' ', '-')}".upper() if crew_name else "CREW-DISCOVERED"
            
            crew_metadata = CrewMetadata(
                crew_id=crew_id,
                name=crew_name or "Discovered Crew",
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
                confidence_score=analysis.get("confidence_score", 0.0),
                diagrams=[{
                    "path": diagram.get("path", ""),
                    "analysis": analysis.get("diagrams", [])[i] if i < len(analysis.get("diagrams", [])) else {}
                } for i, diagram in enumerate(diagrams) if diagram.get("is_likely_diagram", False)]
            )
            
            return result
            
        except Exception as e:
            return RepositoryAnalysisResult(
                errors=[f"Error analyzing repository: {str(e)}"]
            )
    
    def analyze_repository_url(self, repo_url: str) -> RepositoryAnalysisResult:
        """
        Analyze a repository from a URL to extract crew and agent metadata
        
        Args:
            repo_url: URL of the git repository to analyze
            
        Returns:
            RepositoryAnalysisResult with the analysis results
        """
        try:
            # Clone the repository
            repo_path = self.clone_repository(repo_url)
            
            # Analyze the cloned repository
            result = self.analyze_repository_path(repo_path)
            
            # Update repository URL
            if result.crew_metadata:
                result.crew_metadata.repository_url = repo_url
                result.crew_metadata.repository_path = None
            
            for agent in result.agent_metadata:
                agent.repository_url = repo_url
                agent.repository_path = None
            
            return result
            
        except Exception as e:
            return RepositoryAnalysisResult(
                errors=[f"Error analyzing repository URL: {str(e)}"]
            )
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a diagram or screenshot directly
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            return self._analyze_diagram(image_path)
        except Exception as e:
            return {
                "error": str(e),
                "file_path": image_path,
                "confidence": 0.0
            }
    
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
            
            # Convert to dict for JSON serialization
            reg_data = crew_reg.model_dump()
            
            # Fix URL serialization
            def fix_urls(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key.endswith('_url') and value is not None:
                            obj[key] = str(value)
                        elif isinstance(value, (dict, list)):
                            fix_urls(value)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            fix_urls(item)
                return obj
            
            reg_data = fix_urls(reg_data)
                
            response = requests.post(
                f"{self.api_base_url}/crews/register",
                json=reg_data,
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
    """Main function to demonstrate the MultimodalCrewAnalyzer"""
    if len(sys.argv) < 2:
        print("Usage: python multimodal_crew_analyzer.py <repository_path_or_url> [--register]")
        return
        
    source = sys.argv[1]
    should_register = "--register" in sys.argv
    
    analyzer = MultimodalCrewAnalyzer()
    
    print(f"Analyzing source: {source}")
    
    # Determine if the source is a URL or a local path
    if source.startswith(("http://", "https://", "git://")):
        analysis = analyzer.analyze_repository_url(source)
    else:
        analysis = analyzer.analyze_repository_path(source)
    
    if analysis.errors:
        print("Analysis errors:")
        for error in analysis.errors:
            print(f"- {error}")
        return
        
    print("\n=== Analysis Results ===")
    print(f"Crew: {analysis.crew_metadata.name if analysis.crew_metadata else 'Unknown'}")
    print(f"Description: {analysis.crew_metadata.description if analysis.crew_metadata else 'Unknown'}")
    print(f"Agents: {len(analysis.agent_metadata)}")
    print(f"Process IDs: {analysis.process_ids}")
    print(f"Capabilities: {analysis.capabilities}")
    print(f"Confidence score: {analysis.confidence_score}")
    
    print("\nAgents:")
    for agent in analysis.agent_metadata:
        print(f"  - {agent.name}: {agent.description[:100]}...")
        print(f"    Capabilities: {agent.capabilities}")
    
    if analysis.diagrams:
        print("\nDiagrams found:")
        for diagram in analysis.diagrams:
            print(f"  - {diagram.get('path')}")
            analysis_data = diagram.get('analysis', {})
            if 'diagram_type' in analysis_data:
                print(f"    Type: {analysis_data.get('diagram_type')}")
                print(f"    Purpose: {analysis_data.get('overall_purpose', 'Unknown')[:100]}...")
    
    if should_register and analysis.crew_metadata:
        print("\nRegistering with AgentKG...")
        success = analyzer.register_crew(analysis)
        if success:
            print("Registration successful!")
        else:
            print("Registration failed.")


if __name__ == "__main__":
    main()