"""
Compliance Crew Connector Module for AgentKG

This module connects CrewAI crews to the AgentKG knowledge graph,
allowing for agent orchestration based on the graph structure.
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
from crewai import Crew, Agent
import sys
sys.path.append('/Users/arunmenon/projects/AgentKG')
from compliance_crews import ComplianceCrews

# Load environment variables
load_dotenv()


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


class ComplianceCrewConnector:
    """
    Class for connecting CrewAI crews to the AgentKG knowledge graph.
    """
    
    def __init__(self, llm=None):
        """Initialize the crew connector with an optional language model"""
        self.connector = Neo4jConnector()
        self.compliance_crews = ComplianceCrews(llm=llm)
        self.crews = {}  # Dictionary to store instantiated crews
        self.agents = {}  # Dictionary to store all agents
        
    def close(self):
        """Close Neo4j connection"""
        self.connector.close()
        
    def register_crew_in_graph(self, crew_id: str, crew_name: str, process_id: str, 
                              description: str, agent_ids: List[str]):
        """
        Register a crew in the knowledge graph with connection to a process
        
        Args:
            crew_id: Unique identifier for the crew
            crew_name: Name of the crew
            process_id: Process ID this crew is associated with
            description: Description of the crew
            agent_ids: List of agent IDs that are part of this crew
        """
        # Create crew node
        query = """
        MERGE (c:Crew {crewId: $crewId})
        ON CREATE SET 
            c.name = $name,
            c.description = $description,
            c.createdAt = datetime()
        ON MATCH SET
            c.name = $name,
            c.description = $description,
            c.updatedAt = datetime()
        """
        self.connector.execute_query(query, {
            "crewId": crew_id,
            "name": crew_name,
            "description": description
        })
        
        # Connect crew to process
        query = """
        MATCH (c:Crew {crewId: $crewId})
        MATCH (p:Process {processId: $processId})
        MERGE (c)-[:HANDLES]->(p)
        """
        self.connector.execute_query(query, {
            "crewId": crew_id,
            "processId": process_id
        })
        
        # Connect agents to crew
        for agent_id in agent_ids:
            query = """
            MATCH (c:Crew {crewId: $crewId})
            MATCH (a:Agent {agentId: $agentId})
            MERGE (a)-[:MEMBER_OF]->(c)
            """
            self.connector.execute_query(query, {
                "crewId": crew_id,
                "agentId": agent_id
            })
        
        print(f"Crew '{crew_name}' (ID: {crew_id}) registered for process {process_id}")
    
    def register_agent_in_graph(self, agent_id: str, agent_name: str, role: str, 
                              goal: str, capabilities: List[str]):
        """
        Register an agent in the knowledge graph
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Name of the agent
            role: Role of the agent
            goal: Goal of the agent
            capabilities: List of capabilities of this agent
        """
        # Create agent node
        query = """
        MERGE (a:Agent {agentId: $agentId})
        ON CREATE SET 
            a.name = $name,
            a.role = $role,
            a.goal = $goal,
            a.capabilities = $capabilities,
            a.createdAt = datetime()
        ON MATCH SET
            a.name = $name,
            a.role = $role,
            a.goal = $goal,
            a.capabilities = $capabilities,
            a.updatedAt = datetime()
        """
        self.connector.execute_query(query, {
            "agentId": agent_id,
            "name": agent_name,
            "role": role,
            "goal": goal,
            "capabilities": capabilities
        })
        
        print(f"Agent '{agent_name}' (ID: {agent_id}) registered with role: {role}")
    
    def register_content_moderation_crew(self):
        """
        Register the content moderation crew and its agents in the knowledge graph
        """
        # Register agents
        agents_data = [
            {
                "id": "AGENT-CM-001",
                "name": "Product Description Reviewer",
                "role": "Product Description Reviewer",
                "goal": "Ensure product descriptions comply with all company policies and regulations",
                "capabilities": [
                    "Identifying misleading claims",
                    "Detecting inappropriate content",
                    "Verifying regulatory compliance",
                    "Checking for policy violations"
                ]
            },
            {
                "id": "AGENT-CM-002",
                "name": "Product Image Moderator",
                "role": "Product Image Moderator",
                "goal": "Ensure all product images comply with company guidelines and are appropriate",
                "capabilities": [
                    "Identifying inappropriate imagery",
                    "Detecting misleading visuals",
                    "Finding copyright violations",
                    "Verifying image quality standards"
                ]
            },
            {
                "id": "AGENT-CM-003",
                "name": "Customer Review Moderator",
                "role": "Customer Review Moderator",
                "goal": "Ensure customer reviews and Q&A content follows community guidelines",
                "capabilities": [
                    "Detecting inappropriate language",
                    "Identifying spam or fake reviews",
                    "Removing personally identifiable information",
                    "Preserving authentic customer feedback"
                ]
            },
            {
                "id": "AGENT-CM-004",
                "name": "Prohibited Content Detector",
                "role": "Prohibited Content Detector",
                "goal": "Identify and flag prohibited or illegal content across all channels",
                "capabilities": [
                    "Detecting illegal product listings",
                    "Identifying policy violations",
                    "Finding attempts to circumvent detection",
                    "Regulatory compliance verification"
                ]
            },
            {
                "id": "AGENT-CM-005",
                "name": "Moderation Appeals Processor",
                "role": "Moderation Appeals Processor",
                "goal": "Fairly review appeals against moderation decisions",
                "capabilities": [
                    "Evaluating moderation decisions",
                    "Analyzing appeal justifications",
                    "Researching policies and precedents",
                    "Making fair determinations"
                ]
            }
        ]
        
        # Register each agent
        for agent in agents_data:
            self.register_agent_in_graph(
                agent_id=agent["id"],
                agent_name=agent["name"],
                role=agent["role"],
                goal=agent["goal"],
                capabilities=agent["capabilities"]
            )
        
        # Register the crew
        self.register_crew_in_graph(
            crew_id="CREW-CM-001",
            crew_name="Content Moderation Crew",
            process_id="RETAIL-COMPLIANCE-001-002",  # Content Moderation process ID
            description="A specialized crew for moderating content across product descriptions, images, and user-generated content",
            agent_ids=[agent["id"] for agent in agents_data]
        )
        
        # Store the CrewAI crew object for later use
        content_crew = self.compliance_crews.create_content_moderation_crew()
        self.crews["CREW-CM-001"] = content_crew
        
        return content_crew
    
    def register_fraud_prevention_crew(self):
        """
        Register the fraud prevention crew and its agents in the knowledge graph
        """
        # Register agents
        agents_data = [
            {
                "id": "AGENT-FP-001",
                "name": "Fraud Detection Specialist",
                "role": "Fraud Detection Specialist",
                "goal": "Identify patterns and indicators of potential fraud in retail operations",
                "capabilities": [
                    "Pattern recognition",
                    "Anomaly detection",
                    "Transaction analysis",
                    "Behavioral analysis"
                ]
            },
            {
                "id": "AGENT-FP-002",
                "name": "Fraud Investigator",
                "role": "Fraud Investigator",
                "goal": "Thoroughly investigate potential fraud cases to confirm or dismiss concerns",
                "capabilities": [
                    "Forensic analysis",
                    "Evidence trail tracking",
                    "Documentation",
                    "Case evaluation"
                ]
            },
            {
                "id": "AGENT-FP-003",
                "name": "Fraud Mitigation Expert",
                "role": "Fraud Mitigation Expert",
                "goal": "Implement measures to prevent and reduce fraud incidents",
                "capabilities": [
                    "Preventative controls design",
                    "Security protocol development",
                    "Risk assessment",
                    "Customer experience balancing"
                ]
            },
            {
                "id": "AGENT-FP-004",
                "name": "Fraud Analytics and Reporting Specialist",
                "role": "Fraud Analytics and Reporting Specialist",
                "goal": "Analyze fraud trends and create comprehensive reports for stakeholders",
                "capabilities": [
                    "Data analysis",
                    "Trend identification",
                    "Impact assessment",
                    "Stakeholder communication"
                ]
            }
        ]
        
        # Register each agent
        for agent in agents_data:
            self.register_agent_in_graph(
                agent_id=agent["id"],
                agent_name=agent["name"],
                role=agent["role"],
                goal=agent["goal"],
                capabilities=agent["capabilities"]
            )
        
        # Register the crew
        self.register_crew_in_graph(
            crew_id="CREW-FP-001",
            crew_name="Fraud Prevention Crew",
            process_id="RETAIL-COMPLIANCE-001-004",  # Fraud Prevention process ID
            description="A specialized crew for detecting, investigating, and mitigating fraud across retail operations",
            agent_ids=[agent["id"] for agent in agents_data]
        )
        
        # Store the CrewAI crew object for later use
        fraud_crew = self.compliance_crews.create_fraud_prevention_crew()
        self.crews["CREW-FP-001"] = fraud_crew
        
        return fraud_crew
    
    def register_product_safety_crew(self):
        """
        Register the product safety compliance crew and its agents in the knowledge graph
        """
        # Register agents
        agents_data = [
            {
                "id": "AGENT-PS-001",
                "name": "Safety Certification Verifier",
                "role": "Safety Certification Verifier",
                "goal": "Verify product safety certifications and documentation for compliance",
                "capabilities": [
                    "Certification validation",
                    "Documentation verification",
                    "Regulatory knowledge",
                    "Standards assessment"
                ]
            },
            {
                "id": "AGENT-PS-002",
                "name": "Product Recall Manager",
                "role": "Product Recall Manager",
                "goal": "Effectively manage product recalls and safety alerts",
                "capabilities": [
                    "Recall planning",
                    "Stakeholder communication",
                    "Process coordination",
                    "Impact mitigation"
                ]
            },
            {
                "id": "AGENT-PS-003",
                "name": "Product Safety Testing Specialist",
                "role": "Product Safety Testing Specialist",
                "goal": "Ensure products are tested thoroughly for safety compliance",
                "capabilities": [
                    "Testing protocol development",
                    "Standards implementation",
                    "Test results interpretation",
                    "Testing requirements tracking"
                ]
            },
            {
                "id": "AGENT-PS-004",
                "name": "Safety Compliance Reporting Specialist",
                "role": "Safety Compliance Reporting Specialist",
                "goal": "Create comprehensive safety compliance reports for stakeholders",
                "capabilities": [
                    "Compliance documentation",
                    "Regulatory reporting",
                    "Metrics tracking",
                    "Stakeholder communication"
                ]
            }
        ]
        
        # Register each agent
        for agent in agents_data:
            self.register_agent_in_graph(
                agent_id=agent["id"],
                agent_name=agent["name"],
                role=agent["role"],
                goal=agent["goal"],
                capabilities=agent["capabilities"]
            )
        
        # Register the crew
        self.register_crew_in_graph(
            crew_id="CREW-PS-001",
            crew_name="Product Safety Compliance Crew",
            process_id="RETAIL-COMPLIANCE-001-001",  # Product Safety Compliance process ID
            description="A specialized crew for ensuring products meet safety standards and managing product safety compliance",
            agent_ids=[agent["id"] for agent in agents_data]
        )
        
        # Store the CrewAI crew object for later use
        safety_crew = self.compliance_crews.create_product_safety_compliance_crew()
        self.crews["CREW-PS-001"] = safety_crew
        
        return safety_crew
    
    def register_all_compliance_crews(self):
        """
        Register all compliance crews in the knowledge graph
        """
        self.register_content_moderation_crew()
        self.register_fraud_prevention_crew()
        self.register_product_safety_crew()
        
        print("All compliance crews registered successfully!")
    
    def find_crew_for_process(self, process_id: str) -> Dict:
        """
        Find a crew that can handle a specific process
        
        Args:
            process_id: The process ID to find a crew for
            
        Returns:
            Dictionary with crew information or None if not found
        """
        query = """
        MATCH (c:Crew)-[:HANDLES]->(p:Process {processId: $processId})
        OPTIONAL MATCH (a:Agent)-[:MEMBER_OF]->(c)
        RETURN 
            c.crewId as crewId, 
            c.name as crewName, 
            c.description as crewDescription,
            collect({
                agentId: a.agentId,
                name: a.name,
                role: a.role,
                goal: a.goal,
                capabilities: a.capabilities
            }) as agents
        """
        
        results = self.connector.execute_query(query, {"processId": process_id})
        
        if not results:
            return None
            
        crew_info = results[0]
        return {
            "crew_id": crew_info["crewId"],
            "name": crew_info["crewName"],
            "description": crew_info["crewDescription"],
            "process_id": process_id,
            "agents": crew_info["agents"],
            "crew_object": self.crews.get(crew_info["crewId"])
        }
    
    def find_agent_for_capability(self, capability: str) -> List[Dict]:
        """
        Find agents with a specific capability
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of agents with the specified capability
        """
        query = """
        MATCH (a:Agent)
        WHERE $capability IN a.capabilities
        OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
        RETURN 
            a.agentId as agentId,
            a.name as name,
            a.role as role,
            a.goal as goal,
            a.capabilities as capabilities,
            collect(c.crewId) as crews
        """
        
        results = self.connector.execute_query(query, {"capability": capability})
        
        return [{
            "agent_id": result["agentId"],
            "name": result["name"],
            "role": result["role"],
            "goal": result["goal"],
            "capabilities": result["capabilities"],
            "crews": result["crews"]
        } for result in results]
    
    def execute_task_with_crew(self, process_id: str, task_description: str, task_inputs: Dict = None) -> Dict:
        """
        Execute a task using the appropriate crew for a process
        
        Args:
            process_id: Process ID for which to find a crew
            task_description: Description of the task to execute
            task_inputs: Optional inputs for the task
            
        Returns:
            Dictionary with task results or error information
        """
        try:
            # Find the crew for the process
            crew_info = self.find_crew_for_process(process_id)
            
            if not crew_info or not crew_info.get("crew_object"):
                return {
                    "success": False,
                    "error": f"No suitable crew found for process {process_id}",
                    "process_id": process_id
                }
            
            # Get the CrewAI crew object
            crew = crew_info["crew_object"]
            
            # Execute the crew with the task description as input
            result = crew.kickoff(inputs={"task_description": task_description, **(task_inputs or {})})
            
            return {
                "success": True,
                "result": result,
                "crew_id": crew_info["crew_id"],
                "process_id": process_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "process_id": process_id
            }


# Example usage
if __name__ == "__main__":
    # This would typically be run with a language model imported
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o")
    
    # For demonstration, we'll use None which will prevent actual execution
    connector = ComplianceCrewConnector(llm=None)
    
    # Register all crews in the graph
    connector.register_all_compliance_crews()
    
    # Example: Find a crew for a specific process
    content_moderation_crew = connector.find_crew_for_process("RETAIL-COMPLIANCE-001-002")
    print(f"Found crew: {content_moderation_crew['name']} for content moderation")
    
    # Example: Find agents with specific capabilities
    fraud_detection_agents = connector.find_agent_for_capability("Forensic analysis")
    print(f"Found {len(fraud_detection_agents)} agents with forensic analysis capability")
    
    # Close the connection
    connector.close()