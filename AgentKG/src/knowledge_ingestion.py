import os
import json
from .neo4j_connector import Neo4jConnector
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the Pydantic models for structured LLM output
class Process(BaseModel):
    name: str = Field(description="Name of the process")
    description: Optional[str] = Field(None, description="Description of the process")
    processId: str = Field(description="Unique identifier for the process")
    status: Optional[str] = Field(None, description="Status of the process, e.g., 'Active', 'Deprecated'")
    parent_process: Optional[str] = Field(None, description="Name of the parent process if this is a sub-process")
    dependencies: Optional[List[str]] = Field([], description="Names of processes this process depends on")
    domain: Optional[str] = Field(None, description="Domain this process belongs to")

class Agent(BaseModel):
    name: str = Field(description="Name of the agent")
    agentId: str = Field(description="Unique identifier for the agent")
    title: Optional[str] = Field(None, description="Job title of the agent")
    business_functions: List[str] = Field([], description="Business functions the agent performs")
    tech_expertise: List[str] = Field([], description="Technical expertise areas of the agent")
    processes_supported: List[str] = Field([], description="Names of processes this agent supports")
    crews: List[str] = Field([], description="Names of crews this agent is a member of")
    performance_metrics: Optional[dict] = Field({}, description="Performance metrics for the agent")

class Crew(BaseModel):
    name: str = Field(description="Name of the crew")
    crewId: str = Field(description="Unique identifier for the crew")
    parent_crew: Optional[str] = Field(None, description="Name of the parent crew if this is a sub-crew")
    processes_supported: List[str] = Field([], description="Names of processes this crew supports")
    performance_metrics: Optional[dict] = Field({}, description="Performance metrics for the crew")

class Task(BaseModel):
    taskId: str = Field(description="Unique identifier for the task")
    title: str = Field(description="Short description of the task")
    status: str = Field(description="Status of the task (e.g., Pending, In-Progress, Completed)")
    priority: Optional[str] = Field(None, description="Priority of the task (e.g., High, Medium, Low)")
    process: str = Field(description="Name of the process this task is part of")
    assigned_agents: List[str] = Field([], description="Names of agents assigned to this task")

class PerformanceRecord(BaseModel):
    recordId: str = Field(description="Unique identifier for the performance record")
    date: str = Field(description="Date of the performance record (YYYY-MM-DD)")
    entity_id: str = Field(description="ID of the agent or crew this record belongs to")
    entity_type: str = Field(description="Type of entity (Agent or Crew)")
    metrics: dict = Field({}, description="Performance metrics (e.g., tasksCompleted, successRate)")

class KnowledgeGraph(BaseModel):
    processes: List[Process] = Field([], description="Business processes in the knowledge graph")
    agents: List[Agent] = Field([], description="Agents in the knowledge graph")
    crews: List[Crew] = Field([], description="Crews in the knowledge graph")
    tasks: List[Task] = Field([], description="Tasks in the knowledge graph")
    performance_records: List[PerformanceRecord] = Field([], description="Performance records in the knowledge graph")
    business_functions: List[str] = Field([], description="Business functions taxonomy")
    tech_expertise: List[str] = Field([], description="Technical expertise taxonomy")
    domains: List[str] = Field([], description="Domains in the organization")


class KnowledgeIngestion:
    """
    Class responsible for ingesting knowledge into the Neo4j graph
    using LLMs to extract structured information
    """
    
    def __init__(self):
        """Initialize the Neo4j connector and LLM"""
        self.connector = Neo4jConnector()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
    
    def generate_knowledge_graph(self, domain_description):
        """
        Generate a knowledge graph structure from domain description
        
        Args:
            domain_description (str): Text describing the domain knowledge
            
        Returns:
            KnowledgeGraph: Structured knowledge graph data
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert knowledge graph builder for business processes and agent orchestration.
            Extract structured information about business processes, agents, crews, and their relationships.
            Focus on identifying the hierarchical structures, dependencies, and classifications.
            
            The output should be a JSON object matching this Pydantic model:
            ```
            {parser_format}
            ```
            
            Make sure to:
            1. Identify main processes and their sub-processes
            2. Capture process dependencies
            3. Identify agents and their skills
            4. Identify crews and their responsibilities
            5. Classify elements into the proper taxonomies
            
            Assign unique IDs to each entity (you can make these up based on naming conventions).
            """),
            ("human", "{domain_description}")
        ])
        
        # Set format instructions
        prompt = prompt.partial(parser_format=self.parser.get_format_instructions())
        
        # Generate response
        response = self.llm.invoke(prompt.format(domain_description=domain_description))
        
        # Parse the structured output
        try:
            knowledge_graph = self.parser.parse(response.content)
            return knowledge_graph
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            # Fallback: try to extract JSON directly
            try:
                json_str = response.content
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                return KnowledgeGraph(**data)
            except Exception as e2:
                print(f"Fallback parsing also failed: {e2}")
                print(f"Raw response: {response.content}")
                return None
    
    def ingest_knowledge_graph(self, knowledge_graph):
        """
        Ingest the knowledge graph data into Neo4j
        
        Args:
            knowledge_graph (KnowledgeGraph): Structured knowledge graph data
        """
        # Create domains
        for domain in knowledge_graph.domains:
            query = """
            MERGE (d:Domain {name: $name})
            RETURN d
            """
            self.connector.execute_query(query, {"name": domain})
        
        # Create business functions
        for function in knowledge_graph.business_functions:
            query = """
            MERGE (bf:BusinessFunction {name: $name})
            RETURN bf
            """
            self.connector.execute_query(query, {"name": function})
        
        # Create technical expertise
        for expertise in knowledge_graph.tech_expertise:
            query = """
            MERGE (te:TechExpertise {name: $name})
            RETURN te
            """
            self.connector.execute_query(query, {"name": expertise})
        
        # Create processes
        for process in knowledge_graph.processes:
            query = """
            MERGE (p:Process {processId: $processId})
            ON CREATE SET 
                p.name = $name,
                p.description = $description,
                p.status = $status
            ON MATCH SET
                p.name = $name,
                p.description = $description,
                p.status = $status
            RETURN p
            """
            self.connector.execute_query(query, {
                "processId": process.processId,
                "name": process.name,
                "description": process.description,
                "status": process.status
            })
            
            # Link to domain if specified
            if process.domain:
                query = """
                MATCH (p:Process {processId: $processId})
                MATCH (d:Domain {name: $domain})
                MERGE (p)-[:IN_DOMAIN]->(d)
                """
                self.connector.execute_query(query, {
                    "processId": process.processId,
                    "domain": process.domain
                })
        
        # Create process hierarchies and dependencies
        for process in knowledge_graph.processes:
            # Parent process relationship
            if process.parent_process:
                query = """
                MATCH (child:Process {processId: $childId})
                MATCH (parent:Process {name: $parentName})
                MERGE (child)-[:PART_OF]->(parent)
                """
                self.connector.execute_query(query, {
                    "childId": process.processId,
                    "parentName": process.parent_process
                })
            
            # Process dependencies
            for dependency in process.dependencies:
                query = """
                MATCH (proc:Process {processId: $processId})
                MATCH (dep:Process {name: $depName})
                MERGE (proc)-[:DEPENDS_ON]->(dep)
                """
                self.connector.execute_query(query, {
                    "processId": process.processId,
                    "depName": dependency
                })
        
        # Create crews
        for crew in knowledge_graph.crews:
            query = """
            MERGE (c:Crew {crewId: $crewId})
            ON CREATE SET 
                c.name = $name,
                c.performance_metrics = $metrics
            ON MATCH SET
                c.name = $name,
                c.performance_metrics = $metrics
            RETURN c
            """
            self.connector.execute_query(query, {
                "crewId": crew.crewId,
                "name": crew.name,
                "metrics": json.dumps(crew.performance_metrics)
            })
            
            # Parent crew relationship
            if crew.parent_crew:
                query = """
                MATCH (child:Crew {crewId: $childId})
                MATCH (parent:Crew {name: $parentName})
                MERGE (child)-[:SUBTEAM_OF]->(parent)
                """
                self.connector.execute_query(query, {
                    "childId": crew.crewId,
                    "parentName": crew.parent_crew
                })
            
            # Crew-Process support relationships
            for process_name in crew.processes_supported:
                query = """
                MATCH (c:Crew {crewId: $crewId})
                MATCH (p:Process {name: $processName})
                MERGE (c)-[:SUPPORTS]->(p)
                """
                self.connector.execute_query(query, {
                    "crewId": crew.crewId,
                    "processName": process_name
                })
        
        # Create agents
        for agent in knowledge_graph.agents:
            query = """
            MERGE (a:Agent {agentId: $agentId})
            ON CREATE SET 
                a.name = $name,
                a.title = $title,
                a.performance_metrics = $metrics
            ON MATCH SET
                a.name = $name,
                a.title = $title,
                a.performance_metrics = $metrics
            RETURN a
            """
            self.connector.execute_query(query, {
                "agentId": agent.agentId,
                "name": agent.name,
                "title": agent.title,
                "metrics": json.dumps(agent.performance_metrics)
            })
            
            # Agent-BusinessFunction relationships
            for function in agent.business_functions:
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (bf:BusinessFunction {name: $function})
                MERGE (a)-[:HAS_FUNCTION]->(bf)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.agentId,
                    "function": function
                })
            
            # Agent-TechExpertise relationships
            for expertise in agent.tech_expertise:
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (te:TechExpertise {name: $expertise})
                MERGE (a)-[:HAS_EXPERTISE]->(te)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.agentId,
                    "expertise": expertise
                })
            
            # Agent-Process support relationships
            for process_name in agent.processes_supported:
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (p:Process {name: $processName})
                MERGE (a)-[:SUPPORTS]->(p)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.agentId,
                    "processName": process_name
                })
            
            # Agent-Crew membership relationships
            for crew_name in agent.crews:
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (c:Crew {name: $crewName})
                MERGE (a)-[:MEMBER_OF {role: 'Member'}]->(c)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.agentId,
                    "crewName": crew_name
                })
        
        # Create tasks
        for task in knowledge_graph.tasks:
            query = """
            MERGE (t:Task {taskId: $taskId})
            ON CREATE SET 
                t.title = $title,
                t.status = $status,
                t.priority = $priority
            ON MATCH SET
                t.title = $title,
                t.status = $status,
                t.priority = $priority
            RETURN t
            """
            self.connector.execute_query(query, {
                "taskId": task.taskId,
                "title": task.title,
                "status": task.status,
                "priority": task.priority
            })
            
            # Link task to process
            query = """
            MATCH (t:Task {taskId: $taskId})
            MATCH (p:Process {name: $processName})
            MERGE (t)-[:PART_OF]->(p)
            """
            self.connector.execute_query(query, {
                "taskId": task.taskId,
                "processName": task.process
            })
            
            # Agent-Task assignments
            for agent_name in task.assigned_agents:
                query = """
                MATCH (t:Task {taskId: $taskId})
                MATCH (a:Agent {name: $agentName})
                MERGE (a)-[:ASSIGNED_TO]->(t)
                """
                self.connector.execute_query(query, {
                    "taskId": task.taskId,
                    "agentName": agent_name
                })
        
        # Create performance records
        for record in knowledge_graph.performance_records:
            query = """
            MERGE (pr:PerformanceRecord {recordId: $recordId})
            ON CREATE SET 
                pr.date = date($date),
                pr.metrics = $metrics
            ON MATCH SET
                pr.date = date($date),
                pr.metrics = $metrics
            RETURN pr
            """
            self.connector.execute_query(query, {
                "recordId": record.recordId,
                "date": record.date,
                "metrics": json.dumps(record.metrics)
            })
            
            # Link performance record to agent or crew
            if record.entity_type == "Agent":
                query = """
                MATCH (pr:PerformanceRecord {recordId: $recordId})
                MATCH (a:Agent {agentId: $entityId})
                MERGE (a)-[:HAS_PERFORMANCE_RECORD]->(pr)
                """
                self.connector.execute_query(query, {
                    "recordId": record.recordId,
                    "entityId": record.entity_id
                })
            elif record.entity_type == "Crew":
                query = """
                MATCH (pr:PerformanceRecord {recordId: $recordId})
                MATCH (c:Crew {crewId: $entityId})
                MERGE (c)-[:HAS_PERFORMANCE_RECORD]->(pr)
                """
                self.connector.execute_query(query, {
                    "recordId": record.recordId,
                    "entityId": record.entity_id
                })
    
    def create_example_knowledge(self):
        """Generate and ingest example knowledge for demonstration"""
        example_description = """
        Our retail organization has the following main business processes:
        
        1. Catalog Management - This is the top-level process for managing product information.
           a. Pre-Item-Setup - Process for setting up new items before they go live
           b. Item Maintenance - Process for maintaining existing items
           c. Pricing Management - Process for managing product pricing
        
        2. Supply Chain Management - This manages the flow of goods and services.
           a. Inventory Management - Process for tracking and managing inventory
           b. Order Fulfillment - Process for fulfilling customer orders
           c. Logistics - Process for managing transportation and delivery
        
        The Item Maintenance process depends on Pre-Item-Setup being completed.
        The Order Fulfillment process depends on Inventory Management.
        
        We have the following crews:
        1. Catalog Team - Supports Catalog Management
           a. Item Setup Crew - Subteam of Catalog Team, supports Pre-Item-Setup
           b. Item Maintenance Crew - Subteam of Catalog Team, supports Item Maintenance
           c. Pricing Optimization Crew - Subteam of Catalog Team, supports Pricing Management
        
        2. Supply Chain Team - Supports Supply Chain Management
           a. Inventory Control Crew - Subteam of Supply Chain Team, supports Inventory Management
           b. Fulfillment Crew - Subteam of Supply Chain Team, supports Order Fulfillment
           c. Logistics Crew - Subteam of Supply Chain Team, supports Logistics
        
        We have the following agents:
        1. Alice (Agent ID: A001) - Title: Senior Catalog Manager
           - Business Functions: Retail Merchandising, Catalog Management
           - Technical Expertise: Data Analysis, Product Information Management
           - Supports: Catalog Management
           - Member of: Catalog Team
           - Performance Metrics: 95% success rate, 120 tasks completed
        
        2. Bob (Agent ID: A002) - Title: Item Setup Specialist
           - Business Functions: Item Setup, Data Entry
           - Technical Expertise: Database Management, Data Validation
           - Supports: Pre-Item-Setup
           - Member of: Item Setup Crew
           - Performance Metrics: 92% success rate, 85 tasks completed
        
        3. Charlie (Agent ID: A003) - Title: Supply Chain Manager
           - Business Functions: Supply Chain Management, Logistics Planning
           - Technical Expertise: Inventory Systems, Route Optimization
           - Supports: Supply Chain Management
           - Member of: Supply Chain Team
           - Performance Metrics: 97% success rate, 150 tasks completed
        
        4. Diana (Agent ID: A004) - Title: AI Assistant for Pricing
           - Business Functions: Price Analysis, Competitive Intelligence
           - Technical Expertise: Machine Learning, Price Optimization Algorithms
           - Supports: Pricing Management
           - Member of: Pricing Optimization Crew
           - Performance Metrics: 99% success rate, 300 tasks completed
        """
        
        knowledge_graph = self.generate_knowledge_graph(example_description)
        if knowledge_graph:
            self.ingest_knowledge_graph(knowledge_graph)
            print("Example knowledge graph created successfully!")
        else:
            print("Failed to generate knowledge graph")
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


if __name__ == "__main__":
    ingestion = KnowledgeIngestion()
    ingestion.create_example_knowledge()
    ingestion.close()