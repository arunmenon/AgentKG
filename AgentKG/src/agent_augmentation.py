from .neo4j_connector import Neo4jConnector
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KnowledgeAugmentation:
    """
    Class responsible for augmenting the knowledge graph
    using agents with search tools
    """
    
    def __init__(self):
        """Initialize the Neo4j connector, LLM, and search tools"""
        self.connector = Neo4jConnector()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.search_tool = DuckDuckGoSearchRun()
    
    def search_and_extract_knowledge(self, query):
        """
        Use search tools to find information about a topic
        and extract structured knowledge
        
        Args:
            query (str): The search query about the domain
            
        Returns:
            dict: Extracted knowledge
        """
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert knowledge extraction agent.
            You need to search for information about a business process, agent, or team
            and extract structured knowledge for a knowledge graph.
            
            1. Search for relevant information using the search tool
            2. Extract entities like processes, agents, teams, domains, etc.
            3. Identify relationships between these entities
            4. Structure the information in a format compatible with a Neo4j knowledge graph
            """),
            ("human", "{query}")
        ])
        
        # Set up the agent
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=[self.search_tool],
            prompt=search_prompt
        )
        
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=[self.search_tool],
            verbose=True
        )
        
        # Execute the search and knowledge extraction
        result = agent_executor.invoke({"query": query})
        
        # Process and structure the response
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge graph structuring assistant.
            Take the search results and extract structured entities and relationships
            for our knowledge graph.
            
            Return a JSON object with the following structure:
            {
                "processes": [
                    {
                        "name": "Process Name",
                        "description": "Process description",
                        "processId": "PID-123", 
                        "parent_process": "Parent Process Name",
                        "dependencies": ["Dependent Process 1", "Dependent Process 2"],
                        "domain": "Domain Name"
                    }
                ],
                "agents": [
                    {
                        "name": "Agent Name",
                        "agentId": "AID-123",
                        "title": "Agent Title",
                        "business_functions": ["Function1", "Function2"],
                        "tech_expertise": ["Expertise1", "Expertise2"],
                        "processes_supported": ["Process1", "Process2"],
                        "crews": ["Crew1", "Crew2"]
                    }
                ],
                "crews": [
                    {
                        "name": "Crew Name",
                        "crewId": "CID-123",
                        "parent_crew": "Parent Crew Name",
                        "processes_supported": ["Process1", "Process2"]
                    }
                ],
                "business_functions": ["Function1", "Function2"],
                "tech_expertise": ["Expertise1", "Expertise2"],
                "domains": ["Domain1", "Domain2"]
            }
            
            Only include entities that were found in the search results.
            Generate IDs for new entities but keep them consistent with naming.
            """),
            ("human", "Structure this search result into knowledge graph entities: {search_result}")
        ])
        
        extraction_response = self.llm.invoke(
            extraction_prompt.format(search_result=result['output'])
        )
        
        try:
            # Extract JSON structure from the response
            content = extraction_response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            structured_knowledge = json.loads(content)
            return structured_knowledge
        except Exception as e:
            print(f"Error parsing structured knowledge: {e}")
            print(f"Raw response: {extraction_response.content}")
            return None
    
    def add_new_knowledge(self, structured_knowledge):
        """
        Add new knowledge to the graph from structured extraction
        
        Args:
            structured_knowledge (dict): The structured knowledge to add
        """
        # Create new domains if any
        for domain in structured_knowledge.get('domains', []):
            query = """
            MERGE (d:Domain {name: $name})
            RETURN d
            """
            self.connector.execute_query(query, {"name": domain})
        
        # Create new business functions if any
        for function in structured_knowledge.get('business_functions', []):
            query = """
            MERGE (bf:BusinessFunction {name: $name})
            RETURN bf
            """
            self.connector.execute_query(query, {"name": function})
        
        # Create new technical expertise if any
        for expertise in structured_knowledge.get('tech_expertise', []):
            query = """
            MERGE (te:TechExpertise {name: $name})
            RETURN te
            """
            self.connector.execute_query(query, {"name": expertise})
        
        # Create new processes if any
        for process in structured_knowledge.get('processes', []):
            query = """
            MERGE (p:Process {processId: $processId})
            ON CREATE SET 
                p.name = $name,
                p.description = $description,
                p.status = $status
            ON MATCH SET
                p.name = $name,
                p.description = CASE WHEN $description IS NULL THEN p.description ELSE $description END
            RETURN p
            """
            self.connector.execute_query(query, {
                "processId": process.get('processId'),
                "name": process.get('name'),
                "description": process.get('description'),
                "status": process.get('status', 'Active')
            })
            
            # Link to domain if specified
            if process.get('domain'):
                query = """
                MATCH (p:Process {processId: $processId})
                MATCH (d:Domain {name: $domain})
                MERGE (p)-[:IN_DOMAIN]->(d)
                """
                self.connector.execute_query(query, {
                    "processId": process.get('processId'),
                    "domain": process.get('domain')
                })
            
            # Parent process relationship
            if process.get('parent_process'):
                query = """
                MATCH (child:Process {processId: $childId})
                MATCH (parent:Process {name: $parentName})
                MERGE (child)-[:PART_OF]->(parent)
                """
                self.connector.execute_query(query, {
                    "childId": process.get('processId'),
                    "parentName": process.get('parent_process')
                })
            
            # Process dependencies
            for dependency in process.get('dependencies', []):
                query = """
                MATCH (proc:Process {processId: $processId})
                MATCH (dep:Process {name: $depName})
                MERGE (proc)-[:DEPENDS_ON]->(dep)
                """
                self.connector.execute_query(query, {
                    "processId": process.get('processId'),
                    "depName": dependency
                })
        
        # Create new crews if any
        for crew in structured_knowledge.get('crews', []):
            query = """
            MERGE (c:Crew {crewId: $crewId})
            ON CREATE SET 
                c.name = $name,
                c.performance_metrics = $metrics
            ON MATCH SET
                c.name = $name
            RETURN c
            """
            self.connector.execute_query(query, {
                "crewId": crew.get('crewId'),
                "name": crew.get('name'),
                "metrics": "{}"
            })
            
            # Parent crew relationship
            if crew.get('parent_crew'):
                query = """
                MATCH (child:Crew {crewId: $childId})
                MATCH (parent:Crew {name: $parentName})
                MERGE (child)-[:SUBTEAM_OF]->(parent)
                """
                self.connector.execute_query(query, {
                    "childId": crew.get('crewId'),
                    "parentName": crew.get('parent_crew')
                })
            
            # Crew-Process support relationships
            for process_name in crew.get('processes_supported', []):
                query = """
                MATCH (c:Crew {crewId: $crewId})
                MATCH (p:Process {name: $processName})
                MERGE (c)-[:SUPPORTS]->(p)
                """
                self.connector.execute_query(query, {
                    "crewId": crew.get('crewId'),
                    "processName": process_name
                })
        
        # Create new agents if any
        for agent in structured_knowledge.get('agents', []):
            query = """
            MERGE (a:Agent {agentId: $agentId})
            ON CREATE SET 
                a.name = $name,
                a.title = $title,
                a.performance_metrics = $metrics
            ON MATCH SET
                a.name = $name,
                a.title = CASE WHEN $title IS NULL THEN a.title ELSE $title END
            RETURN a
            """
            self.connector.execute_query(query, {
                "agentId": agent.get('agentId'),
                "name": agent.get('name'),
                "title": agent.get('title'),
                "metrics": "{}"
            })
            
            # Agent-BusinessFunction relationships
            for function in agent.get('business_functions', []):
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (bf:BusinessFunction {name: $function})
                MERGE (a)-[:HAS_FUNCTION]->(bf)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.get('agentId'),
                    "function": function
                })
            
            # Agent-TechExpertise relationships
            for expertise in agent.get('tech_expertise', []):
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (te:TechExpertise {name: $expertise})
                MERGE (a)-[:HAS_EXPERTISE]->(te)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.get('agentId'),
                    "expertise": expertise
                })
            
            # Agent-Process support relationships
            for process_name in agent.get('processes_supported', []):
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (p:Process {name: $processName})
                MERGE (a)-[:SUPPORTS]->(p)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.get('agentId'),
                    "processName": process_name
                })
            
            # Agent-Crew membership relationships
            for crew_name in agent.get('crews', []):
                query = """
                MATCH (a:Agent {agentId: $agentId})
                MATCH (c:Crew {name: $crewName})
                MERGE (a)-[:MEMBER_OF]->(c)
                """
                self.connector.execute_query(query, {
                    "agentId": agent.get('agentId'),
                    "crewName": crew_name
                })
    
    def augment_knowledge_with_search(self, topic):
        """
        Search for information about a topic and add it to the knowledge graph
        
        Args:
            topic (str): The topic to search for
        
        Returns:
            dict: The structured knowledge that was added
        """
        print(f"Searching for knowledge about: {topic}")
        
        # Search and extract structured knowledge
        structured_knowledge = self.search_and_extract_knowledge(topic)
        
        if structured_knowledge:
            # Add the new knowledge to the graph
            self.add_new_knowledge(structured_knowledge)
            print(f"Successfully added new knowledge about {topic} to the graph")
            return structured_knowledge
        else:
            print(f"Failed to extract structured knowledge about {topic}")
            return None
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


if __name__ == "__main__":
    # Example usage
    augmentation = KnowledgeAugmentation()
    
    # Example search topics
    topics = [
        "retail catalog management process",
        "e-commerce supply chain optimization",
        "AI agents in retail pricing",
        "inventory management crews"
    ]
    
    for topic in topics:
        augmentation.augment_knowledge_with_search(topic)
    
    augmentation.close()