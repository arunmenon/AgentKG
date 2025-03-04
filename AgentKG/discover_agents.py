"""
Standalone script to discover agents for a task using GraphRAG.
"""

import os
import json
import openai
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase

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


class GraphRAGAgentDiscovery:
    """
    GraphRAG-based agent discovery system that combines graph querying,
    vector search, and LLM reasoning to find the most appropriate agents
    for executing specific tasks.
    """
    
    def __init__(self):
        """Initialize the GraphRAGAgentDiscovery system with required components"""
        self.connector = Neo4jConnector()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize vector store for agent capabilities (in-memory for demo)
        self.agent_embeddings = {}
        self.agent_capabilities = {}
    
    def close(self):
        """Close database connections"""
        self.connector.close()
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI's embedding model
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Vector embedding
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            a (List[float]): First vector
            b (List[float]): Second vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def index_agents(self):
        """
        Index all agents in the database with their capabilities, expertise, and process context
        for later vector search
        """
        print("Indexing agents with their capabilities...")
        
        # Get all agents with their capabilities
        query = """
        MATCH (a:Agent)
        OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(bf:BusinessFunction)
        OPTIONAL MATCH (a)-[:HAS_EXPERTISE]->(te:TechExpertise)
        OPTIONAL MATCH (a)-[:SUPPORTS]->(p:Process)
        OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
        OPTIONAL MATCH (p)-[:IN_DOMAIN]->(d:Domain)
        
        RETURN 
            a.agentId AS id,
            a.name AS name,
            a.title AS title,
            COLLECT(DISTINCT bf.name) AS business_functions,
            COLLECT(DISTINCT te.name) AS tech_expertise,
            COLLECT(DISTINCT p.name) AS processes,
            COLLECT(DISTINCT d.name) AS domains,
            COLLECT(DISTINCT c.name) AS crews
        """
        
        agents = self.connector.execute_query(query)
        
        for agent in agents:
            # Create a rich description of agent capabilities for embedding
            capability_text = f"Agent {agent['name']}"
            
            if agent['title']:
                capability_text += f" is a {agent['title']}"
            
            if agent['business_functions']:
                capability_text += f" with functions in {', '.join(agent['business_functions'])}"
            
            if agent['tech_expertise']:
                capability_text += f" and technical expertise in {', '.join(agent['tech_expertise'])}"
            
            if agent['processes']:
                capability_text += f". Supports processes: {', '.join(agent['processes'])}"
            
            if agent['domains']:
                capability_text += f" in domains: {', '.join(agent['domains'])}"
            
            if agent['crews']:
                capability_text += f". Member of crews: {', '.join(agent['crews'])}"
            
            # Store agent capabilities for later reference
            self.agent_capabilities[agent['id']] = {
                'id': agent['id'],
                'name': agent['name'],
                'title': agent['title'],
                'business_functions': agent['business_functions'],
                'tech_expertise': agent['tech_expertise'],
                'processes': agent['processes'],
                'domains': agent['domains'],
                'crews': agent['crews'],
                'description': capability_text
            }
            
            # Create and store embedding for agent capabilities
            self.agent_embeddings[agent['id']] = self._get_embedding(capability_text)
        
        print(f"Indexed {len(agents)} agents with their capabilities")
    
    def get_process_context(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Get relevant processes for the task and their context, including agent indicators.
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of process contexts with agent indicators
        """
        # First, get all domains and their processes
        domains_query = """
        MATCH (d:Domain)-[:CONTAINS_PROCESS]->(p:Process)
        RETURN d.name AS domain, COLLECT(p.name) AS processes
        """
        domains = self.connector.execute_query(domains_query)
        
        # Format domain information for LLM
        domain_info = ""
        for domain in domains:
            domain_info += f"Domain: {domain['domain']}\n"
            domain_info += f"Processes: {', '.join(domain['processes'])}\n\n"
        
        # Prompt LLM to identify relevant processes
        prompt = f"""
        Given a task description, identify the most relevant business processes that should be involved.
        
        Task description: {task_description}
        
        Available domains and processes:
        {domain_info}
        
        Return a comma-separated list of the most relevant process names (max 5) from the available processes.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a business process analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        process_text = response.choices[0].message.content
        process_names = [p.strip() for p in process_text.split(",")]
        
        # Get detailed context for the identified processes including agent indicators
        process_contexts = []
        for process_name in process_names:
            query = """
            MATCH (p:Process)
            WHERE p.name CONTAINS $process_name OR $process_name CONTAINS p.name
            OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)
            OPTIONAL MATCH (child:Process)-[:PART_OF]->(p)
            OPTIONAL MATCH (p)-[:IN_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (p)-[:DEPENDS_ON]->(dep:Process)
            OPTIONAL MATCH (p)<-[:DEPENDS_ON]-(dependent:Process)
            
            // Add agent indicators
            OPTIONAL MATCH (a:Agent)-[:SUPPORTS]->(p)
            OPTIONAL MATCH (c:Crew)-[:SUPPORTS]->(p)
            OPTIONAL MATCH (a2:Agent)-[:MEMBER_OF]->(c)
            
            RETURN 
                p.processId AS id,
                p.name AS name,
                p.description AS description,
                parent.name AS parent_process,
                COLLECT(DISTINCT child.name) AS subprocesses,
                d.name AS domain,
                COLLECT(DISTINCT dep.name) AS dependencies,
                COLLECT(DISTINCT dependent.name) AS dependents,
                COUNT(DISTINCT a) AS direct_agent_count,
                COUNT(DISTINCT a2) AS crew_agent_count,
                COLLECT(DISTINCT a.agentId) as agent_ids,
                COLLECT(DISTINCT c.crewId) as crew_ids
            """
            
            results = self.connector.execute_query(query, {"process_name": process_name})
            process_contexts.extend(results)
        
        return process_contexts
    
    def get_agents_by_graph(self, process_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find agents related to specific processes using graph traversal
        
        Args:
            process_contexts (List[Dict[str, Any]]): Process contexts to query for
            
        Returns:
            List[Dict[str, Any]]: Agents with their capabilities related to the processes
        """
        if not process_contexts:
            return []
        
        # Extract process IDs
        process_ids = [context['id'] for context in process_contexts]
        
        # Query for agents directly supporting these processes or their parent/child processes
        query = """
        MATCH (a:Agent)
        MATCH (p:Process)
        WHERE p.processId IN $process_ids
        
        // Direct support
        OPTIONAL MATCH (a)-[:SUPPORTS]->(p)
        
        // Support through crew
        OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)-[:SUPPORTS]->(p)
        
        // Support for parent process
        OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)<-[:SUPPORTS]-(a)
        
        // Support for child process
        OPTIONAL MATCH (child:Process)-[:PART_OF]->(p)
        OPTIONAL MATCH (a)-[:SUPPORTS]->(child)
        
        WHERE (a)-[:SUPPORTS]->(p) OR 
              (a)-[:MEMBER_OF]->(:Crew)-[:SUPPORTS]->(p) OR
              (a)-[:SUPPORTS]->(parent) OR
              (a)-[:SUPPORTS]->(child)
        
        OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(bf:BusinessFunction)
        OPTIONAL MATCH (a)-[:HAS_EXPERTISE]->(te:TechExpertise)
        OPTIONAL MATCH (a)-[:MEMBER_OF]->(crew:Crew)
        
        RETURN 
            a.agentId AS id,
            a.name AS name,
            a.title AS title,
            COLLECT(DISTINCT p.name) AS supported_processes,
            COLLECT(DISTINCT bf.name) AS business_functions,
            COLLECT(DISTINCT te.name) AS tech_expertise,
            COLLECT(DISTINCT crew.name) AS crews,
            CASE
                WHEN (a)-[:SUPPORTS]->(p) THEN 3
                WHEN (a)-[:MEMBER_OF]->(:Crew)-[:SUPPORTS]->(p) THEN 2
                WHEN (a)-[:SUPPORTS]->(parent) THEN 1.5
                WHEN (a)-[:SUPPORTS]->(child) THEN 1
                ELSE 0
            END AS relevance_score
        ORDER BY relevance_score DESC
        """
        
        agents = self.connector.execute_query(query, {"process_ids": process_ids})
        return agents
    
    def get_agents_by_vector_search(self, task_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find agents based on semantic similarity to task description
        
        Args:
            task_description (str): Description of the task
            top_k (int): Number of top agents to return
            
        Returns:
            List[Dict[str, Any]]: Top agents matching the task description
        """
        if not self.agent_embeddings:
            print("No agent embeddings available. Please run index_agents() first.")
            return []
        
        # Get embedding for task description
        task_embedding = self._get_embedding(task_description)
        
        # Calculate similarity scores
        agent_scores = []
        for agent_id, embedding in self.agent_embeddings.items():
            similarity = self._cosine_similarity(task_embedding, embedding)
            agent_scores.append((agent_id, similarity))
        
        # Sort by similarity and get top-k
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        top_agents = agent_scores[:top_k]
        
        # Get full agent details
        result = []
        for agent_id, score in top_agents:
            agent_info = self.agent_capabilities[agent_id].copy()
            agent_info['similarity_score'] = score
            result.append(agent_info)
        
        return result
    
    def rank_agents_with_llm(self, 
                           task_description: str, 
                           graph_agents: List[Dict[str, Any]], 
                           vector_agents: List[Dict[str, Any]], 
                           process_contexts: List[Dict[str, Any]]
                          ) -> List[Dict[str, Any]]:
        """
        Rank agents for a task using LLM reasoning based on both graph and vector search results
        
        Args:
            task_description (str): Description of the task
            graph_agents (List[Dict[str, Any]]): Agents found via graph search
            vector_agents (List[Dict[str, Any]]): Agents found via vector search
            process_contexts (List[Dict[str, Any]]): Process contexts related to the task
            
        Returns:
            List[Dict[str, Any]]: Ranked list of agents with reasoning
        """
        # Combine unique agents from both sources
        all_agents = {}
        
        for agent in graph_agents:
            agent_id = agent['id']
            if agent_id not in all_agents:
                all_agents[agent_id] = agent.copy()
                all_agents[agent_id]['source'] = 'graph'
                all_agents[agent_id]['graph_relevance'] = agent.get('relevance_score', 0)
            else:
                all_agents[agent_id]['source'] = 'both'
                all_agents[agent_id]['graph_relevance'] = agent.get('relevance_score', 0)
        
        for agent in vector_agents:
            agent_id = agent['id']
            if agent_id not in all_agents:
                all_agents[agent_id] = agent.copy()
                all_agents[agent_id]['source'] = 'vector'
                all_agents[agent_id]['vector_similarity'] = agent.get('similarity_score', 0)
            else:
                all_agents[agent_id]['source'] = 'both'
                all_agents[agent_id]['vector_similarity'] = agent.get('similarity_score', 0)
        
        # Prepare process context information
        process_info = ""
        for i, process in enumerate(process_contexts, 1):
            process_info += f"Process {i}: {process['name']}"
            if process.get('description'):
                process_info += f" - {process['description']}"
            if process.get('domain'):
                process_info += f" (Domain: {process['domain']})"
            process_info += "\n"
            
            if process.get('subprocesses'):
                process_info += f"  Subprocesses: {', '.join(process['subprocesses'])}\n"
            
            if process.get('dependencies'):
                process_info += f"  Dependencies: {', '.join(process['dependencies'])}\n"
            
            process_info += "\n"
        
        # Prepare agent information for LLM reasoning
        agent_info = ""
        for i, (agent_id, agent) in enumerate(all_agents.items(), 1):
            agent_info += f"Agent {i}: {agent['name']}"
            if agent.get('title'):
                agent_info += f" - {agent['title']}"
            agent_info += "\n"
            
            if agent.get('business_functions'):
                agent_info += f"  Business Functions: {', '.join(agent['business_functions'])}\n"
            
            if agent.get('tech_expertise'):
                agent_info += f"  Technical Expertise: {', '.join(agent['tech_expertise'])}\n"
            
            if agent.get('processes'):
                agent_info += f"  Processes: {', '.join(agent['processes'])}\n"
            
            if agent.get('crews'):
                agent_info += f"  Crews: {', '.join(agent['crews'])}\n"
            
            source_info = ""
            if agent.get('source') == 'graph':
                source_info = f"Found via process relationship (relevance: {agent.get('graph_relevance', 0):.2f})"
            elif agent.get('source') == 'vector':
                source_info = f"Found via semantic search (similarity: {agent.get('vector_similarity', 0):.2f})"
            else:
                source_info = f"Found via both process relationship (relevance: {agent.get('graph_relevance', 0):.2f}) and semantic search (similarity: {agent.get('vector_similarity', 0):.2f})"
            
            agent_info += f"  Source: {source_info}\n\n"
        
        # Use LLM to rank agents and provide reasoning
        prompt = f"""
        Task: {task_description}
        
        Process Context:
        {process_info}
        
        Available Agents:
        {agent_info}
        
        Based on the task description and the agents' capabilities, please analyze which agents would be most suitable to execute this task. Consider:
        
        1. Relevance of the agent's business functions to the task
        2. Relevance of the agent's technical expertise to the task
        3. Whether the agent supports the processes involved in the task
        4. The agent's membership in relevant crews
        5. The agent's graph relevance score and semantic similarity score
        
        Provide a ranked list of the top 3 most suitable agents, with reasoning for each. Include:
        1. The agent's name and ID
        2. A score from 0-100 indicating how well they match the task requirements
        3. Specific reasoning for why this agent is suitable
        
        Format your response as a JSON object with a "ranked_agents" key containing a list of objects.
        Each object should have "agent_id", "name", "score", and "reasoning" keys.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert in agent selection and orchestration."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            ranking_result = json.loads(response.choices[0].message.content)
            ranked_agents = ranking_result.get("ranked_agents", [])
            
            # Enhance the ranked agents with full details from our agent records
            for agent in ranked_agents:
                agent_id = agent['agent_id']
                if agent_id in all_agents:
                    # Add the original agent details back to the result
                    for key, value in all_agents[agent_id].items():
                        if key not in agent:
                            agent[key] = value
            
            return ranked_agents
        
        except Exception as e:
            print(f"Error ranking agents with LLM: {e}")
            return list(all_agents.values())
    
    def discover_agents(self, task_description: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Discover agents for a task using GraphRAG approach.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple of (ranked agents, explanation)
        """
        # Ensure agents are indexed
        if not self.agent_embeddings:
            self.index_agents()
        
        explanation = []
        
        # Step 1: Get process context with agent indicators
        explanation.append("Step 1: Identifying relevant business processes for the task...")
        process_contexts = self.get_process_context(task_description)
        
        if process_contexts:
            process_names = [p['name'] for p in process_contexts]
            agent_counts = sum([p.get('direct_agent_count', 0) for p in process_contexts])
            crew_agent_counts = sum([p.get('crew_agent_count', 0) for p in process_contexts])
            
            explanation.append(
                f"Found {len(process_contexts)} relevant processes: {', '.join(process_names)}\n" +
                f"These processes are supported by {agent_counts} direct agents and approximately {crew_agent_counts} agents through crews."
            )
        else:
            explanation.append("No specific processes identified. Proceeding with vector search only.")
        
        # Step 2: Find agents by graph relationships
        explanation.append("\nStep 2: Finding agents through graph relationships to these processes...")
        graph_agents = self.get_agents_by_graph(process_contexts) if process_contexts else []
        
        if graph_agents:
            agent_names = [a['name'] for a in graph_agents[:3]]
            explanation.append(f"Found {len(graph_agents)} agents through graph connections, including {', '.join(agent_names[:3])}{' and others.' if len(agent_names) < len(graph_agents) else '.'}")
        else:
            explanation.append("No agents found through direct graph connections.")
        
        # Step 3: Find agents by vector similarity
        explanation.append("\nStep 3: Finding agents through semantic similarity...")
        vector_agents = self.get_agents_by_vector_search(task_description)
        
        if vector_agents:
            agent_names = [a['name'] for a in vector_agents[:3]]
            explanation.append(f"Found {len(vector_agents)} agents through semantic similarity, including {', '.join(agent_names[:3])}{' and others.' if len(agent_names) < len(vector_agents) else '.'}")
        
        # Step 4: Rank agents with LLM
        explanation.append("\nStep 4: Ranking agents based on context, relationships and relevance...")
        ranked_agents = self.rank_agents_with_llm(task_description, graph_agents, vector_agents, process_contexts)
        explanation.append(f"Completed agent ranking.")
        
        explanation_text = "\n".join(explanation)
        return ranked_agents, explanation_text


def add_demo_agents():
    """Add demo agents to the database for testing"""
    connector = Neo4jConnector()
    
    # Create business functions
    business_functions = [
        "Inventory Manager", 
        "Supply Chain Analyst", 
        "Demand Forecaster", 
        "Retail Operations", 
        "Data Analyst",
        "Sales Manager",
        "Marketing Specialist",
        "Customer Service Rep",
        "Financial Analyst",
        "Product Developer"
    ]
    
    for bf in business_functions:
        query = """
        MERGE (bf:BusinessFunction {name: $name})
        RETURN bf
        """
        connector.execute_query(query, {"name": bf})
    
    # Create technical expertise
    tech_expertise = [
        "Data Science", 
        "Machine Learning", 
        "Inventory Systems", 
        "Supply Chain Optimization", 
        "Predictive Analytics",
        "Database Management",
        "Python",
        "SQL",
        "Business Intelligence",
        "Data Visualization"
    ]
    
    for te in tech_expertise:
        query = """
        MERGE (te:TechExpertise {name: $name})
        RETURN te
        """
        connector.execute_query(query, {"name": te})
    
    # Create crews
    crews = [
        {"id": "CREW-101", "name": "Inventory Optimization Team"},
        {"id": "CREW-102", "name": "Supply Chain Analytics"},
        {"id": "CREW-103", "name": "Retail Operations Center"},
        {"id": "CREW-104", "name": "Data Science Team"},
        {"id": "CREW-105", "name": "Customer Analytics Group"}
    ]
    
    for crew in crews:
        query = """
        MERGE (c:Crew {crewId: $id})
        ON CREATE SET c.name = $name
        RETURN c
        """
        connector.execute_query(query, {"id": crew["id"], "name": crew["name"]})
    
    # Link crews to relevant processes
    crew_process_links = [
        {"crew_id": "CREW-101", "process_id": "PROC-002", "domain": "Retail"},  # Inventory Optimization -> Inventory Management
        {"crew_id": "CREW-102", "process_id": "PROC-001", "domain": "Supply Chain"},  # Supply Chain Analytics -> Sourcing and Procurement
        {"crew_id": "CREW-103", "process_id": "PROC-001", "domain": "Retail"},  # Retail Operations -> Sales Management
        {"crew_id": "CREW-104", "process_id": "PROC-002-001", "domain": "Supply Chain"},  # Data Science -> Demand Forecasting
        {"crew_id": "CREW-105", "process_id": "PROC-001", "domain": "Customer Service"}  # Customer Analytics -> Customer Inquiry Management
    ]
    
    for link in crew_process_links:
        # Find the process with the given ID in the specified domain
        process_query = """
        MATCH (p:Process {processId: $process_id})-[:IN_DOMAIN]->(d:Domain {name: $domain})
        RETURN p
        """
        processes = connector.execute_query(process_query, {"process_id": link["process_id"], "domain": link["domain"]})
        
        if processes:
            crew_query = """
            MATCH (c:Crew {crewId: $crew_id})
            MATCH (p:Process {processId: $process_id})
            MERGE (c)-[:SUPPORTS]->(p)
            """
            connector.execute_query(crew_query, {"crew_id": link["crew_id"], "process_id": link["process_id"]})
    
    # Create agents
    agents = [
        {
            "id": "AGENT-101",
            "name": "Alex Thompson",
            "title": "Senior Inventory Analyst",
            "business_functions": ["Inventory Manager", "Supply Chain Analyst"],
            "tech_expertise": ["Data Science", "Inventory Systems"],
            "crew": "CREW-101",
            "process_id": "PROC-002",
            "domain": "Retail"
        },
        {
            "id": "AGENT-102",
            "name": "Jordan Lee",
            "title": "Supply Chain Optimization Specialist",
            "business_functions": ["Supply Chain Analyst", "Demand Forecaster"],
            "tech_expertise": ["Supply Chain Optimization", "Predictive Analytics"],
            "crew": "CREW-102",
            "process_id": "PROC-003",
            "domain": "Supply Chain"
        },
        {
            "id": "AGENT-103",
            "name": "Morgan Chen",
            "title": "Retail Analytics Manager",
            "business_functions": ["Retail Operations", "Data Analyst"],
            "tech_expertise": ["Business Intelligence", "Data Visualization"],
            "crew": "CREW-103",
            "process_id": "PROC-001",
            "domain": "Retail"
        },
        {
            "id": "AGENT-104",
            "name": "Dana Williams",
            "title": "Data Scientist",
            "business_functions": ["Data Analyst", "Demand Forecaster"],
            "tech_expertise": ["Machine Learning", "Python", "SQL"],
            "crew": "CREW-104",
            "process_id": "PROC-002-001",
            "domain": "Supply Chain"
        },
        {
            "id": "AGENT-105",
            "name": "Taylor Rodriguez",
            "title": "Customer Experience Analyst",
            "business_functions": ["Customer Service Rep", "Data Analyst"],
            "tech_expertise": ["Data Analysis", "Customer Behavior Modeling"],
            "crew": "CREW-105",
            "process_id": "PROC-001",
            "domain": "Customer Service"
        }
    ]
    
    for agent in agents:
        # Create the agent
        agent_query = """
        MERGE (a:Agent {agentId: $id})
        ON CREATE SET
            a.name = $name,
            a.title = $title,
            a.performance_metrics = '{"tasksCompleted": 120, "successRate": 0.95, "efficiency": 0.90}'
        RETURN a
        """
        connector.execute_query(agent_query, {"id": agent["id"], "name": agent["name"], "title": agent["title"]})
        
        # Link to business functions
        for bf in agent["business_functions"]:
            bf_query = """
            MATCH (a:Agent {agentId: $agent_id})
            MATCH (bf:BusinessFunction {name: $bf_name})
            MERGE (a)-[:HAS_FUNCTION]->(bf)
            """
            connector.execute_query(bf_query, {"agent_id": agent["id"], "bf_name": bf})
        
        # Link to technical expertise
        for te in agent["tech_expertise"]:
            te_query = """
            MATCH (a:Agent {agentId: $agent_id})
            MATCH (te:TechExpertise {name: $te_name})
            MERGE (a)-[:HAS_EXPERTISE]->(te)
            """
            connector.execute_query(te_query, {"agent_id": agent["id"], "te_name": te})
        
        # Link to crew
        crew_query = """
        MATCH (a:Agent {agentId: $agent_id})
        MATCH (c:Crew {crewId: $crew_id})
        MERGE (a)-[:MEMBER_OF {role: 'Member'}]->(c)
        """
        connector.execute_query(crew_query, {"agent_id": agent["id"], "crew_id": agent["crew"]})
        
        # Find and link to process
        process_query = """
        MATCH (p:Process {processId: $process_id})-[:IN_DOMAIN]->(d:Domain {name: $domain})
        RETURN p
        """
        processes = connector.execute_query(process_query, {"process_id": agent["process_id"], "domain": agent["domain"]})
        
        if processes:
            support_query = """
            MATCH (a:Agent {agentId: $agent_id})
            MATCH (p:Process {processId: $process_id})
            MERGE (a)-[:SUPPORTS]->(p)
            """
            connector.execute_query(support_query, {"agent_id": agent["id"], "process_id": agent["process_id"]})
    
    print("Demo agents added successfully!")
    connector.close()

def main():
    """Main function for discovering agents for a task"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover agents for a task using GraphRAG")
    parser.add_argument("--task", type=str, required=True, help="Description of the task")
    parser.add_argument("--add-demo-agents", action="store_true", help="Add demo agents to the database before discovery")
    
    args = parser.parse_args()
    
    if args.add_demo_agents:
        add_demo_agents()
    
    discovery = GraphRAGAgentDiscovery()
    ranked_agents, explanation = discovery.discover_agents(args.task)
    
    print(explanation)
    
    print("\n=== Top Ranked Agents ===")
    for i, agent in enumerate(ranked_agents[:3], 1):
        print(f"{i}. {agent['name']} (Score: {agent['score']})")
        print(f"   {agent['reasoning']}")
        print()
    
    discovery.close()

if __name__ == "__main__":
    main()