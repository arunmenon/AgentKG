"""
GraphRAG Agent Discovery for AgentKG.

This module implements GraphRAG (Graph Retrieval Augmented Generation) recipes
for discovering appropriate agents to execute specific tasks based on:
1. Graph traversal to find relevant agents by expertise, function, and process
2. Semantic search to match task requirements with agent capabilities
3. RAG to enhance agent selection with context-aware reasoning
"""

import os
import json
import openai
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from .neo4j_connector import Neo4jConnector

# Load environment variables
load_dotenv()

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
        Get process context related to a task description using RAG
        
        Args:
            task_description (str): Description of the task
            
        Returns:
            List[Dict[str, Any]]: Related processes with their contexts
        """
        # Use LLM to identify key processes from task description
        prompt = f"""
        Extract the most relevant business processes mentioned or implied in the following task description.
        For each process, provide its name and the domain it might belong to.
        
        Task description: {task_description}
        
        Format your response as a JSON list of objects, each with 'process_name' and 'domain' keys.
        If the domain is uncertain, use "Unknown" as the value.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert in business process analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        try:
            extracted_processes = json.loads(response.choices[0].message.content)
            processes = extracted_processes.get("processes", [])
            
            # If no processes were identified, return empty list
            if not processes:
                return []
            
            # Query the graph database for these processes to get more context
            process_contexts = []
            
            for process_info in processes:
                process_name = process_info.get("process_name")
                domain = process_info.get("domain")
                
                query = """
                MATCH (p:Process)
                WHERE p.name CONTAINS $process_name OR $process_name CONTAINS p.name
                OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)
                OPTIONAL MATCH (child:Process)-[:PART_OF]->(p)
                OPTIONAL MATCH (p)-[:IN_DOMAIN]->(d:Domain)
                OPTIONAL MATCH (p)-[:DEPENDS_ON]->(dep:Process)
                OPTIONAL MATCH (p)<-[:DEPENDS_ON]-(dependent:Process)
                
                RETURN 
                    p.processId AS id,
                    p.name AS name,
                    p.description AS description,
                    parent.name AS parent_process,
                    COLLECT(DISTINCT child.name) AS subprocesses,
                    d.name AS domain,
                    COLLECT(DISTINCT dep.name) AS dependencies,
                    COLLECT(DISTINCT dependent.name) AS dependents
                """
                
                results = self.connector.execute_query(query, {"process_name": process_name})
                
                for result in results:
                    # If domain from the graph doesn't match the predicted domain and both are known, skip
                    if result['domain'] and domain != "Unknown" and result['domain'] != domain:
                        continue
                        
                    process_contexts.append(result)
            
            return process_contexts
            
        except Exception as e:
            print(f"Error extracting processes: {e}")
            return []
    
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
        Main method to discover appropriate agents for a task using GraphRAG
        
        Args:
            task_description (str): Description of the task
            
        Returns:
            Tuple[List[Dict[str, Any]], str]: Ranked agents and explanation of the discovery process
        """
        process_contexts = []
        graph_agents = []
        vector_agents = []
        explanation = []
        
        # Make sure we have indexed agents for vector search
        if not self.agent_embeddings:
            self.index_agents()
        
        # Step 1: Get process context related to the task
        explanation.append("Step 1: Identifying relevant business processes for the task...")
        process_contexts = self.get_process_context(task_description)
        
        if process_contexts:
            process_names = [p['name'] for p in process_contexts]
            explanation.append(f"Found {len(process_contexts)} relevant processes: {', '.join(process_names)}")
        else:
            explanation.append("No specific processes identified for this task.")
        
        # Step 2: Find agents through graph traversal
        explanation.append("\nStep 2: Finding agents through graph relationships...")
        graph_agents = self.get_agents_by_graph(process_contexts)
        
        if graph_agents:
            agent_names = [a['name'] for a in graph_agents[:5]]
            explanation.append(f"Found {len(graph_agents)} agents through process relationships. Top agents: {', '.join(agent_names)}")
        else:
            explanation.append("No agents found through process relationships.")
        
        # Step 3: Find agents through vector similarity
        explanation.append("\nStep 3: Finding agents through semantic similarity...")
        vector_agents = self.get_agents_by_vector_search(task_description)
        
        if vector_agents:
            agent_names = [f"{a['name']} (similarity: {a['similarity_score']:.2f})" for a in vector_agents[:5]]
            explanation.append(f"Found {len(vector_agents)} agents through semantic search. Top agents: {', '.join(agent_names)}")
        else:
            explanation.append("No agents found through semantic search.")
        
        # Step 4: Rank agents using LLM reasoning
        explanation.append("\nStep 4: Ranking agents using contextual reasoning...")
        ranked_agents = self.rank_agents_with_llm(task_description, graph_agents, vector_agents, process_contexts)
        
        if ranked_agents:
            explanation.append(f"Final ranking complete. Top agent: {ranked_agents[0]['name']} (Score: {ranked_agents[0]['score']})")
            for agent in ranked_agents[:3]:
                explanation.append(f"\n{agent['name']} (Score: {agent['score']}): {agent['reasoning']}")
        else:
            explanation.append("No suitable agents found for this task.")
        
        return ranked_agents, "\n".join(explanation)


def main():
    """Example usage of GraphRAGAgentDiscovery"""
    discovery = GraphRAGAgentDiscovery()
    
    # Example task
    task_description = "We need to optimize our inventory levels across all retail stores to reduce stockouts while minimizing excess inventory."
    
    # Discover agents for the task
    ranked_agents, explanation = discovery.discover_agents(task_description)
    
    # Print explanation of the discovery process
    print(explanation)
    
    # Print the top ranked agents
    print("\n=== Top Ranked Agents ===")
    for i, agent in enumerate(ranked_agents[:3], 1):
        print(f"{i}. {agent['name']} (Score: {agent['score']})")
        print(f"   {agent['reasoning']}")
        print()
    
    # Close connections
    discovery.close()


if __name__ == "__main__":
    main()