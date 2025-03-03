import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

class Neo4jConnector:
    """
    A connector class for Neo4j database operations
    """
    
    def __init__(self):
        """Initialize the Neo4j connection using environment variables"""
        load_dotenv()
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
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
            
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record for record in result]
    
    def execute_transaction(self, func, *args, **kwargs):
        """
        Execute a function within a transaction
        
        Args:
            func (callable): Function to execute inside transaction
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Any: Result of the transaction function
        """
        with self.driver.session() as session:
            result = session.execute_write(func, *args, **kwargs)
            return result