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

class SchemaCreator:
    """
    Class responsible for creating the Neo4j schema with constraints and indexes
    """
    
    def __init__(self):
        """Initialize the Neo4j connector"""
        self.connector = Neo4jConnector()
    
    def create_constraints(self):
        """Create constraints for uniqueness and indexing"""
        constraints = [
            # Node uniqueness constraints
            "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.processId IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.agentId IS UNIQUE",
            "CREATE CONSTRAINT crew_id IF NOT EXISTS FOR (c:Crew) REQUIRE c.crewId IS UNIQUE",
            "CREATE CONSTRAINT business_function_name IF NOT EXISTS FOR (bf:BusinessFunction) REQUIRE bf.name IS UNIQUE",
            "CREATE CONSTRAINT tech_expertise_name IF NOT EXISTS FOR (te:TechExpertise) REQUIRE te.name IS UNIQUE",
            "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.taskId IS UNIQUE",
            "CREATE CONSTRAINT performance_record_id IF NOT EXISTS FOR (pr:PerformanceRecord) REQUIRE pr.recordId IS UNIQUE",
            
            # Indexes for faster lookups
            "CREATE INDEX process_name_idx IF NOT EXISTS FOR (p:Process) ON (p.name)",
            "CREATE INDEX agent_name_idx IF NOT EXISTS FOR (a:Agent) ON (a.name)",
            "CREATE INDEX crew_name_idx IF NOT EXISTS FOR (c:Crew) ON (c.name)",
            "CREATE INDEX task_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX performance_record_date_idx IF NOT EXISTS FOR (pr:PerformanceRecord) ON (pr.date)"
        ]
        
        for constraint in constraints:
            self.connector.execute_query(constraint)
    
    def create_sample_data(self):
        """Create sample data to show the schema structure"""
        # This function is just to visualize the schema, not to populate real data
        sample_queries = [
            # Create a domain
            """
            CREATE (d:Domain {name: 'Retail', description: 'Retail business domain'})
            """,
            
            # Create a top-level process and sub-process
            """
            CREATE 
              (catalog:Process {processId: 'PROC-001', name: 'Catalog Management', status: 'Active', description: 'Top-level process for catalog management'}),
              (preItemSetup:Process {processId: 'PROC-002', name: 'Pre-Item-Setup', status: 'Active', description: 'Process for setting up new items'})
            MERGE (preItemSetup)-[:PART_OF]->(catalog)
            MERGE (catalog)-[:IN_DOMAIN]->(:Domain {name: 'Retail'})
            """,
            
            # Create business functions and tech expertise
            """
            CREATE 
              (bf1:BusinessFunction {name: 'Catalog Manager'}),
              (bf2:BusinessFunction {name: 'Data Entry Specialist'}),
              (te1:TechExpertise {name: 'Data Analysis'}),
              (te2:TechExpertise {name: 'Product Information Management'})
            """,
            
            # Create crews with hierarchy
            """
            CREATE 
              (c1:Crew {crewId: 'CREW-001', name: 'Catalog Team'}),
              (c2:Crew {crewId: 'CREW-002', name: 'Item Setup Crew'})
            MERGE (c2)-[:SUBTEAM_OF]->(c1)
            MERGE (c1)-[:SUPPORTS]->(:Process {name: 'Catalog Management'})
            MERGE (c2)-[:SUPPORTS]->(:Process {name: 'Pre-Item-Setup'})
            """,
            
            # Create an agent with functions, expertise and crew membership
            """
            CREATE (a:Agent {agentId: 'AGENT-001', name: 'Alice', title: 'Senior Catalog Manager'})
            MERGE (a)-[:HAS_FUNCTION]->(:BusinessFunction {name: 'Catalog Manager'})
            MERGE (a)-[:HAS_EXPERTISE]->(:TechExpertise {name: 'Data Analysis'})
            MERGE (a)-[:MEMBER_OF {role: 'Manager'}]->(:Crew {name: 'Catalog Team'})
            MERGE (a)-[:SUPPORTS]->(:Process {name: 'Catalog Management'})
            """,
            
            # Create a task
            """
            CREATE (t:Task {
              taskId: 'TASK-001', 
              title: 'Update product descriptions', 
              status: 'In-Progress',
              priority: 'High'
            })
            MERGE (t)-[:PART_OF]->(:Process {name: 'Pre-Item-Setup'})
            MERGE (:Agent {name: 'Alice'})-[:ASSIGNED_TO]->(t)
            """,
            
            # Create a performance record linked to an agent
            """
            CREATE (pr:PerformanceRecord {
              recordId: 'PR-001',
              date: date('2023-01-15'),
              tasksCompleted: 95,
              successRate: 0.92,
              efficiency: 0.88
            })
            MERGE (:Agent {name: 'Alice'})-[:HAS_PERFORMANCE_RECORD]->(pr)
            """
        ]
        
        # Only execute these in development/test environments
        # Uncommenting the following line will create sample data which can be used to visualize the schema
        # for query in sample_queries:
        #     self.connector.execute_query(query)
    
    def clear_database(self):
        """Clear all data from the database (for testing/reset purposes)"""
        query = "MATCH (n) DETACH DELETE n"
        self.connector.execute_query(query)
    
    def create_schema(self):
        """Create the complete schema for the knowledge graph"""
        # First create constraints and indexes
        self.create_constraints()
        
        print("Schema constraints and indexes created successfully")
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


if __name__ == "__main__":
    schema_creator = SchemaCreator()
    
    # Uncomment to reset the database (BE CAREFUL!)
    # schema_creator.clear_database()
    
    # Create schema
    schema_creator.create_schema()
    schema_creator.close()
    print("Schema setup complete!")