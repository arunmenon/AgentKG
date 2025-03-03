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
        
        # Execute these queries to create sample data
        for query in sample_queries:
            self.connector.execute_query(query)
    
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


class GraphQuerier:
    """
    Utility class for querying the graph
    """
    
    def __init__(self):
        """Initialize the Neo4j connector"""
        self.connector = Neo4jConnector()
    
    def get_process_hierarchy(self):
        """
        Get the complete process hierarchy with dependencies
        
        Returns:
            list: Process hierarchy data
        """
        query = """
        MATCH (p:Process)
        OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)
        OPTIONAL MATCH (p)-[:DEPENDS_ON]->(dependency:Process)
        OPTIONAL MATCH (p)-[:IN_DOMAIN]->(domain:Domain)
        RETURN 
            p.processId AS id,
            p.name AS name, 
            p.description AS description,
            p.status AS status,
            parent.name AS parent_process,
            COLLECT(DISTINCT dependency.name) AS dependencies,
            domain.name AS domain
        ORDER BY 
            CASE WHEN parent.name IS NULL THEN 0 ELSE 1 END,
            parent.name,
            p.name
        """
        
        return self.connector.execute_query(query)
    
    def get_agent_details(self, agent_name=None):
        """
        Get agent details with their functions, expertise, and assignments
        
        Args:
            agent_name (str, optional): Filter by agent name
            
        Returns:
            list: Agent details
        """
        query = """
        MATCH (a:Agent)
        WHERE $agent_name IS NULL OR a.name = $agent_name
        OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(bf:BusinessFunction)
        OPTIONAL MATCH (a)-[:HAS_EXPERTISE]->(te:TechExpertise)
        OPTIONAL MATCH (a)-[:SUPPORTS]->(p:Process)
        OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
        OPTIONAL MATCH (a)-[:HAS_PERFORMANCE_RECORD]->(pr:PerformanceRecord)
        RETURN 
            a.agentId AS id,
            a.name AS name,
            a.title AS title,
            COLLECT(DISTINCT bf.name) AS business_functions,
            COLLECT(DISTINCT te.name) AS tech_expertise,
            COLLECT(DISTINCT p.name) AS processes_supported,
            COLLECT(DISTINCT c.name) AS crews,
            COLLECT(DISTINCT {
                recordId: pr.recordId,
                date: pr.date
            }) AS performance_records
        ORDER BY a.name
        """
        
        return self.connector.execute_query(query, {"agent_name": agent_name})
    
    def get_crew_structure(self):
        """
        Get the complete crew structure with hierarchy and assignments
        
        Returns:
            list: Crew structure data
        """
        query = """
        MATCH (c:Crew)
        OPTIONAL MATCH (c)-[:SUBTEAM_OF]->(parent:Crew)
        OPTIONAL MATCH (c)-[:SUPPORTS]->(p:Process)
        OPTIONAL MATCH (member:Agent)-[:MEMBER_OF]->(c)
        RETURN 
            c.crewId AS id,
            c.name AS name,
            parent.name AS parent_crew,
            COLLECT(DISTINCT p.name) AS processes_supported,
            COLLECT(DISTINCT member.name) AS members
        ORDER BY 
            CASE WHEN parent.name IS NULL THEN 0 ELSE 1 END,
            parent.name,
            c.name
        """
        
        return self.connector.execute_query(query)
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


def test_schema():
    """Test the schema creation and sample data"""
    # Create schema
    schema_creator = SchemaCreator()
    schema_creator.clear_database()
    schema_creator.create_schema()
    schema_creator.create_sample_data()
    schema_creator.close()
    
    print("Schema and sample data created successfully")
    
    # Query data
    querier = GraphQuerier()
    
    print("\n=== Process Hierarchy ===")
    process_hierarchy = querier.get_process_hierarchy()
    for process in process_hierarchy:
        print(f"Process: {process['name']}")
        if process['parent_process']:
            print(f"  Parent: {process['parent_process']}")
        if process['dependencies'] and len(process['dependencies']) > 0:
            print(f"  Dependencies: {', '.join(process['dependencies'])}")
        if process['domain']:
            print(f"  Domain: {process['domain']}")
        print()
    
    print("\n=== Agent Details ===")
    agents = querier.get_agent_details()
    for agent in agents:
        print(f"Agent: {agent['name']} ({agent['title']})")
        print(f"  Business Functions: {', '.join(agent['business_functions'])}")
        print(f"  Technical Expertise: {', '.join(agent['tech_expertise'])}")
        print(f"  Processes Supported: {', '.join(agent['processes_supported'])}")
        print(f"  Member of Crews: {', '.join(agent['crews'])}")
        print()
    
    print("\n=== Crew Structure ===")
    crews = querier.get_crew_structure()
    for crew in crews:
        print(f"Crew: {crew['name']}")
        if crew['parent_crew']:
            print(f"  Parent Crew: {crew['parent_crew']}")
        print(f"  Processes Supported: {', '.join(crew['processes_supported'])}")
        print(f"  Members: {', '.join(crew['members'])}")
        print()
    
    querier.close()


if __name__ == "__main__":
    test_schema()