"""
Standalone script to populate the AgentKG graph with detailed process hierarchies.
"""

import os
import json
import openai
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    
    def execute_transaction(self, func, *args, **kwargs):
        """
        Execute a function within a transaction
        
        Args:
            func (callable): Function to execute inside transaction
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Any: Result of the transaction function
        """
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(func, *args, **kwargs)
            return result


class ProcessHierarchyPopulator:
    """
    Class responsible for populating the Neo4j database with detailed process hierarchies
    """
    
    def __init__(self):
        """Initialize the Neo4j connector and OpenAI client"""
        self.connector = Neo4jConnector()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def clear_database(self):
        """Clear all data from the database (CAUTION: This will delete all data)"""
        confirm = input("This will delete ALL data in the database. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            query = "MATCH (n) DETACH DELETE n"
            self.connector.execute_query(query)
            print("Database cleared successfully.")
        else:
            print("Database clear operation cancelled.")
    
    def create_constraints(self):
        """Create constraints for the process hierarchy"""
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
            try:
                self.connector.execute_query(constraint)
            except Exception as e:
                print(f"Error creating constraint: {e}")
        
        print("Schema constraints and indexes created successfully")
    
    def add_domain(self, name, description):
        """Add a domain to the graph"""
        query = """
        MERGE (d:Domain {name: $name})
        ON CREATE SET d.description = $description
        ON MATCH SET d.description = $description
        RETURN d
        """
        self.connector.execute_query(query, {"name": name, "description": description})
        print(f"Domain added: {name}")
    
    def add_process(self, process_id, name, description, domain_name, status="Active"):
        """Add a top-level process to the graph"""
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
        
        WITH p
        
        MATCH (d:Domain {name: $domainName})
        MERGE (p)-[:IN_DOMAIN]->(d)
        
        RETURN p
        """
        self.connector.execute_query(query, {
            "processId": process_id,
            "name": name,
            "description": description,
            "status": status,
            "domainName": domain_name
        })
        print(f"Process added: {name} (ID: {process_id})")
    
    def add_subprocess(self, process_id, name, description, parent_process_id, status="Active"):
        """Add a subprocess to the graph"""
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
        
        WITH p
        
        MATCH (parent:Process {processId: $parentProcessId})
        MERGE (p)-[:PART_OF]->(parent)
        
        RETURN p
        """
        self.connector.execute_query(query, {
            "processId": process_id,
            "name": name,
            "description": description,
            "status": status,
            "parentProcessId": parent_process_id
        })
        print(f"Subprocess added: {name} (ID: {process_id}, Parent: {parent_process_id})")
    
    def add_process_dependency(self, process_id, depends_on_process_id):
        """Add a dependency between processes"""
        query = """
        MATCH (p1:Process {processId: $processId})
        MATCH (p2:Process {processId: $dependsOnProcessId})
        MERGE (p1)-[:DEPENDS_ON]->(p2)
        """
        self.connector.execute_query(query, {
            "processId": process_id,
            "dependsOnProcessId": depends_on_process_id
        })
        print(f"Process dependency added: {process_id} depends on {depends_on_process_id}")
    
    def generate_process_hierarchy_with_llm(self, domain):
        """Generate a process hierarchy for a domain using LLM"""
        prompt = f"""
        Create a comprehensive, multi-level process hierarchy for the {domain} domain.
        
        Include:
        1. Top-level business processes in the {domain} domain
        2. For each top-level process, include 3-5 sub-processes
        3. For each sub-process, include 2-4 additional sub-processes
        4. Add brief descriptions for each process
        5. Include 5-10 logical dependencies between processes
        
        Format the output as a JSON object with the following structure:
        {{
            "domain": {{
                "name": "{domain}",
                "description": "Description of the {domain} domain"
            }},
            "processes": [
                {{
                    "processId": "PROC-001",
                    "name": "Top-level Process Name",
                    "description": "Description of the process",
                    "status": "Active",
                    "subprocesses": [
                        {{
                            "processId": "PROC-001-001",
                            "name": "Subprocess Name",
                            "description": "Description of the subprocess",
                            "status": "Active",
                            "subprocesses": [
                                {{
                                    "processId": "PROC-001-001-001",
                                    "name": "Sub-subprocess Name",
                                    "description": "Description of the sub-subprocess",
                                    "status": "Active"
                                }}
                            ]
                        }}
                    ]
                }}
            ],
            "dependencies": [
                {{
                    "source": "PROC-001-001",
                    "target": "PROC-002-001",
                    "description": "Why this dependency exists"
                }}
            ]
        }}
        
        Make sure the process IDs are consistent and hierarchical (e.g., PROC-001, PROC-001-001, etc.).
        Use sequential IDs for each domain, starting from PROC-001, PROC-002, etc. for top-level processes.
        """
        
        print(f"Generating process hierarchy for {domain} domain...")
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in business process management and knowledge graphs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        try:
            # Extract JSON from the response
            content = response.choices[0].message.content
            
            # Handle potential markdown formatting in the response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            process_hierarchy = json.loads(json_str)
            
            # Save the raw hierarchy to a file for reference
            os.makedirs('output', exist_ok=True)
            with open(f"output/{domain.lower().replace(' ', '_')}_hierarchy.json", 'w') as f:
                json.dump(process_hierarchy, f, indent=2)
            
            return process_hierarchy
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            return None
    
    def populate_domain_process_hierarchy(self, domain_name):
        """Populate the graph with a process hierarchy for a domain"""
        process_hierarchy = self.generate_process_hierarchy_with_llm(domain_name)
        
        if process_hierarchy:
            # Add domain
            domain = process_hierarchy["domain"]
            self.add_domain(domain["name"], domain["description"])
            
            # Add top-level processes and their subprocesses
            for process in process_hierarchy["processes"]:
                self.add_process(
                    process["processId"],
                    process["name"],
                    process["description"],
                    domain["name"],
                    process.get("status", "Active")
                )
                
                # Add subprocesses (level 2)
                if "subprocesses" in process:
                    for subprocess in process["subprocesses"]:
                        self.add_subprocess(
                            subprocess["processId"],
                            subprocess["name"],
                            subprocess["description"],
                            process["processId"],
                            subprocess.get("status", "Active")
                        )
                        
                        # Add subprocesses (level 3)
                        if "subprocesses" in subprocess:
                            for subsubprocess in subprocess["subprocesses"]:
                                self.add_subprocess(
                                    subsubprocess["processId"],
                                    subsubprocess["name"],
                                    subsubprocess["description"],
                                    subprocess["processId"],
                                    subsubprocess.get("status", "Active")
                                )
            
            # Add process dependencies
            if "dependencies" in process_hierarchy:
                for dependency in process_hierarchy["dependencies"]:
                    self.add_process_dependency(dependency["source"], dependency["target"])
            
            print(f"Process hierarchy for {domain_name} domain populated successfully")
        else:
            print(f"Failed to generate process hierarchy for {domain_name} domain")
    
    def populate_multiple_domains(self, domains):
        """Populate the graph with process hierarchies for multiple domains"""
        for domain in domains:
            self.populate_domain_process_hierarchy(domain)
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


def main():
    """Main function to populate the process hierarchy"""
    print("=== AgentKG Process Hierarchy Populator ===\n")
    
    # Set up the populator
    populator = ProcessHierarchyPopulator()
    
    # Create constraints
    populator.create_constraints()
    
    # Ask if user wants to clear the database
    should_clear = input("Do you want to clear the database before populating? (yes/no): ")
    if should_clear.lower() == 'yes':
        populator.clear_database()
    
    # Define domains to populate
    business_domains = [
        "Retail",
        "Supply Chain",
        "Manufacturing",
        "Customer Service",
        "Human Resources",
        "Finance",
        "Marketing",
        "Information Technology",
        "Research and Development"
    ]
    
    # Let the user select domains or use all
    print("\nAvailable domains:")
    for i, domain in enumerate(business_domains, 1):
        print(f"{i}. {domain}")
    
    domain_input = input("\nEnter domain numbers to populate (comma-separated) or 'all' for all domains: ")
    
    selected_domains = []
    if domain_input.lower() == 'all':
        selected_domains = business_domains
    else:
        try:
            domain_indices = [int(idx.strip()) - 1 for idx in domain_input.split(',')]
            selected_domains = [business_domains[idx] for idx in domain_indices if 0 <= idx < len(business_domains)]
        except:
            print("Invalid input. Using all domains.")
            selected_domains = business_domains
    
    print(f"\nPopulating {len(selected_domains)} domains: {', '.join(selected_domains)}\n")
    
    # Populate the graph with process hierarchies for each selected domain
    populator.populate_multiple_domains(selected_domains)
    
    # Close the connection
    populator.close()
    
    print("\nProcess hierarchy population complete.")
    print("The Neo4j graph now contains detailed process hierarchies for the selected business domains.")


if __name__ == "__main__":
    main()