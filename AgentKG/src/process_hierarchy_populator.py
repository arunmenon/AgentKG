"""
Process Hierarchy Populator for AgentKG.

This script populates the Neo4j database with detailed process hierarchies
based on common business domains.
"""

import os
import json
from .neo4j_connector import Neo4jConnector
import openai
from dotenv import load_dotenv

class ProcessHierarchyPopulator:
    """
    Class responsible for populating the Neo4j database with detailed process hierarchies
    """
    
    def __init__(self):
        """Initialize the Neo4j connector and OpenAI client"""
        load_dotenv()
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
    # Set up the populator
    populator = ProcessHierarchyPopulator()
    
    # Create constraints
    populator.create_constraints()
    
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
    
    # Populate the graph with process hierarchies for each domain
    populator.populate_multiple_domains(business_domains)
    
    # Close the connection
    populator.close()
    
    print("\nProcess hierarchy population complete.")
    print("The Neo4j graph now contains detailed process hierarchies for multiple business domains.")


if __name__ == "__main__":
    main()