"""
Knowledge Ingestion Demo for AgentKG.

This script demonstrates how the knowledge ingestion process works,
without requiring an actual connection to Neo4j or OpenAI.
"""

import json
from datetime import datetime


class KnowledgeGraph:
    """Simulated Knowledge Graph structure"""
    
    def __init__(self):
        """Initialize an empty knowledge graph"""
        self.processes = []
        self.agents = []
        self.crews = []
        self.tasks = []
        self.performance_records = []
        self.business_functions = []
        self.tech_expertise = []
        self.domains = []
    
    def add_domain(self, name, description=None):
        """Add a domain to the knowledge graph"""
        self.domains.append({
            "name": name,
            "description": description
        })
        print(f"Added Domain: {name}")
    
    def add_business_function(self, name):
        """Add a business function to the knowledge graph"""
        self.business_functions.append({
            "name": name
        })
        print(f"Added Business Function: {name}")
    
    def add_tech_expertise(self, name):
        """Add a technical expertise to the knowledge graph"""
        self.tech_expertise.append({
            "name": name
        })
        print(f"Added Technical Expertise: {name}")
    
    def add_process(self, process_id, name, description=None, status="Active", parent_process=None, domain=None):
        """Add a process to the knowledge graph"""
        process = {
            "processId": process_id,
            "name": name,
            "description": description,
            "status": status,
            "parent_process": parent_process,
            "dependencies": [],
            "domain": domain
        }
        self.processes.append(process)
        print(f"Added Process: {name} (ID: {process_id})")
        
        return process
    
    def add_process_dependency(self, process_id, dependency_process_id):
        """Add a dependency between processes"""
        process = next((p for p in self.processes if p["processId"] == process_id), None)
        dependency = next((p for p in self.processes if p["processId"] == dependency_process_id), None)
        
        if process and dependency:
            process["dependencies"].append(dependency_process_id)
            print(f"Added Dependency: {process['name']} depends on {dependency['name']}")
    
    def add_crew(self, crew_id, name, parent_crew=None):
        """Add a crew to the knowledge graph"""
        crew = {
            "crewId": crew_id,
            "name": name,
            "parent_crew": parent_crew,
            "processes_supported": [],
            "performance_metrics": {
                "successRate": 0,
                "tasksCompleted": 0
            }
        }
        self.crews.append(crew)
        print(f"Added Crew: {name} (ID: {crew_id})")
        
        return crew
    
    def add_crew_process_support(self, crew_id, process_id, level_of_ownership="Primary"):
        """Add a relationship indicating a crew supports a process"""
        crew = next((c for c in self.crews if c["crewId"] == crew_id), None)
        process = next((p for p in self.processes if p["processId"] == process_id), None)
        
        if crew and process:
            crew["processes_supported"].append({
                "processId": process_id,
                "levelOfOwnership": level_of_ownership
            })
            print(f"Added Support: Crew '{crew['name']}' supports Process '{process['name']}' as {level_of_ownership}")
    
    def add_agent(self, agent_id, name, title=None):
        """Add an agent to the knowledge graph"""
        agent = {
            "agentId": agent_id,
            "name": name,
            "title": title,
            "business_functions": [],
            "tech_expertise": [],
            "processes_supported": [],
            "crews": [],
            "performance_metrics": {
                "successRate": 0,
                "tasksCompleted": 0,
                "efficiency": 0
            }
        }
        self.agents.append(agent)
        print(f"Added Agent: {name} (ID: {agent_id})")
        
        return agent
    
    def add_agent_business_function(self, agent_id, business_function):
        """Add a business function to an agent"""
        agent = next((a for a in self.agents if a["agentId"] == agent_id), None)
        
        if agent:
            if business_function not in self.business_functions:
                self.add_business_function(business_function)
            
            agent["business_functions"].append(business_function)
            print(f"Added Business Function: Agent '{agent['name']}' has function '{business_function}'")
    
    def add_agent_tech_expertise(self, agent_id, tech_expertise):
        """Add a technical expertise to an agent"""
        agent = next((a for a in self.agents if a["agentId"] == agent_id), None)
        
        if agent:
            if tech_expertise not in self.tech_expertise:
                self.add_tech_expertise(tech_expertise)
            
            agent["tech_expertise"].append(tech_expertise)
            print(f"Added Technical Expertise: Agent '{agent['name']}' has expertise '{tech_expertise}'")
    
    def add_agent_to_crew(self, agent_id, crew_id, role="Member"):
        """Add an agent to a crew"""
        agent = next((a for a in self.agents if a["agentId"] == agent_id), None)
        crew = next((c for c in self.crews if c["crewId"] == crew_id), None)
        
        if agent and crew:
            agent["crews"].append({
                "crewId": crew_id,
                "role": role
            })
            print(f"Added Crew Membership: Agent '{agent['name']}' is a {role} of Crew '{crew['name']}'")
    
    def add_agent_process_support(self, agent_id, process_id, time_allocated=None):
        """Add a relationship indicating an agent supports a process"""
        agent = next((a for a in self.agents if a["agentId"] == agent_id), None)
        process = next((p for p in self.processes if p["processId"] == process_id), None)
        
        if agent and process:
            agent["processes_supported"].append({
                "processId": process_id,
                "timeAllocated": time_allocated
            })
            print(f"Added Support: Agent '{agent['name']}' supports Process '{process['name']}'")
    
    def add_task(self, task_id, title, process_id, status="Pending", priority="Medium"):
        """Add a task to the knowledge graph"""
        task = {
            "taskId": task_id,
            "title": title,
            "status": status,
            "priority": priority,
            "process": process_id,
            "assigned_agents": []
        }
        self.tasks.append(task)
        print(f"Added Task: {title} (ID: {task_id})")
        
        return task
    
    def assign_agent_to_task(self, agent_id, task_id):
        """Assign an agent to a task"""
        agent = next((a for a in self.agents if a["agentId"] == agent_id), None)
        task = next((t for t in self.tasks if t["taskId"] == task_id), None)
        
        if agent and task:
            task["assigned_agents"].append(agent_id)
            print(f"Added Assignment: Agent '{agent['name']}' assigned to Task '{task['title']}'")
    
    def add_performance_record(self, record_id, entity_id, entity_type, date, metrics):
        """Add a performance record for an agent or crew"""
        record = {
            "recordId": record_id,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "date": date,
            "metrics": metrics
        }
        self.performance_records.append(record)
        
        entity_name = ""
        if entity_type == "Agent":
            entity = next((a for a in self.agents if a["agentId"] == entity_id), None)
            if entity:
                entity_name = entity['name']
        elif entity_type == "Crew":
            entity = next((c for c in self.crews if c["crewId"] == entity_id), None)
            if entity:
                entity_name = entity['name']
        
        print(f"Added Performance Record: For {entity_type} '{entity_name}' on {date}")
    
    def to_dict(self):
        """Convert the knowledge graph to a dictionary"""
        return {
            "processes": self.processes,
            "agents": self.agents,
            "crews": self.crews,
            "tasks": self.tasks,
            "performance_records": self.performance_records,
            "business_functions": [{"name": bf} for bf in self.business_functions],
            "tech_expertise": [{"name": te} for te in self.tech_expertise],
            "domains": self.domains
        }
    
    def to_json(self, indent=2):
        """Convert the knowledge graph to JSON"""
        return json.dumps(self.to_dict(), indent=indent)


class MockLLMParser:
    """
    Mock LLM parser that simulates extracting structured knowledge from text.
    """
    
    def parse_description(self, description):
        """
        Simulate parsing a text description into structured knowledge.
        This would normally be done by an LLM.
        
        Args:
            description (str): Text description to parse
            
        Returns:
            KnowledgeGraph: Structured knowledge graph
        """
        print("=== Simulating LLM Knowledge Extraction ===")
        
        kg = KnowledgeGraph()
        
        # In a real implementation, an LLM would parse the description and extract entities
        # For this demo, we'll manually create a knowledge graph based on the example provided
        
        # Add domains
        kg.add_domain("Retail", "Retail business domain")
        
        # Add business functions
        business_functions = [
            "Retail Merchandising", 
            "Catalog Management", 
            "Item Setup", 
            "Data Entry",
            "Supply Chain Management", 
            "Logistics Planning",
            "Price Analysis", 
            "Competitive Intelligence"
        ]
        for bf in business_functions:
            kg.add_business_function(bf)
        
        # Add technical expertise
        tech_expertise = [
            "Data Analysis", 
            "Product Information Management", 
            "Database Management", 
            "Data Validation",
            "Inventory Systems", 
            "Route Optimization",
            "Machine Learning", 
            "Price Optimization Algorithms"
        ]
        for te in tech_expertise:
            kg.add_tech_expertise(te)
        
        # Add processes
        catalog_mgmt = kg.add_process("PROC-001", "Catalog Management", "Top-level process for managing product information", domain="Retail")
        pre_item_setup = kg.add_process("PROC-002", "Pre-Item-Setup", "Process for setting up new items before they go live", parent_process="PROC-001")
        item_maintenance = kg.add_process("PROC-003", "Item Maintenance", "Process for maintaining existing items", parent_process="PROC-001")
        pricing_mgmt = kg.add_process("PROC-004", "Pricing Management", "Process for managing product pricing", parent_process="PROC-001")
        
        supply_chain = kg.add_process("PROC-005", "Supply Chain Management", "Process for managing the flow of goods and services", domain="Retail")
        inventory_mgmt = kg.add_process("PROC-006", "Inventory Management", "Process for tracking and managing inventory", parent_process="PROC-005")
        order_fulfillment = kg.add_process("PROC-007", "Order Fulfillment", "Process for fulfilling customer orders", parent_process="PROC-005")
        logistics = kg.add_process("PROC-008", "Logistics", "Process for managing transportation and delivery", parent_process="PROC-005")
        
        # Add process dependencies
        kg.add_process_dependency("PROC-003", "PROC-002")  # Item Maintenance depends on Pre-Item-Setup
        kg.add_process_dependency("PROC-007", "PROC-006")  # Order Fulfillment depends on Inventory Management
        
        # Add crews
        catalog_team = kg.add_crew("CREW-001", "Catalog Team")
        item_setup_crew = kg.add_crew("CREW-002", "Item Setup Crew", parent_crew="CREW-001")
        item_maintenance_crew = kg.add_crew("CREW-003", "Item Maintenance Crew", parent_crew="CREW-001")
        pricing_crew = kg.add_crew("CREW-004", "Pricing Optimization Crew", parent_crew="CREW-001")
        
        supply_chain_team = kg.add_crew("CREW-005", "Supply Chain Team")
        inventory_crew = kg.add_crew("CREW-006", "Inventory Control Crew", parent_crew="CREW-005")
        fulfillment_crew = kg.add_crew("CREW-007", "Fulfillment Crew", parent_crew="CREW-005")
        logistics_crew = kg.add_crew("CREW-008", "Logistics Crew", parent_crew="CREW-005")
        
        # Link crews to processes
        kg.add_crew_process_support("CREW-001", "PROC-001")
        kg.add_crew_process_support("CREW-002", "PROC-002")
        kg.add_crew_process_support("CREW-003", "PROC-003")
        kg.add_crew_process_support("CREW-004", "PROC-004")
        
        kg.add_crew_process_support("CREW-005", "PROC-005")
        kg.add_crew_process_support("CREW-006", "PROC-006")
        kg.add_crew_process_support("CREW-007", "PROC-007")
        kg.add_crew_process_support("CREW-008", "PROC-008")
        
        # Add agents
        alice = kg.add_agent("A001", "Alice", "Senior Catalog Manager")
        bob = kg.add_agent("A002", "Bob", "Item Setup Specialist")
        charlie = kg.add_agent("A003", "Charlie", "Supply Chain Manager")
        diana = kg.add_agent("A004", "Diana", "AI Assistant for Pricing")
        
        # Add agent business functions
        kg.add_agent_business_function("A001", "Retail Merchandising")
        kg.add_agent_business_function("A001", "Catalog Management")
        kg.add_agent_business_function("A002", "Item Setup")
        kg.add_agent_business_function("A002", "Data Entry")
        kg.add_agent_business_function("A003", "Supply Chain Management")
        kg.add_agent_business_function("A003", "Logistics Planning")
        kg.add_agent_business_function("A004", "Price Analysis")
        kg.add_agent_business_function("A004", "Competitive Intelligence")
        
        # Add agent technical expertise
        kg.add_agent_tech_expertise("A001", "Data Analysis")
        kg.add_agent_tech_expertise("A001", "Product Information Management")
        kg.add_agent_tech_expertise("A002", "Database Management")
        kg.add_agent_tech_expertise("A002", "Data Validation")
        kg.add_agent_tech_expertise("A003", "Inventory Systems")
        kg.add_agent_tech_expertise("A003", "Route Optimization")
        kg.add_agent_tech_expertise("A004", "Machine Learning")
        kg.add_agent_tech_expertise("A004", "Price Optimization Algorithms")
        
        # Add agents to crews
        kg.add_agent_to_crew("A001", "CREW-001", "Manager")
        kg.add_agent_to_crew("A002", "CREW-002", "Member")
        kg.add_agent_to_crew("A003", "CREW-005", "Manager")
        kg.add_agent_to_crew("A004", "CREW-004", "Member")
        
        # Link agents to processes
        kg.add_agent_process_support("A001", "PROC-001")
        kg.add_agent_process_support("A002", "PROC-002")
        kg.add_agent_process_support("A003", "PROC-005")
        kg.add_agent_process_support("A004", "PROC-004")
        
        # Add tasks
        task1 = kg.add_task("TASK-001", "Update product descriptions", "PROC-002", "In-Progress", "High")
        task2 = kg.add_task("TASK-002", "Review price changes", "PROC-004", "Pending", "Medium")
        task3 = kg.add_task("TASK-003", "Optimize inventory levels", "PROC-006", "Completed", "High")
        
        # Assign agents to tasks
        kg.assign_agent_to_task("A002", "TASK-001")
        kg.assign_agent_to_task("A004", "TASK-002")
        kg.assign_agent_to_task("A003", "TASK-003")
        
        # Add performance records
        today = datetime.now().strftime("%Y-%m-%d")
        kg.add_performance_record("PR-001", "A001", "Agent", "2023-01-15", {"tasksCompleted": 95, "successRate": 0.95, "efficiency": 0.90})
        kg.add_performance_record("PR-002", "A002", "Agent", "2023-01-15", {"tasksCompleted": 85, "successRate": 0.92, "efficiency": 0.88})
        kg.add_performance_record("PR-003", "A003", "Agent", "2023-01-15", {"tasksCompleted": 150, "successRate": 0.97, "efficiency": 0.95})
        kg.add_performance_record("PR-004", "A004", "Agent", "2023-01-15", {"tasksCompleted": 300, "successRate": 0.99, "efficiency": 0.98})
        
        kg.add_performance_record("PR-005", "A001", "Agent", today, {"tasksCompleted": 120, "successRate": 0.96, "efficiency": 0.92})
        kg.add_performance_record("PR-006", "CREW-001", "Crew", today, {"tasksCompleted": 500, "successRate": 0.94, "efficiency": 0.91})
        
        print("=== Knowledge Extraction Complete ===")
        return kg


def main():
    """Main function to demonstrate knowledge ingestion"""
    print("=== AgentKG Knowledge Ingestion Demo ===\n")
    
    # Example text description that would be processed by an LLM
    description = """
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
    
    # In a real implementation, this would call an LLM to process the description
    parser = MockLLMParser()
    knowledge_graph = parser.parse_description(description)
    
    # Export the knowledge graph to JSON (would be used for Neo4j import)
    json_data = knowledge_graph.to_json()
    
    # Save the knowledge graph to a file
    with open("AgentKG/knowledge_graph.json", "w") as f:
        f.write(json_data)
    
    print("\n=== Knowledge Graph has been created and exported to knowledge_graph.json ===")
    print("\nIn a production system, this structured knowledge would be ingested into Neo4j")
    print("using the Cypher queries and connection methods implemented in the main codebase.")


if __name__ == "__main__":
    main()