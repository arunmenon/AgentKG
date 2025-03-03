"""
Schema Visualizer for AgentKG.

This script provides a textual visualization of the Neo4j schema without requiring
an actual Neo4j database connection.
"""

class SchemaVisualizer:
    """
    Class to visualize the Neo4j schema design
    """
    
    def __init__(self):
        """Initialize the schema visualizer"""
        pass
    
    def print_node_labels(self):
        """Print the node labels and their properties"""
        print("\n=== Node Labels and Properties ===\n")
        
        nodes = [
            {
                "label": "Process",
                "description": "Represents a business process or sub-process",
                "properties": [
                    {"name": "processId", "type": "string", "description": "Unique identifier for the process"},
                    {"name": "name", "type": "string", "description": "Name of the process"},
                    {"name": "description", "type": "string", "description": "Description of the process"},
                    {"name": "status", "type": "string", "description": "Status of the process (Active, Deprecated, etc.)"}
                ]
            },
            {
                "label": "Agent",
                "description": "Represents an individual agent (human or AI)",
                "properties": [
                    {"name": "agentId", "type": "string", "description": "Unique identifier for the agent"},
                    {"name": "name", "type": "string", "description": "Name of the agent"},
                    {"name": "title", "type": "string", "description": "Job title of the agent"},
                    {"name": "performance_metrics", "type": "json", "description": "Current performance metrics (JSON)"}
                ]
            },
            {
                "label": "Crew",
                "description": "Represents a team or group of agents",
                "properties": [
                    {"name": "crewId", "type": "string", "description": "Unique identifier for the crew"},
                    {"name": "name", "type": "string", "description": "Name of the crew"},
                    {"name": "performance_metrics", "type": "json", "description": "Current performance metrics (JSON)"}
                ]
            },
            {
                "label": "BusinessFunction",
                "description": "Represents a business function or role category",
                "properties": [
                    {"name": "name", "type": "string", "description": "Name of the business function"}
                ]
            },
            {
                "label": "TechExpertise",
                "description": "Represents a technical specialty or skill category",
                "properties": [
                    {"name": "name", "type": "string", "description": "Name of the technical expertise"}
                ]
            },
            {
                "label": "Domain",
                "description": "Represents a top-level or sub-level domain area",
                "properties": [
                    {"name": "name", "type": "string", "description": "Name of the domain"},
                    {"name": "description", "type": "string", "description": "Description of the domain"}
                ]
            },
            {
                "label": "Task",
                "description": "Operational tasks within a process",
                "properties": [
                    {"name": "taskId", "type": "string", "description": "Unique identifier for the task"},
                    {"name": "title", "type": "string", "description": "Title or short description of the task"},
                    {"name": "status", "type": "string", "description": "Status of the task (Pending, In-Progress, etc.)"},
                    {"name": "priority", "type": "string", "description": "Priority of the task (High, Medium, Low)"}
                ]
            },
            {
                "label": "PerformanceRecord",
                "description": "Historical performance records for agents or crews",
                "properties": [
                    {"name": "recordId", "type": "string", "description": "Unique identifier for the record"},
                    {"name": "date", "type": "date", "description": "Date of the performance record"},
                    {"name": "metrics", "type": "json", "description": "Performance metrics for the record (JSON)"}
                ]
            }
        ]
        
        # Print each node label and its properties
        for node in nodes:
            print(f"Node Label: {node['label']}")
            print(f"  Description: {node['description']}")
            print("  Properties:")
            for prop in node['properties']:
                print(f"    - {prop['name']} ({prop['type']}): {prop['description']}")
            print()
    
    def print_relationships(self):
        """Print the relationship types and their properties"""
        print("\n=== Relationship Types ===\n")
        
        relationships = [
            {
                "type": "PART_OF",
                "source": "Process",
                "target": "Process",
                "description": "Sub-process is part of a larger parent process",
                "properties": [
                    {"name": "order", "type": "integer", "description": "Optional ordering among siblings"}
                ]
            },
            {
                "type": "DEPENDS_ON",
                "source": "Process",
                "target": "Process",
                "description": "One process depends on or must be completed before another",
                "properties": []
            },
            {
                "type": "SUPPORTS",
                "source": "Crew",
                "target": "Process",
                "description": "A crew is responsible for or actively supports a process",
                "properties": [
                    {"name": "levelOfOwnership", "type": "string", "description": "Level of ownership (Primary, Secondary, etc.)"}
                ]
            },
            {
                "type": "SUPPORTS",
                "source": "Agent",
                "target": "Process",
                "description": "An agent is directly involved in or responsible for a process",
                "properties": [
                    {"name": "timeAllocated", "type": "float", "description": "Time allocated to the process (optional)"},
                    {"name": "startDate", "type": "date", "description": "Start date of the support (optional)"},
                    {"name": "endDate", "type": "date", "description": "End date of the support (optional)"}
                ]
            },
            {
                "type": "MEMBER_OF",
                "source": "Agent",
                "target": "Crew",
                "description": "An agent is a member of a particular crew",
                "properties": [
                    {"name": "role", "type": "string", "description": "Role within the crew (Manager, Member, etc.)"},
                    {"name": "joinDate", "type": "date", "description": "Date when agent joined the crew (optional)"}
                ]
            },
            {
                "type": "SUBTEAM_OF",
                "source": "Crew",
                "target": "Crew",
                "description": "A crew is a subteam of a larger crew",
                "properties": []
            },
            {
                "type": "HAS_FUNCTION",
                "source": "Agent",
                "target": "BusinessFunction",
                "description": "Links an agent to one or more business functions",
                "properties": []
            },
            {
                "type": "HAS_EXPERTISE",
                "source": "Agent",
                "target": "TechExpertise",
                "description": "Links an agent to a technical skill category",
                "properties": []
            },
            {
                "type": "IN_DOMAIN",
                "source": "Process",
                "target": "Domain",
                "description": "A process is associated with a particular domain area",
                "properties": []
            },
            {
                "type": "ASSIGNED_TO",
                "source": "Agent",
                "target": "Task",
                "description": "An agent is assigned to handle a specific task",
                "properties": [
                    {"name": "startDate", "type": "date", "description": "Assignment start date (optional)"},
                    {"name": "endDate", "type": "date", "description": "Assignment end date (optional)"},
                    {"name": "statusUpdate", "type": "string", "description": "Latest status update (optional)"}
                ]
            },
            {
                "type": "PART_OF",
                "source": "Task",
                "target": "Process",
                "description": "A task is part of a specific process",
                "properties": []
            },
            {
                "type": "HAS_PERFORMANCE_RECORD",
                "source": "Agent",
                "target": "PerformanceRecord",
                "description": "Links an agent to their performance record",
                "properties": []
            },
            {
                "type": "HAS_PERFORMANCE_RECORD",
                "source": "Crew",
                "target": "PerformanceRecord",
                "description": "Links a crew to their performance record",
                "properties": []
            }
        ]
        
        # Print each relationship type and its properties
        for rel in relationships:
            print(f"Relationship: (:{rel['source']})-[:{rel['type']}]->(:${rel['target']})")
            print(f"  Description: {rel['description']}")
            if rel['properties']:
                print("  Properties:")
                for prop in rel['properties']:
                    print(f"    - {prop['name']} ({prop['type']}): {prop['description']}")
            print()
    
    def print_cypher_constraints(self):
        """Print the Cypher constraints and indexes"""
        print("\n=== Cypher Constraints and Indexes ===\n")
        
        constraints = [
            "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.processId IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.agentId IS UNIQUE",
            "CREATE CONSTRAINT crew_id IF NOT EXISTS FOR (c:Crew) REQUIRE c.crewId IS UNIQUE",
            "CREATE CONSTRAINT business_function_name IF NOT EXISTS FOR (bf:BusinessFunction) REQUIRE bf.name IS UNIQUE",
            "CREATE CONSTRAINT tech_expertise_name IF NOT EXISTS FOR (te:TechExpertise) REQUIRE te.name IS UNIQUE",
            "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.taskId IS UNIQUE",
            "CREATE CONSTRAINT performance_record_id IF NOT EXISTS FOR (pr:PerformanceRecord) REQUIRE pr.recordId IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX process_name_idx IF NOT EXISTS FOR (p:Process) ON (p.name)",
            "CREATE INDEX agent_name_idx IF NOT EXISTS FOR (a:Agent) ON (a.name)",
            "CREATE INDEX crew_name_idx IF NOT EXISTS FOR (c:Crew) ON (c.name)",
            "CREATE INDEX task_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX performance_record_date_idx IF NOT EXISTS FOR (pr:PerformanceRecord) ON (pr.date)"
        ]
        
        print("Constraints:")
        for constraint in constraints:
            print(f"  {constraint}")
        
        print("\nIndexes:")
        for index in indexes:
            print(f"  {index}")
    
    def print_example_cypher_queries(self):
        """Print example Cypher queries"""
        print("\n=== Example Cypher Queries ===\n")
        
        queries = [
            {
                "name": "Find all sub-processes under Catalog",
                "query": """
MATCH (p:Process {name: "Catalog"})<-[:PART_OF*]-(child:Process)
RETURN child.name
"""
            },
            {
                "name": "Which agents have a successRate > 90% in the Pre-Item-Setup process?",
                "query": """
MATCH (a:Agent)-[:SUPPORTS]->(p:Process {name: "Pre-Item-Setup"})
WHERE a.performance_metrics->>'$.successRate' > 90
RETURN a.name, a.performance_metrics->>'$.successRate' as successRate
"""
            },
            {
                "name": "Which crew is responsible for the greatest number of tasks in Item Maintenance?",
                "query": """
MATCH (crew:Crew)-[:SUPPORTS]->(proc:Process {name: "Item Maintenance"})
OPTIONAL MATCH (crew)-[:ASSIGNED_TO]->(t:Task) 
RETURN crew.name, count(t) AS totalTasks
ORDER BY totalTasks DESC
LIMIT 1
"""
            },
            {
                "name": "Who are the top Data Scientists supporting the Logistics process?",
                "query": """
MATCH (agent:Agent)-[:HAS_EXPERTISE]->(tech:TechExpertise {name:"Data Scientist"}),
      (agent)-[:SUPPORTS]->(proc:Process {name:"Logistics"})
RETURN agent.name, agent.performance_metrics
ORDER BY agent.performance_metrics->>'$.successRate' DESC
"""
            },
            {
                "name": "Get the complete process hierarchy with dependencies",
                "query": """
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
            },
            {
                "name": "Get agent details with their functions, expertise, and assignments",
                "query": """
MATCH (a:Agent)
WHERE a.name = "Alice"
OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(bf:BusinessFunction)
OPTIONAL MATCH (a)-[:HAS_EXPERTISE]->(te:TechExpertise)
OPTIONAL MATCH (a)-[:SUPPORTS]->(p:Process)
OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
OPTIONAL MATCH (a)-[:HAS_PERFORMANCE_RECORD]->(pr:PerformanceRecord)
RETURN 
    a.agentId AS id,
    a.name AS name,
    a.title AS title,
    a.performance_metrics AS metrics,
    COLLECT(DISTINCT bf.name) AS business_functions,
    COLLECT(DISTINCT te.name) AS tech_expertise,
    COLLECT(DISTINCT p.name) AS processes_supported,
    COLLECT(DISTINCT c.name) AS crews,
    COLLECT(DISTINCT {
        recordId: pr.recordId,
        date: pr.date,
        metrics: pr.metrics
    }) AS performance_records
"""
            }
        ]
        
        for query in queries:
            print(f"Query: {query['name']}")
            print(f"{query['query']}")
            print()
    
    def print_schema_summary(self):
        """Print a concise summary of the schema"""
        print("\n=== Schema Summary ===\n")
        
        print("Node Labels:")
        print("  1. Process")
        print("  2. Agent")
        print("  3. Crew")
        print("  4. BusinessFunction")
        print("  5. TechExpertise")
        print("  6. Domain")
        print("  7. Task")
        print("  8. PerformanceRecord")
        
        print("\nRelationship Types:")
        print("  1. (:Process)-[:PART_OF]->(:Process)")
        print("  2. (:Process)-[:DEPENDS_ON]->(:Process)")
        print("  3. (:Crew)-[:SUPPORTS]->(:Process)")
        print("  4. (:Agent)-[:SUPPORTS]->(:Process)")
        print("  5. (:Agent)-[:MEMBER_OF {role:...}]->(:Crew)")
        print("  6. (:Crew)-[:SUBTEAM_OF]->(:Crew)")
        print("  7. (:Agent)-[:HAS_FUNCTION]->(:BusinessFunction)")
        print("  8. (:Agent)-[:HAS_EXPERTISE]->(:TechExpertise)")
        print("  9. (:Process)-[:IN_DOMAIN]->(:Domain)")
        print("  10. (:Agent)-[:ASSIGNED_TO]->(:Task)")
        print("  11. (:Task)-[:PART_OF]->(:Process)")
        print("  12. (:Agent)-[:HAS_PERFORMANCE_RECORD]->(:PerformanceRecord)")
        print("  13. (:Crew)-[:HAS_PERFORMANCE_RECORD]->(:PerformanceRecord)")
    
    def visualize_schema(self):
        """Visualize the complete schema"""
        print("\n===========================")
        print("AgentKG Neo4j Schema Design")
        print("===========================")
        
        self.print_schema_summary()
        self.print_node_labels()
        self.print_relationships()
        self.print_cypher_constraints()
        self.print_example_cypher_queries()


if __name__ == "__main__":
    visualizer = SchemaVisualizer()
    visualizer.visualize_schema()