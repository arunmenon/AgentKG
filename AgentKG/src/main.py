import argparse
import json
from .schema_creator import SchemaCreator
from .knowledge_ingestion import KnowledgeIngestion
from .agent_augmentation import KnowledgeAugmentation
from .graph_queries import GraphQueries

def init_database():
    """Initialize the database schema"""
    print("Initializing database schema...")
    schema_creator = SchemaCreator()
    schema_creator.create_schema()
    schema_creator.close()
    print("Database schema initialized successfully")

def load_example_data():
    """Load example knowledge graph data"""
    print("Loading example knowledge data...")
    ingestion = KnowledgeIngestion()
    ingestion.create_example_knowledge()
    ingestion.close()
    print("Example data loaded successfully")

def augment_knowledge(topics):
    """
    Augment the knowledge graph with information from search
    
    Args:
        topics (list): List of topics to search for
    """
    print(f"Augmenting knowledge graph with {len(topics)} topics...")
    augmentation = KnowledgeAugmentation()
    
    for topic in topics:
        print(f"\nSearching for: {topic}")
        knowledge = augmentation.augment_knowledge_with_search(topic)
        if knowledge:
            print(f"Added knowledge about: {topic}")
        else:
            print(f"Failed to add knowledge about: {topic}")
    
    augmentation.close()
    print("\nKnowledge augmentation completed")

def query_graph(query_type=None, filter_value=None, second_filter=None):
    """
    Run queries against the knowledge graph
    
    Args:
        query_type (str): Type of query to run
        filter_value (str): Primary filter value for the query
        second_filter (str): Secondary filter value for the query (for tasks and performance records)
    """
    queries = GraphQueries()
    
    if query_type == "processes":
        print("\n=== Process Hierarchy ===")
        process_hierarchy = queries.get_process_hierarchy()
        for process in process_hierarchy:
            print(f"Process: {process['name']}")
            if process['parent_process']:
                print(f"  Parent: {process['parent_process']}")
            if process['dependencies']:
                print(f"  Dependencies: {', '.join(process['dependencies'])}")
            if process['domain']:
                print(f"  Domain: {process['domain']}")
            print()
    
    elif query_type == "agents":
        print("\n=== Agent Details ===")
        agents = queries.get_agent_details(filter_value)
        for agent in agents:
            print(f"Agent: {agent['name']} ({agent['title']})")
            print(f"  Business Functions: {', '.join(agent['business_functions'])}")
            print(f"  Technical Expertise: {', '.join(agent['tech_expertise'])}")
            print(f"  Processes Supported: {', '.join(agent['processes_supported'])}")
            print(f"  Member of Crews: {', '.join(agent['crews'])}")
            
            # Show performance metrics if available
            if agent['metrics']:
                try:
                    metrics = json.loads(agent['metrics'])
                    print(f"  Current Metrics: {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
                except:
                    pass
            
            # Show historical performance records if available
            if agent['performance_records'] and agent['performance_records'][0] and agent['performance_records'][0]['recordId']:
                print("  Performance History:")
                for record in sorted(agent['performance_records'], key=lambda x: x['date'], reverse=True)[:3]:  # Show latest 3
                    try:
                        metrics = json.loads(record['metrics'])
                        print(f"    {record['date']}: {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
                    except:
                        pass
            print()
    
    elif query_type == "crews":
        print("\n=== Crew Structure ===")
        crews = queries.get_crew_structure()
        for crew in crews:
            print(f"Crew: {crew['name']}")
            if crew['parent_crew']:
                print(f"  Parent Crew: {crew['parent_crew']}")
            print(f"  Processes Supported: {', '.join(crew['processes_supported'])}")
            print(f"  Members: {', '.join(crew['members'])}")
            
            # Show performance metrics if available
            if crew['metrics']:
                try:
                    metrics = json.loads(crew['metrics'])
                    print(f"  Current Metrics: {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
                except:
                    pass
            
            # Show historical performance records if available
            if crew['performance_records'] and crew['performance_records'][0] and crew['performance_records'][0]['recordId']:
                print("  Performance History:")
                for record in sorted(crew['performance_records'], key=lambda x: x['date'], reverse=True)[:3]:  # Show latest 3
                    try:
                        metrics = json.loads(record['metrics'])
                        print(f"    {record['date']}: {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
                    except:
                        pass
            print()
    
    elif query_type == "tasks":
        print("\n=== Tasks ===")
        tasks = queries.get_tasks(filter_value, second_filter)
        for task in tasks:
            print(f"Task: {task['title']} (ID: {task['id']})")
            print(f"  Status: {task['status']}")
            print(f"  Priority: {task['priority']}")
            print(f"  Process: {task['process']}")
            print(f"  Assigned Agents: {', '.join(task['assigned_agents']) if task['assigned_agents'] else 'None'}")
            print()
    
    elif query_type == "performance":
        print("\n=== Performance Records ===")
        records = queries.get_performance_records(
            entity_name=filter_value,
            entity_type=second_filter
        )
        for record in records:
            print(f"Record for {record['entity_type']} '{record['entity_name']}' on {record['date']}")
            try:
                metrics = json.loads(record['metrics'])
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            except:
                print(f"  Metrics: {record['metrics']}")
            print()
    
    elif query_type == "experts" and filter_value:
        print(f"\n=== Experts for Process: {filter_value} ===")
        experts = queries.find_experts_for_process(filter_value)
        for expert in experts:
            print(f"Expert: {expert['name']} ({expert['title']})")
            print(f"  Direct Support: {'Yes' if expert['direct_support'] else 'No'}")
            print(f"  Business Functions: {', '.join(expert['business_functions'])}")
            print(f"  Technical Expertise: {', '.join(expert['tech_expertise'])}")
            print(f"  Supporting Crews: {', '.join(expert['supporting_crews'])}")
            
            # Show performance metrics if available
            if expert['metrics']:
                try:
                    metrics = json.loads(expert['metrics'])
                    print(f"  Current Metrics: {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
                except:
                    pass
            print()
    
    elif query_type == "dependencies" and filter_value:
        print(f"\n=== Dependencies for Process: {filter_value} ===")
        dependencies = queries.get_process_dependencies(filter_value)
        for dep in dependencies:
            print(f"Process {dep['process']} depends on {dep['dependency']} (Level: {dep['dependency_level']})")
    
    elif query_type == "process_tasks" and filter_value:
        print(f"\n=== Tasks for Process: {filter_value} ===")
        tasks = queries.get_process_tasks(filter_value)
        for task in tasks:
            print(f"Task: {task['title']} (ID: {task['id']})")
            print(f"  Status: {task['status']}")
            print(f"  Priority: {task['priority']}")
            print(f"  Assigned Agents: {', '.join(task['assigned_agents']) if task['assigned_agents'] else 'None'}")
            print()
    
    elif query_type == "taxonomy":
        print("\n=== Taxonomy ===")
        taxonomy = queries.get_taxonomy()
        print(f"Business Functions: {', '.join(taxonomy['business_functions'])}")
        print(f"Technical Expertise: {', '.join(taxonomy['tech_expertise'])}")
        print(f"Domains: {', '.join(taxonomy['domains'])}")
    
    else:
        # Run all queries
        query_graph("processes")
        query_graph("agents")
        query_graph("crews")
        query_graph("tasks")
        query_graph("taxonomy")
    
    queries.close()

def main():
    """Main function to parse arguments and run appropriate functions"""
    parser = argparse.ArgumentParser(description="AgentKG - Agent Knowledge Graph")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the database schema")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load example knowledge data")
    
    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Augment knowledge with search")
    augment_parser.add_argument("topics", nargs="+", help="Topics to search for")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--type", choices=[
        "processes", "agents", "crews", "tasks", "performance", 
        "experts", "dependencies", "process_tasks", "taxonomy"
    ], help="Type of query to run")
    query_parser.add_argument("--filter", help="Primary filter value for the query")
    query_parser.add_argument("--second-filter", help="Secondary filter value for the query (for tasks and performance records)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate function based on command
    if args.command == "init":
        init_database()
    elif args.command == "load":
        load_example_data()
    elif args.command == "augment":
        augment_knowledge(args.topics)
    elif args.command == "query":
        query_graph(args.type, args.filter, args.second_filter)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()