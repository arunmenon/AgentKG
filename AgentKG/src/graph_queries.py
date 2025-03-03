from .neo4j_connector import Neo4jConnector

class GraphQueries:
    """
    Utility class for running common graph queries
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
        OPTIONAL MATCH (c)-[:HAS_PERFORMANCE_RECORD]->(pr:PerformanceRecord)
        RETURN 
            c.crewId AS id,
            c.name AS name,
            parent.name AS parent_crew,
            COLLECT(DISTINCT p.name) AS processes_supported,
            COLLECT(DISTINCT member.name) AS members,
            c.performance_metrics AS metrics,
            COLLECT(DISTINCT {
                recordId: pr.recordId,
                date: pr.date,
                metrics: pr.metrics
            }) AS performance_records
        ORDER BY 
            CASE WHEN parent.name IS NULL THEN 0 ELSE 1 END,
            parent.name,
            c.name
        """
        
        return self.connector.execute_query(query)
    
    def get_tasks(self, process_name=None, status=None):
        """
        Get tasks with their assignments and process contexts
        
        Args:
            process_name (str, optional): Filter by process name
            status (str, optional): Filter by task status
            
        Returns:
            list: Task data
        """
        query = """
        MATCH (t:Task)
        MATCH (t)-[:PART_OF]->(p:Process)
        OPTIONAL MATCH (a:Agent)-[:ASSIGNED_TO]->(t)
        WHERE 
            ($process_name IS NULL OR p.name = $process_name) AND
            ($status IS NULL OR t.status = $status)
        RETURN 
            t.taskId AS id,
            t.title AS title,
            t.status AS status,
            t.priority AS priority,
            p.name AS process,
            COLLECT(DISTINCT a.name) AS assigned_agents
        ORDER BY 
            CASE t.priority
                WHEN 'High' THEN 0
                WHEN 'Medium' THEN 1
                WHEN 'Low' THEN 2
                ELSE 3
            END,
            t.status
        """
        
        return self.connector.execute_query(query, {
            "process_name": process_name,
            "status": status
        })
    
    def get_performance_records(self, entity_name=None, entity_type=None, start_date=None, end_date=None):
        """
        Get performance records for agents or crews within a date range
        
        Args:
            entity_name (str, optional): Filter by agent or crew name
            entity_type (str, optional): Filter by entity type ('Agent' or 'Crew')
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            list: Performance record data
        """
        query = """
        MATCH (pr:PerformanceRecord)
        MATCH (entity)-[:HAS_PERFORMANCE_RECORD]->(pr)
        WHERE 
            ($entity_type IS NULL OR $entity_type = 'Agent' AND entity:Agent OR $entity_type = 'Crew' AND entity:Crew) AND
            ($entity_name IS NULL OR entity.name = $entity_name) AND
            ($start_date IS NULL OR pr.date >= date($start_date)) AND
            ($end_date IS NULL OR pr.date <= date($end_date))
        RETURN 
            pr.recordId AS id,
            pr.date AS date,
            pr.metrics AS metrics,
            entity.name AS entity_name,
            CASE 
                WHEN entity:Agent THEN 'Agent'
                WHEN entity:Crew THEN 'Crew'
                ELSE NULL
            END AS entity_type
        ORDER BY pr.date DESC
        """
        
        return self.connector.execute_query(query, {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "start_date": start_date,
            "end_date": end_date
        })
    
    def find_experts_for_process(self, process_name):
        """
        Find the best agents for a specific process based on expertise and performance
        
        Args:
            process_name (str): The name of the process
            
        Returns:
            list: Matching agents with their qualifications
        """
        query = """
        MATCH (p:Process {name: $process_name})
        OPTIONAL MATCH (p)-[:IN_DOMAIN]->(d:Domain)
        
        // Find agents directly supporting the process
        OPTIONAL MATCH (direct_agent:Agent)-[:SUPPORTS]->(p)
        
        // Find agents with relevant crews
        OPTIONAL MATCH (crew_agent:Agent)-[:MEMBER_OF]->(c:Crew)-[:SUPPORTS]->(p)
        
        // Find agents with relevant functions
        MATCH (func_agent:Agent)-[:HAS_FUNCTION]->(bf:BusinessFunction)
        WHERE func_agent = direct_agent OR func_agent = crew_agent
        
        // Find agents with relevant expertise
        OPTIONAL MATCH (func_agent)-[:HAS_EXPERTISE]->(te:TechExpertise)
        
        // Get performance metrics if available
        OPTIONAL MATCH (func_agent)-[:HAS_PERFORMANCE_RECORD]->(pr:PerformanceRecord)
        
        RETURN 
            func_agent.agentId AS id,
            func_agent.name AS name,
            func_agent.title AS title,
            func_agent.performance_metrics AS metrics,
            CASE 
                WHEN direct_agent IS NOT NULL THEN true 
                ELSE false 
            END AS direct_support,
            COLLECT(DISTINCT bf.name) AS business_functions,
            COLLECT(DISTINCT te.name) AS tech_expertise,
            COLLECT(DISTINCT c.name) AS supporting_crews,
            COLLECT(DISTINCT {
                recordId: pr.recordId,
                date: pr.date,
                metrics: pr.metrics
            }) AS performance_records
        ORDER BY 
            direct_support DESC,
            size(COLLECT(DISTINCT bf.name)) DESC,
            size(COLLECT(DISTINCT te.name)) DESC
        """
        
        return self.connector.execute_query(query, {"process_name": process_name})
    
    def get_process_dependencies(self, process_name):
        """
        Get all dependencies for a specific process (both direct and indirect)
        
        Args:
            process_name (str): The name of the process
            
        Returns:
            list: Dependency chain for the process
        """
        query = """
        MATCH (p:Process {name: $process_name})
        MATCH (p)-[:DEPENDS_ON*1..]->(dep:Process)
        RETURN 
            p.name AS process,
            dep.name AS dependency,
            length(shortestPath((p)-[:DEPENDS_ON*1..]->(dep))) AS dependency_level
        ORDER BY dependency_level
        """
        
        return self.connector.execute_query(query, {"process_name": process_name})
    
    def get_process_tasks(self, process_name):
        """
        Get all tasks associated with a specific process
        
        Args:
            process_name (str): The name of the process
            
        Returns:
            list: Tasks for the process
        """
        query = """
        MATCH (p:Process {name: $process_name})
        MATCH (t:Task)-[:PART_OF]->(p)
        OPTIONAL MATCH (agent:Agent)-[:ASSIGNED_TO]->(t)
        RETURN 
            t.taskId AS id,
            t.title AS title,
            t.status AS status,
            t.priority AS priority,
            COLLECT(DISTINCT agent.name) AS assigned_agents
        ORDER BY 
            CASE t.priority
                WHEN 'High' THEN 0
                WHEN 'Medium' THEN 1
                WHEN 'Low' THEN 2
                ELSE 3
            END,
            t.status
        """
        
        return self.connector.execute_query(query, {"process_name": process_name})
    
    def get_taxonomy(self):
        """
        Get the complete taxonomy (business functions, tech expertise, domains)
        
        Returns:
            dict: Complete taxonomy structure
        """
        functions_query = "MATCH (bf:BusinessFunction) RETURN bf.name AS name ORDER BY bf.name"
        expertise_query = "MATCH (te:TechExpertise) RETURN te.name AS name ORDER BY te.name"
        domains_query = "MATCH (d:Domain) RETURN d.name AS name ORDER BY d.name"
        
        business_functions = [record["name"] for record in self.connector.execute_query(functions_query)]
        tech_expertise = [record["name"] for record in self.connector.execute_query(expertise_query)]
        domains = [record["name"] for record in self.connector.execute_query(domains_query)]
        
        return {
            "business_functions": business_functions,
            "tech_expertise": tech_expertise,
            "domains": domains
        }
    
    def close(self):
        """Close the Neo4j connection"""
        self.connector.close()


if __name__ == "__main__":
    queries = GraphQueries()
    
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
    
    print("\n=== Agent Details ===")
    agents = queries.get_agent_details()
    for agent in agents:
        print(f"Agent: {agent['name']} ({agent['title']})")
        print(f"  Business Functions: {', '.join(agent['business_functions'])}")
        print(f"  Technical Expertise: {', '.join(agent['tech_expertise'])}")
        print(f"  Processes Supported: {', '.join(agent['processes_supported'])}")
        print(f"  Member of Crews: {', '.join(agent['crews'])}")
        print()
    
    print("\n=== Crew Structure ===")
    crews = queries.get_crew_structure()
    for crew in crews:
        print(f"Crew: {crew['name']}")
        if crew['parent_crew']:
            print(f"  Parent Crew: {crew['parent_crew']}")
        print(f"  Processes Supported: {', '.join(crew['processes_supported'])}")
        print(f"  Members: {', '.join(crew['members'])}")
        print()
    
    print("\n=== Taxonomy ===")
    taxonomy = queries.get_taxonomy()
    print(f"Business Functions: {', '.join(taxonomy['business_functions'])}")
    print(f"Technical Expertise: {', '.join(taxonomy['tech_expertise'])}")
    print(f"Domains: {', '.join(taxonomy['domains'])}")
    
    queries.close()