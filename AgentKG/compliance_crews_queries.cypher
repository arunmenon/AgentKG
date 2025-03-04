// COMPLIANCE CREWS CYPHER QUERIES
// These queries help explore and visualize the relationship between
// compliance processes and their associated crews and agents.

// 1. Complete Visualization of All Crews and their Agents
// This query returns all crews, their agents, and the processes they handle
MATCH (c:Crew)
MATCH (a:Agent)-[:MEMBER_OF]->(c)
MATCH (c)-[:HANDLES]->(p:Process)
RETURN c, a, p;

// 2. Content Moderation Crew Visualization
// Visualize the Content Moderation crew, its agents, and the process it handles
MATCH (c:Crew {crewId: 'CREW-CM-001'})
MATCH (a:Agent)-[:MEMBER_OF]->(c)
MATCH (c)-[:HANDLES]->(p:Process)
OPTIONAL MATCH (p)-[:PART_OF*]->(parent:Process)
RETURN c, a, p, parent;

// 3. Fraud Prevention Crew Visualization
// Visualize the Fraud Prevention crew, its agents, and the process it handles
MATCH (c:Crew {crewId: 'CREW-FP-001'})
MATCH (a:Agent)-[:MEMBER_OF]->(c)
MATCH (c)-[:HANDLES]->(p:Process)
OPTIONAL MATCH (p)-[:PART_OF*]->(parent:Process)
RETURN c, a, p, parent;

// 4. Product Safety Crew Visualization
// Visualize the Product Safety crew, its agents, and the process it handles
MATCH (c:Crew {crewId: 'CREW-PS-001'})
MATCH (a:Agent)-[:MEMBER_OF]->(c)
MATCH (c)-[:HANDLES]->(p:Process)
OPTIONAL MATCH (p)-[:PART_OF*]->(parent:Process)
RETURN c, a, p, parent;

// 5. Agent Capabilities Overview
// List all agents and their capabilities in a tabular format
MATCH (a:Agent)
OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
RETURN 
    a.agentId as AgentID,
    a.name as AgentName,
    a.role as Role,
    a.goal as Goal,
    a.capabilities as Capabilities,
    collect(DISTINCT c.name) as MemberOfCrews
ORDER BY AgentID;

// 6. Cross-Process Agent Capabilities
// Find capabilities that exist across multiple crews
MATCH (a:Agent)-[:MEMBER_OF]->(c:Crew)
MATCH (c)-[:HANDLES]->(p:Process)
WITH a.capabilities as caps
UNWIND caps as capability
RETURN 
    capability as Capability,
    count(DISTINCT capability) as Frequency
ORDER BY Frequency DESC;

// 7. Find Process and its Crew Handlers
// Given a process ID, find the crew that handles it and all agents
MATCH (p:Process {processId: 'RETAIL-COMPLIANCE-001-002'}) // Change this ID as needed
OPTIONAL MATCH (c:Crew)-[:HANDLES]->(p)
OPTIONAL MATCH (a:Agent)-[:MEMBER_OF]->(c)
RETURN 
    p.name as Process,
    p.description as ProcessDescription,
    c.name as Crew,
    collect(DISTINCT a.name) as Agents;

// 8. Agents by Capability
// Find agents that have a specific capability
MATCH (a:Agent)
WHERE "Regulatory compliance verification" IN a.capabilities // Change capability as needed
OPTIONAL MATCH (a)-[:MEMBER_OF]->(c:Crew)
OPTIONAL MATCH (c)-[:HANDLES]->(p:Process)
RETURN 
    a.name as Agent,
    a.role as Role,
    c.name as Crew,
    p.name as HandlesProcess;

// 9. Process Hierarchy with Crews
// Show the full process hierarchy with assigned crews
MATCH (domain:Domain {name: 'Retail'})
MATCH (p:Process)-[:IN_DOMAIN]->(domain)
WHERE p.processId STARTS WITH 'RETAIL-COMPLIANCE'
OPTIONAL MATCH (sub:Process)-[:PART_OF*0..3]->(p)
OPTIONAL MATCH (c:Crew)-[:HANDLES]->(sub)
RETURN 
    p.name as TopLevelProcess,
    sub.name as Process,
    sub.processId as ProcessID,
    c.name as AssignedCrew;

// 10. Crew Coverage Analysis
// Find processes that don't have an assigned crew yet
MATCH (p:Process)
WHERE p.processId STARTS WITH 'RETAIL-COMPLIANCE'
OPTIONAL MATCH (c:Crew)-[:HANDLES]->(p)
WITH p, c
WHERE c IS NULL
RETURN 
    p.processId as ProcessID,
    p.name as Process,
    p.description as Description,
    "No Crew Assigned" as Status;