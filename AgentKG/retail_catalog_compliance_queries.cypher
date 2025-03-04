// RETAIL CATALOG AND COMPLIANCE PROCESS HIERARCHY CYPHER QUERIES
// These queries help explore and visualize the Catalog and Compliance (Trust and Safety)
// process hierarchies under the Retail domain.

// 1. Complete Visualization of Retail Catalog and Compliance Process Hierarchies
// This query returns all processes in both hierarchies with their dependencies for visualization
MATCH (domain:Domain {name: 'Retail'})
MATCH (p:Process)-[:IN_DOMAIN]->(domain)
WHERE p.processId STARTS WITH 'RETAIL-CATALOG' OR p.processId STARTS WITH 'RETAIL-COMPLIANCE'
MATCH (sub:Process)-[:PART_OF*0..3]->(p)
OPTIONAL MATCH (sub)-[:DEPENDS_ON]->(dep:Process)
RETURN sub, p, dep, domain;

// 2. Catalog Process Hierarchy with Three Levels
// Get the complete Catalog hierarchy in a tabular format
MATCH (catalog:Process {processId: 'RETAIL-CATALOG-001'})
OPTIONAL MATCH (level2:Process)-[:PART_OF]->(catalog)
OPTIONAL MATCH (level3:Process)-[:PART_OF]->(level2)
RETURN 
    catalog.name as Catalog,
    level2.name as SubProcess,
    level3.name as DetailedProcess,
    level2.processId as SubProcessID,
    level3.processId as DetailedProcessID
ORDER BY SubProcessID, DetailedProcessID;

// 3. Compliance Process Hierarchy with Three Levels
// Get the complete Compliance hierarchy in a tabular format
MATCH (compliance:Process {processId: 'RETAIL-COMPLIANCE-001'})
OPTIONAL MATCH (level2:Process)-[:PART_OF]->(compliance)
OPTIONAL MATCH (level3:Process)-[:PART_OF]->(level2)
RETURN 
    compliance.name as Compliance,
    level2.name as SubProcess,
    level3.name as DetailedProcess,
    level2.processId as SubProcessID,
    level3.processId as DetailedProcessID
ORDER BY SubProcessID, DetailedProcessID;

// 4. Cross-Area Dependencies
// Find dependencies between Catalog and Compliance processes
MATCH (p1:Process)-[:DEPENDS_ON]->(p2:Process)
WHERE 
    (p1.processId STARTS WITH 'RETAIL-CATALOG' AND p2.processId STARTS WITH 'RETAIL-COMPLIANCE')
    OR 
    (p1.processId STARTS WITH 'RETAIL-COMPLIANCE' AND p2.processId STARTS WITH 'RETAIL-CATALOG')
RETURN 
    p1.name as Process,
    p1.processId as ProcessID,
    'DEPENDS ON' as Relationship,
    p2.name as DependsOn,
    p2.processId as DependsOnID;

// 5. Complete Process List with Dependencies
// Get all processes and their dependencies in both areas
MATCH (p:Process)
WHERE p.processId STARTS WITH 'RETAIL-CATALOG' OR p.processId STARTS WITH 'RETAIL-COMPLIANCE'
OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)
OPTIONAL MATCH (p)-[:DEPENDS_ON]->(dep:Process)
RETURN 
    p.processId as ProcessID,
    p.name as ProcessName,
    p.description as Description,
    parent.name as ParentProcess,
    COLLECT(DISTINCT dep.name) as Dependencies
ORDER BY ProcessID;

// 6. Top Two Levels Visual Graph
// Visual graph showing the top two levels of both hierarchies
MATCH (domain:Domain {name: 'Retail'})
MATCH (p:Process)-[:IN_DOMAIN]->(domain)
WHERE p.processId = 'RETAIL-CATALOG-001' OR p.processId = 'RETAIL-COMPLIANCE-001'
MATCH (sub:Process)-[:PART_OF]->(p)
OPTIONAL MATCH (sub)-[:DEPENDS_ON]->(dep:Process)
WHERE dep.processId STARTS WITH 'RETAIL-CATALOG' OR dep.processId STARTS WITH 'RETAIL-COMPLIANCE'
RETURN p, sub, dep, domain;

// 7. Find Specific Process with Its Children and Parents
// This query can be used to focus on a specific process area
// Replace 'RETAIL-CATALOG-001-001' with any process ID to focus on that area
MATCH (p:Process {processId: 'RETAIL-CATALOG-001-001'}) // Item Setup
OPTIONAL MATCH (child:Process)-[:PART_OF]->(p)
OPTIONAL MATCH (p)-[:PART_OF]->(parent:Process)
OPTIONAL MATCH (p)-[:DEPENDS_ON]->(dep:Process)
OPTIONAL MATCH (dependent:Process)-[:DEPENDS_ON]->(p)
RETURN 
    p.name as FocusProcess,
    p.description as Description,
    parent.name as ParentProcess,
    COLLECT(DISTINCT child.name) as ChildProcesses,
    COLLECT(DISTINCT dep.name) as DependsOn,
    COLLECT(DISTINCT dependent.name) as RequiredBy;

// 8. Find Critical Path in Process Dependencies
// This query finds the longest dependency chain in the process hierarchy
MATCH path = (start:Process)-[:DEPENDS_ON*]->(end:Process)
WHERE 
    (start.processId STARTS WITH 'RETAIL-CATALOG' OR start.processId STARTS WITH 'RETAIL-COMPLIANCE')
    AND NOT (start)<-[:DEPENDS_ON]-(:Process)
    AND NOT (end)-[:DEPENDS_ON]->(:Process)
RETURN path, length(path) as PathLength
ORDER BY PathLength DESC
LIMIT 5;

// 9. Find Processes Without Dependencies
// This identifies processes that don't depend on other processes
MATCH (p:Process)
WHERE 
    (p.processId STARTS WITH 'RETAIL-CATALOG' OR p.processId STARTS WITH 'RETAIL-COMPLIANCE')
    AND NOT (p)-[:DEPENDS_ON]->(:Process)
RETURN p.name as IndependentProcess, p.processId as ProcessID, p.description as Description;

// 10. Find Leaf Processes (no children)
// This identifies the most detailed processes in the hierarchy
MATCH (p:Process)
WHERE 
    (p.processId STARTS WITH 'RETAIL-CATALOG' OR p.processId STARTS WITH 'RETAIL-COMPLIANCE')
    AND NOT (:Process)-[:PART_OF]->(p)
RETURN p.name as LeafProcess, p.processId as ProcessID, p.description as Description;