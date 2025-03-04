"""
Script to expand the process hierarchy for Catalog and Compliance (Trust and Safety)
under the Retail domain in the AgentKG graph.
"""

import os
import json
import openai
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

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


class RetailProcessExpander:
    """
    Class for expanding the Retail process hierarchy, specifically
    for Catalog and Compliance (Trust and Safety) processes.
    """
    
    def __init__(self):
        """Initialize the RetailProcessExpander"""
        self.connector = Neo4jConnector()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def close(self):
        """Close Neo4j connection"""
        self.connector.close()
    
    def check_retail_domain_exists(self):
        """Check if the Retail domain exists in the graph"""
        query = """
        MATCH (d:Domain {name: 'Retail'})
        RETURN d
        """
        result = self.connector.execute_query(query)
        return len(result) > 0
    
    def create_retail_domain(self):
        """Create the Retail domain if it doesn't exist"""
        query = """
        MERGE (d:Domain {name: 'Retail'})
        ON CREATE SET d.description = 'Retail business domain involving catalog management, merchandising, compliance, and customer-facing operations'
        RETURN d
        """
        self.connector.execute_query(query)
        print("Retail domain created")
    
    def check_if_process_exists(self, process_id):
        """Check if a process with the given ID exists"""
        query = """
        MATCH (p:Process {processId: $processId})
        RETURN p
        """
        result = self.connector.execute_query(query, {"processId": process_id})
        return len(result) > 0
    
    def add_process(self, process_id, name, description, domain_name="Retail", status="Active"):
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
    
    def generate_catalog_processes(self):
        """Generate detailed process hierarchy for Catalog management"""
        print("\nGenerating detailed Catalog process hierarchy...")
        
        # Check if the catalog management process already exists
        catalog_exists = self.check_if_process_exists("RETAIL-CATALOG-001")
        
        if not catalog_exists:
            # Create top-level Catalog Management process
            self.add_process(
                "RETAIL-CATALOG-001",
                "Catalog Management",
                "End-to-end process for managing the retail product catalog, including item setup, maintenance, and optimization"
            )
        
        # Level 2 processes (sub-processes of Catalog Management)
        level2_processes = [
            {
                "id": "RETAIL-CATALOG-001-001",
                "name": "Item Setup",
                "description": "Process for setting up new items in the catalog",
                "parent_id": "RETAIL-CATALOG-001"
            },
            {
                "id": "RETAIL-CATALOG-001-002",
                "name": "Item Maintenance",
                "description": "Process for maintaining and updating existing items in the catalog",
                "parent_id": "RETAIL-CATALOG-001"
            },
            {
                "id": "RETAIL-CATALOG-001-003",
                "name": "Catalog Optimization",
                "description": "Process for optimizing the catalog structure, categories, and product relationships",
                "parent_id": "RETAIL-CATALOG-001"
            },
            {
                "id": "RETAIL-CATALOG-001-004",
                "name": "Catalog Data Quality",
                "description": "Process for ensuring catalog data quality, completeness, and accuracy",
                "parent_id": "RETAIL-CATALOG-001"
            },
            {
                "id": "RETAIL-CATALOG-001-005",
                "name": "Catalog Compliance",
                "description": "Process for ensuring catalog compliance with regulations and company policies",
                "parent_id": "RETAIL-CATALOG-001"
            }
        ]
        
        # Add level 2 processes
        for process in level2_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Item Setup
        item_setup_processes = [
            {
                "id": "RETAIL-CATALOG-001-001-001",
                "name": "Item Data Collection",
                "description": "Collecting all necessary data elements for new item setup",
                "parent_id": "RETAIL-CATALOG-001-001"
            },
            {
                "id": "RETAIL-CATALOG-001-001-002",
                "name": "Item Classification",
                "description": "Classifying items into appropriate categories and hierarchies",
                "parent_id": "RETAIL-CATALOG-001-001"
            },
            {
                "id": "RETAIL-CATALOG-001-001-003",
                "name": "Item Attribute Definition",
                "description": "Defining and assigning product attributes and specifications",
                "parent_id": "RETAIL-CATALOG-001-001"
            },
            {
                "id": "RETAIL-CATALOG-001-001-004",
                "name": "Item Pricing Setup",
                "description": "Setting up initial pricing information for new items",
                "parent_id": "RETAIL-CATALOG-001-001"
            },
            {
                "id": "RETAIL-CATALOG-001-001-005",
                "name": "Item Media Association",
                "description": "Associating product images, videos, and other media with items",
                "parent_id": "RETAIL-CATALOG-001-001"
            },
            {
                "id": "RETAIL-CATALOG-001-001-006",
                "name": "Item Validation and Approval",
                "description": "Validating item data and obtaining necessary approvals before publication",
                "parent_id": "RETAIL-CATALOG-001-001"
            }
        ]
        
        # Add level 3 processes for Item Setup
        for process in item_setup_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Item Maintenance
        item_maintenance_processes = [
            {
                "id": "RETAIL-CATALOG-001-002-001",
                "name": "Item Data Updates",
                "description": "Updating existing item data elements",
                "parent_id": "RETAIL-CATALOG-001-002"
            },
            {
                "id": "RETAIL-CATALOG-001-002-002",
                "name": "Item Status Management",
                "description": "Managing item lifecycle states (active, discontinued, seasonal, etc.)",
                "parent_id": "RETAIL-CATALOG-001-002"
            },
            {
                "id": "RETAIL-CATALOG-001-002-003",
                "name": "Item Relationship Management",
                "description": "Managing relationships between items (accessories, substitutes, complementary products)",
                "parent_id": "RETAIL-CATALOG-001-002"
            },
            {
                "id": "RETAIL-CATALOG-001-002-004",
                "name": "Item Version Control",
                "description": "Tracking and managing item versions and historical changes",
                "parent_id": "RETAIL-CATALOG-001-002"
            },
            {
                "id": "RETAIL-CATALOG-001-002-005",
                "name": "Item Bulk Updates",
                "description": "Performing batch updates to multiple items simultaneously",
                "parent_id": "RETAIL-CATALOG-001-002"
            }
        ]
        
        # Add level 3 processes for Item Maintenance
        for process in item_maintenance_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Catalog Optimization
        catalog_optimization_processes = [
            {
                "id": "RETAIL-CATALOG-001-003-001",
                "name": "Category Structure Optimization",
                "description": "Optimizing the structure and hierarchy of product categories",
                "parent_id": "RETAIL-CATALOG-001-003"
            },
            {
                "id": "RETAIL-CATALOG-001-003-002",
                "name": "Search Relevance Optimization",
                "description": "Optimizing product data to improve search relevance and findability",
                "parent_id": "RETAIL-CATALOG-001-003"
            },
            {
                "id": "RETAIL-CATALOG-001-003-003",
                "name": "Cross-Sell and Up-Sell Setup",
                "description": "Setting up cross-sell and up-sell recommendations between products",
                "parent_id": "RETAIL-CATALOG-001-003"
            },
            {
                "id": "RETAIL-CATALOG-001-003-004",
                "name": "Catalog Performance Analysis",
                "description": "Analyzing catalog performance metrics and identifying improvement opportunities",
                "parent_id": "RETAIL-CATALOG-001-003"
            }
        ]
        
        # Add level 3 processes for Catalog Optimization
        for process in catalog_optimization_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Catalog Data Quality
        catalog_data_quality_processes = [
            {
                "id": "RETAIL-CATALOG-001-004-001",
                "name": "Data Completeness Checks",
                "description": "Checking catalog data for completeness against required fields",
                "parent_id": "RETAIL-CATALOG-001-004"
            },
            {
                "id": "RETAIL-CATALOG-001-004-002",
                "name": "Data Accuracy Validation",
                "description": "Validating the accuracy of catalog data against trusted sources",
                "parent_id": "RETAIL-CATALOG-001-004"
            },
            {
                "id": "RETAIL-CATALOG-001-004-003",
                "name": "Data Consistency Enforcement",
                "description": "Ensuring consistency of data formats and values across the catalog",
                "parent_id": "RETAIL-CATALOG-001-004"
            },
            {
                "id": "RETAIL-CATALOG-001-004-004",
                "name": "Data Enrichment",
                "description": "Enhancing catalog data with additional attributes and information",
                "parent_id": "RETAIL-CATALOG-001-004"
            },
            {
                "id": "RETAIL-CATALOG-001-004-005",
                "name": "Data Quality Reporting",
                "description": "Generating reports on catalog data quality metrics and issues",
                "parent_id": "RETAIL-CATALOG-001-004"
            }
        ]
        
        # Add level 3 processes for Catalog Data Quality
        for process in catalog_data_quality_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Catalog Compliance
        catalog_compliance_processes = [
            {
                "id": "RETAIL-CATALOG-001-005-001",
                "name": "Regulatory Compliance Checks",
                "description": "Checking catalog data for compliance with applicable regulations",
                "parent_id": "RETAIL-CATALOG-001-005"
            },
            {
                "id": "RETAIL-CATALOG-001-005-002",
                "name": "Policy Compliance Validation",
                "description": "Validating catalog data against internal policies and standards",
                "parent_id": "RETAIL-CATALOG-001-005"
            },
            {
                "id": "RETAIL-CATALOG-001-005-003",
                "name": "Compliance Issue Resolution",
                "description": "Resolving identified compliance issues in the catalog",
                "parent_id": "RETAIL-CATALOG-001-005"
            },
            {
                "id": "RETAIL-CATALOG-001-005-004",
                "name": "Compliance Reporting",
                "description": "Generating reports on catalog compliance status and issues",
                "parent_id": "RETAIL-CATALOG-001-005"
            }
        ]
        
        # Add level 3 processes for Catalog Compliance
        for process in catalog_compliance_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Add process dependencies
        dependencies = [
            # Item Setup dependencies
            ("RETAIL-CATALOG-001-001-002", "RETAIL-CATALOG-001-001-001"),  # Classification depends on Data Collection
            ("RETAIL-CATALOG-001-001-003", "RETAIL-CATALOG-001-001-002"),  # Attributes depend on Classification
            ("RETAIL-CATALOG-001-001-006", "RETAIL-CATALOG-001-001-003"),  # Validation depends on Attributes
            ("RETAIL-CATALOG-001-001-006", "RETAIL-CATALOG-001-001-004"),  # Validation depends on Pricing
            ("RETAIL-CATALOG-001-001-006", "RETAIL-CATALOG-001-001-005"),  # Validation depends on Media
            
            # Item Maintenance dependencies
            ("RETAIL-CATALOG-001-002", "RETAIL-CATALOG-001-001"),  # Maintenance depends on Setup
            ("RETAIL-CATALOG-001-002-002", "RETAIL-CATALOG-001-002-001"),  # Status depends on Data Updates
            
            # Cross-area dependencies
            ("RETAIL-CATALOG-001-003", "RETAIL-CATALOG-001-002"),  # Optimization depends on Maintenance
            ("RETAIL-CATALOG-001-004", "RETAIL-CATALOG-001-002"),  # Data Quality depends on Maintenance
            ("RETAIL-CATALOG-001-005", "RETAIL-CATALOG-001-004"),  # Compliance depends on Data Quality
        ]
        
        # Add dependencies
        for dep in dependencies:
            self.add_process_dependency(dep[0], dep[1])
        
        print("Catalog process hierarchy expanded successfully")
    
    def generate_compliance_processes(self):
        """Generate detailed process hierarchy for Compliance (Trust and Safety)"""
        print("\nGenerating detailed Compliance (Trust and Safety) process hierarchy...")
        
        # Check if the compliance management process already exists
        compliance_exists = self.check_if_process_exists("RETAIL-COMPLIANCE-001")
        
        if not compliance_exists:
            # Create top-level Compliance Management process
            self.add_process(
                "RETAIL-COMPLIANCE-001",
                "Trust and Safety Compliance",
                "End-to-end process for ensuring retail operations comply with regulations, policies, and safety standards"
            )
        
        # Level 2 processes (sub-processes of Compliance Management)
        level2_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-001",
                "name": "Product Safety Compliance",
                "description": "Ensuring products meet safety standards and regulations",
                "parent_id": "RETAIL-COMPLIANCE-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-002",
                "name": "Content Moderation",
                "description": "Reviewing and moderating product descriptions, images, and user-generated content",
                "parent_id": "RETAIL-COMPLIANCE-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-003",
                "name": "Regulatory Compliance",
                "description": "Ensuring compliance with local, national, and international regulations",
                "parent_id": "RETAIL-COMPLIANCE-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-004",
                "name": "Fraud Prevention",
                "description": "Detecting and preventing fraudulent activities in retail operations",
                "parent_id": "RETAIL-COMPLIANCE-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-005",
                "name": "Policy Enforcement",
                "description": "Enforcing company policies across retail operations",
                "parent_id": "RETAIL-COMPLIANCE-001"
            }
        ]
        
        # Add level 2 processes
        for process in level2_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Product Safety Compliance
        product_safety_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-001-001",
                "name": "Safety Certification Verification",
                "description": "Verifying product safety certifications and documentation",
                "parent_id": "RETAIL-COMPLIANCE-001-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-001-002",
                "name": "Product Recall Management",
                "description": "Managing the process for product recalls and safety alerts",
                "parent_id": "RETAIL-COMPLIANCE-001-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-001-003",
                "name": "Product Safety Testing",
                "description": "Conducting or reviewing safety testing of products",
                "parent_id": "RETAIL-COMPLIANCE-001-001"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-001-004",
                "name": "Safety Compliance Reporting",
                "description": "Reporting on product safety compliance status and issues",
                "parent_id": "RETAIL-COMPLIANCE-001-001"
            }
        ]
        
        # Add level 3 processes for Product Safety Compliance
        for process in product_safety_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Content Moderation
        content_moderation_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-002-001",
                "name": "Product Description Review",
                "description": "Reviewing product descriptions for policy compliance",
                "parent_id": "RETAIL-COMPLIANCE-001-002"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-002-002",
                "name": "Product Image Moderation",
                "description": "Moderating product images for policy compliance",
                "parent_id": "RETAIL-COMPLIANCE-001-002"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-002-003",
                "name": "Review Moderation",
                "description": "Moderating customer reviews and questions",
                "parent_id": "RETAIL-COMPLIANCE-001-002"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-002-004",
                "name": "Prohibited Content Detection",
                "description": "Detecting and removing prohibited content",
                "parent_id": "RETAIL-COMPLIANCE-001-002"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-002-005",
                "name": "Moderation Appeals Processing",
                "description": "Processing appeals against moderation decisions",
                "parent_id": "RETAIL-COMPLIANCE-001-002"
            }
        ]
        
        # Add level 3 processes for Content Moderation
        for process in content_moderation_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Regulatory Compliance
        regulatory_compliance_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-003-001",
                "name": "Regulatory Requirements Tracking",
                "description": "Tracking and documenting applicable regulatory requirements",
                "parent_id": "RETAIL-COMPLIANCE-001-003"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-003-002",
                "name": "Compliance Audit",
                "description": "Conducting audits to verify regulatory compliance",
                "parent_id": "RETAIL-COMPLIANCE-001-003"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-003-003",
                "name": "Regulatory Reporting",
                "description": "Preparing and submitting required regulatory reports",
                "parent_id": "RETAIL-COMPLIANCE-001-003"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-003-004",
                "name": "Compliance Training",
                "description": "Providing training on regulatory requirements and compliance procedures",
                "parent_id": "RETAIL-COMPLIANCE-001-003"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-003-005",
                "name": "Regulatory Change Management",
                "description": "Managing the implementation of changes required by new regulations",
                "parent_id": "RETAIL-COMPLIANCE-001-003"
            }
        ]
        
        # Add level 3 processes for Regulatory Compliance
        for process in regulatory_compliance_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Fraud Prevention
        fraud_prevention_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-004-001",
                "name": "Fraud Detection",
                "description": "Detecting potential fraudulent activities",
                "parent_id": "RETAIL-COMPLIANCE-001-004"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-004-002",
                "name": "Fraud Investigation",
                "description": "Investigating suspected fraudulent activities",
                "parent_id": "RETAIL-COMPLIANCE-001-004"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-004-003",
                "name": "Fraud Mitigation",
                "description": "Implementing measures to prevent and mitigate fraud",
                "parent_id": "RETAIL-COMPLIANCE-001-004"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-004-004",
                "name": "Fraud Reporting",
                "description": "Reporting on fraud incidents and prevention measures",
                "parent_id": "RETAIL-COMPLIANCE-001-004"
            }
        ]
        
        # Add level 3 processes for Fraud Prevention
        for process in fraud_prevention_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Level 3 processes for Policy Enforcement
        policy_enforcement_processes = [
            {
                "id": "RETAIL-COMPLIANCE-001-005-001",
                "name": "Policy Development",
                "description": "Developing and documenting company policies",
                "parent_id": "RETAIL-COMPLIANCE-001-005"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-005-002",
                "name": "Policy Communication",
                "description": "Communicating policies to relevant stakeholders",
                "parent_id": "RETAIL-COMPLIANCE-001-005"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-005-003",
                "name": "Policy Compliance Monitoring",
                "description": "Monitoring compliance with company policies",
                "parent_id": "RETAIL-COMPLIANCE-001-005"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-005-004",
                "name": "Policy Violation Handling",
                "description": "Handling and resolving policy violations",
                "parent_id": "RETAIL-COMPLIANCE-001-005"
            },
            {
                "id": "RETAIL-COMPLIANCE-001-005-005",
                "name": "Policy Effectiveness Review",
                "description": "Reviewing and evaluating the effectiveness of policies",
                "parent_id": "RETAIL-COMPLIANCE-001-005"
            }
        ]
        
        # Add level 3 processes for Policy Enforcement
        for process in policy_enforcement_processes:
            if not self.check_if_process_exists(process["id"]):
                self.add_subprocess(
                    process["id"],
                    process["name"],
                    process["description"],
                    process["parent_id"]
                )
        
        # Add process dependencies
        dependencies = [
            # Product Safety dependencies
            ("RETAIL-COMPLIANCE-001-001-002", "RETAIL-COMPLIANCE-001-001-001"),  # Recall depends on Certification
            ("RETAIL-COMPLIANCE-001-001-004", "RETAIL-COMPLIANCE-001-001-003"),  # Reporting depends on Testing
            
            # Content Moderation dependencies
            ("RETAIL-COMPLIANCE-001-002-004", "RETAIL-COMPLIANCE-001-002-001"),  # Prohibited Content depends on Description Review
            ("RETAIL-COMPLIANCE-001-002-004", "RETAIL-COMPLIANCE-001-002-002"),  # Prohibited Content depends on Image Moderation
            ("RETAIL-COMPLIANCE-001-002-005", "RETAIL-COMPLIANCE-001-002-004"),  # Appeals depends on Prohibited Content
            
            # Fraud Prevention dependencies
            ("RETAIL-COMPLIANCE-001-004-002", "RETAIL-COMPLIANCE-001-004-001"),  # Investigation depends on Detection
            ("RETAIL-COMPLIANCE-001-004-003", "RETAIL-COMPLIANCE-001-004-002"),  # Mitigation depends on Investigation
            
            # Policy Enforcement dependencies
            ("RETAIL-COMPLIANCE-001-005-002", "RETAIL-COMPLIANCE-001-005-001"),  # Communication depends on Development
            ("RETAIL-COMPLIANCE-001-005-003", "RETAIL-COMPLIANCE-001-005-002"),  # Monitoring depends on Communication
            ("RETAIL-COMPLIANCE-001-005-004", "RETAIL-COMPLIANCE-001-005-003"),  # Violation Handling depends on Monitoring
            ("RETAIL-COMPLIANCE-001-005-005", "RETAIL-COMPLIANCE-001-005-004"),  # Effectiveness depends on Violation Handling
            
            # Cross-area dependencies
            ("RETAIL-COMPLIANCE-001-002", "RETAIL-CATALOG-001-005"),  # Content Moderation depends on Catalog Compliance
            ("RETAIL-COMPLIANCE-001-001", "RETAIL-CATALOG-001-005"),  # Product Safety depends on Catalog Compliance
        ]
        
        # Add dependencies
        for dep in dependencies:
            self.add_process_dependency(dep[0], dep[1])
        
        print("Compliance (Trust and Safety) process hierarchy expanded successfully")


def main():
    """Main function to expand Retail process hierarchies"""
    expander = RetailProcessExpander()
    
    # Check if Retail domain exists, create if not
    if not expander.check_retail_domain_exists():
        expander.create_retail_domain()
    
    # Generate detailed process hierarchies
    expander.generate_catalog_processes()
    expander.generate_compliance_processes()
    
    # Close connection
    expander.close()
    
    print("\nProcess hierarchy expansion complete!")
    print("The Neo4j graph now contains detailed process hierarchies for Catalog and Compliance (Trust and Safety) in the Retail domain.")


if __name__ == "__main__":
    main()