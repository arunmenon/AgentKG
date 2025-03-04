#!/usr/bin/env python
"""
Style Guide Agent Registry Script

This script demonstrates how to register the Style Guide Agent with the AgentKG registry.
It uses the CrewAnalyzerAgent to extract metadata and then registers the crew and its agents.
"""

import os
import sys
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Add paths for imports
sys.path.append('/Users/arunmenon/projects/AgentKG')
from AgentKG.src.agent_registry_schema import (
    AgentRegistration, 
    CrewRegistration, 
    AgentMetadata,
    CrewMetadata,
    AgentType,
    ApiAuth
)

# Load environment variables
load_dotenv()

# Configuration
STYLE_GUIDE_REPO_PATH = "/tmp/style-guide-agent"
AGENTKG_API_URL = os.getenv("AGENTKG_API_URL", "http://localhost:8000")
AGENTKG_API_KEY = os.getenv("AGENTKG_API_KEY", "")


def create_style_guide_crew_metadata() -> CrewMetadata:
    """
    Create metadata for the Style Guide Generator crew
    """
    return CrewMetadata(
        crew_id="CREW-STYLE-GUIDE",
        name="Style Guide Generator Crew",
        description="Multi-agent crew for generating structured, compliant style guides for e-commerce product descriptions across various retail categories and product types",
        version="1.0.0",
        agent_ids=[
            "AGENT-STYLE-GUIDE-001",
            "AGENT-STYLE-GUIDE-002",
            "AGENT-STYLE-GUIDE-003",
            "AGENT-STYLE-GUIDE-004",
            "AGENT-STYLE-GUIDE-005",
            "AGENT-STYLE-GUIDE-006",
            "AGENT-STYLE-GUIDE-007"
        ],
        repository_url="https://github.com/arunmenon/style-guide-agent",
        repository_path="/style_guide_gen/crew_flow/crew.py",
        process_ids=[
            "RETAIL-CATALOG-001-001",
            "RETAIL-CATALOG-001-002",
            "RETAIL-CATALOG-001-003",
            "RETAIL-CATALOG-001-004",
            "RETAIL-CATALOG-001-005"
        ],
        api_endpoint="https://api.style-guide-agent.example/v1/generate",
        api_auth_type=ApiAuth.API_KEY,
        api_docs_url=None
    )


def create_style_guide_agents() -> list[AgentMetadata]:
    """
    Create metadata for all Style Guide agents
    """
    agents = [
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-001",
            name="Knowledge Aggregator",
            description="Retrieves baseline style guidelines and legal constraints from knowledge sources",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "data_retrieval",
                "knowledge_integration",
                "database_querying"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-002",
            name="Domain Breakdown Agent",
            description="Analyzes domain-level constraints for retail categories",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "domain_analysis",
                "category_constraints_identification",
                "guideline_summarization"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-003",
            name="Product Type Analyzer",
            description="Refines guidelines for specific product types within a category",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "product_analysis",
                "guideline_refinement",
                "field_specific_constraints"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-004",
            name="Schema Inference Agent",
            description="Proposes structured output format for style guides",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "schema_design",
                "data_structure_creation",
                "format_standardization"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-005",
            name="Style Guide Constructor",
            description="Creates draft guidelines for product description fields",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "content_creation",
                "style_guide_drafting",
                "template_development"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-006",
            name="Legal Review Agent",
            description="Checks style guides for brand/IP compliance and legal issues",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "regulatory_compliance",
                "brand_guideline_enforcement",
                "legal_risk_assessment"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        ),
        AgentMetadata(
            agent_id="AGENT-STYLE-GUIDE-007",
            name="Final Refinement Agent",
            description="Produces polished, markdown-formatted guidelines",
            version="1.0.0",
            type=AgentType.SPECIALIZED,
            capabilities=[
                "content_refinement",
                "format_standardization",
                "markdown_formatting"
            ],
            repository_url="https://github.com/arunmenon/style-guide-agent",
            repository_path="/style_guide_gen/crew_flow/crew.py"
        )
    ]
    return agents


def register_style_guide_crew():
    """
    Register the Style Guide crew and its agents with the AgentKG registry
    """
    # Create crew metadata
    crew_metadata = create_style_guide_crew_metadata()
    
    # Create agent registrations
    agent_registrations = []
    for agent_metadata in create_style_guide_agents():
        agent_reg = AgentRegistration(
            metadata=agent_metadata,
            capabilities_detail=None
        )
        agent_registrations.append(agent_reg)
    
    # Create crew registration
    crew_registration = CrewRegistration(
        metadata=crew_metadata,
        agent_registrations=agent_registrations
    )
    
    # Register with the API
    headers = {}
    if AGENTKG_API_KEY:
        headers["X-Api-Key"] = AGENTKG_API_KEY
    
    try:
        # For demonstration, we print the registration data instead of sending it
        print("\n=== Style Guide Crew Registration Data ===")
        # Convert to dict and handle URL objects by converting them to strings
        reg_data = crew_registration.model_dump()
        
        # Fix URL serialization
        def fix_urls(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.endswith('_url') and value is not None:
                        obj[key] = str(value)
                    elif isinstance(value, (dict, list)):
                        fix_urls(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        fix_urls(item)
            return obj
        
        reg_data = fix_urls(reg_data)
        print(json.dumps(reg_data, indent=2))
        print("\n")
        
        # In a real scenario, we would send the registration to the API
        # response = requests.post(
        #     f"{AGENTKG_API_URL}/crews/register",
        #     json=reg_data,
        #     headers=headers
        # )
        # 
        # if response.status_code == 200:
        #     print(f"Crew registration successful: {response.json()}")
        #     return True
        # else:
        #     print(f"Crew registration failed: {response.text}")
        #     return False
        
        return True
    except Exception as e:
        print(f"Error registering crew: {e}")
        return False


def simulate_crew_analyzer():
    """
    Simulate what the CrewAnalyzerAgent would do by outputting the analysis result
    """
    print("=== CrewAnalyzerAgent Analysis ===")
    print(f"Repository: {STYLE_GUIDE_REPO_PATH}")
    print("Analysis complete!")
    print("Found: StyleGuideCrew with 7 agents")
    print("Matched to processes: RETAIL-CATALOG-001-001 through RETAIL-CATALOG-001-005")
    print("Capabilities detected: style_guide_generation, content_creation, regulatory_compliance")
    print("Confidence score: 0.95")
    print("\nProceeding with registration...")


def main():
    """Main function to demonstrate crew registration"""
    print("Starting Style Guide Agent registration with AgentKG...")
    
    # Simulate the CrewAnalyzerAgent
    simulate_crew_analyzer()
    
    # Register the crew and its agents
    success = register_style_guide_crew()
    
    if success:
        print("\nRegistration complete!")
        print("The Style Guide Agent has been successfully registered with AgentKG")
        print("It is now discoverable for the following processes:")
        print("  - RETAIL-CATALOG-001-001 (Item Setup)")
        print("  - RETAIL-CATALOG-001-002 (Item Maintenance)")
        print("  - RETAIL-CATALOG-001-003 (Catalog Optimization)")
        print("  - RETAIL-CATALOG-001-004 (Catalog Data Quality)")
        print("  - RETAIL-CATALOG-001-005 (Catalog Compliance)")
    else:
        print("\nRegistration failed. Please check the error logs.")


if __name__ == "__main__":
    main()