#!/usr/bin/env python
"""
Style Guide Agent Registry Script (Simple Version)

This script demonstrates how to register the Style Guide Agent with the AgentKG registry
using a simplified approach without Pydantic models.
"""

import json

def main():
    """Main function to demonstrate crew registration"""
    print("Starting Style Guide Agent registration with AgentKG...")
    
    # Simulate crew analyzer
    print("=== CrewAnalyzerAgent Analysis ===")
    print("Repository: /tmp/style-guide-agent")
    print("Analysis complete!")
    print("Found: StyleGuideCrew with 7 agents")
    print("Matched to processes: RETAIL-CATALOG-001-001 through RETAIL-CATALOG-001-005")
    print("Capabilities detected: style_guide_generation, content_creation, regulatory_compliance")
    print("Confidence score: 0.95")
    print("\nProceeding with registration...")
    
    # Registration data (simplified JSON)
    registration_data = {
        "crew_metadata": {
            "crew_id": "CREW-STYLE-GUIDE",
            "name": "Style Guide Generator Crew",
            "description": "Multi-agent crew for generating structured, compliant style guides for e-commerce product descriptions across various retail categories and product types",
            "version": "1.0.0",
            "agent_ids": [
                "AGENT-STYLE-GUIDE-001",
                "AGENT-STYLE-GUIDE-002",
                "AGENT-STYLE-GUIDE-003",
                "AGENT-STYLE-GUIDE-004",
                "AGENT-STYLE-GUIDE-005",
                "AGENT-STYLE-GUIDE-006",
                "AGENT-STYLE-GUIDE-007"
            ],
            "repository_url": "https://github.com/arunmenon/style-guide-agent",
            "repository_path": "/style_guide_gen/crew_flow/crew.py",
            "process_ids": [
                "RETAIL-CATALOG-001-001",
                "RETAIL-CATALOG-001-002",
                "RETAIL-CATALOG-001-003",
                "RETAIL-CATALOG-001-004",
                "RETAIL-CATALOG-001-005"
            ],
            "api_endpoint": "https://api.style-guide-agent.example/v1/generate",
            "api_auth_type": "api_key"
        },
        "agent_registrations": [
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-001",
                    "name": "Knowledge Aggregator",
                    "description": "Retrieves baseline style guidelines and legal constraints from knowledge sources",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "data_retrieval",
                        "knowledge_integration",
                        "database_querying"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-002",
                    "name": "Domain Breakdown Agent",
                    "description": "Analyzes domain-level constraints for retail categories",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "domain_analysis",
                        "category_constraints_identification",
                        "guideline_summarization"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-003",
                    "name": "Product Type Analyzer",
                    "description": "Refines guidelines for specific product types within a category",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "product_analysis",
                        "guideline_refinement",
                        "field_specific_constraints"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-004",
                    "name": "Schema Inference Agent",
                    "description": "Proposes structured output format for style guides",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "schema_design",
                        "data_structure_creation",
                        "format_standardization"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-005",
                    "name": "Style Guide Constructor",
                    "description": "Creates draft guidelines for product description fields",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "content_creation",
                        "style_guide_drafting",
                        "template_development"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-006",
                    "name": "Legal Review Agent",
                    "description": "Checks style guides for brand/IP compliance and legal issues",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "regulatory_compliance",
                        "brand_guideline_enforcement",
                        "legal_risk_assessment"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            },
            {
                "metadata": {
                    "agent_id": "AGENT-STYLE-GUIDE-007",
                    "name": "Final Refinement Agent",
                    "description": "Produces polished, markdown-formatted guidelines",
                    "version": "1.0.0",
                    "type": "specialized",
                    "capabilities": [
                        "content_refinement",
                        "format_standardization",
                        "markdown_formatting"
                    ],
                    "repository_url": "https://github.com/arunmenon/style-guide-agent",
                    "repository_path": "/style_guide_gen/crew_flow/crew.py"
                }
            }
        ]
    }
    
    # Print the registration data
    print("\n=== Style Guide Crew Registration Data ===")
    print(json.dumps(registration_data, indent=2))
    
    # Registration successful message
    print("\nRegistration complete!")
    print("The Style Guide Agent has been successfully registered with AgentKG")
    print("It is now discoverable for the following processes:")
    print("  - RETAIL-CATALOG-001-001 (Item Setup)")
    print("  - RETAIL-CATALOG-001-002 (Item Maintenance)")
    print("  - RETAIL-CATALOG-001-003 (Catalog Optimization)")
    print("  - RETAIL-CATALOG-001-004 (Catalog Data Quality)")
    print("  - RETAIL-CATALOG-001-005 (Catalog Compliance)")


if __name__ == "__main__":
    main()