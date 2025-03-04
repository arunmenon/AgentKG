# Style Guide Agent Registry Mapping

## AgentKG Registry Entry

Below is how the style-guide-agent would be represented in the AgentKG registry:

```json
{
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
  "agent_metadata": [
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
  ],
  "process_ids": [
    "RETAIL-CATALOG-001-001",
    "RETAIL-CATALOG-001-002",
    "RETAIL-CATALOG-001-003",
    "RETAIL-CATALOG-001-004",
    "RETAIL-CATALOG-001-005"
  ],
  "capabilities": [
    "style_guide_generation",
    "content_creation",
    "regulatory_compliance",
    "product_description_standardization",
    "brand_guideline_enforcement",
    "markdown_formatting",
    "schema_design"
  ],
  "confidence_score": 0.95
}
```

## Process Mapping

This crew is particularly well-suited for these catalog processes:

1. **RETAIL-CATALOG-001-001** (Item Setup) - Providing style guides for initial item creation
2. **RETAIL-CATALOG-001-002** (Item Maintenance) - Supporting updates to existing items
3. **RETAIL-CATALOG-001-003** (Catalog Optimization) - Improving item descriptions and searchability
4. **RETAIL-CATALOG-001-004** (Catalog Data Quality) - Ensuring consistency and quality of descriptions
5. **RETAIL-CATALOG-001-005** (Catalog Compliance) - Enforcing regulatory and brand compliance

## Integration Approach

To integrate this crew with AgentKG:

1. **API Wrapper Development**:
   - Create a RESTful API wrapper around the style-guide-agent
   - Implement authentication and rate limiting
   - Standardize input/output formats to match AgentKG expectations

2. **Database Integration**:
   - Ensure knowledge sources can connect to standardized databases
   - Support both SQLite (development) and enterprise databases (production)

3. **Execution Environment**:
   - Containerize the crew for deployment
   - Define resource requirements and scaling parameters

4. **Monitoring & Logging**:
   - Implement standardized logging compatible with AgentKG observability
   - Add performance metrics collection

## Benefits to AgentKG

This crew provides significant value to the AgentKG ecosystem:

1. **Specialized Retail Catalog Expertise**: Deep domain knowledge for e-commerce product descriptions
2. **Compliance Enforcement**: Built-in legal review capabilities ensure all content meets requirements
3. **Multi-Field Handling**: Support for various content types (title, short description, long description)
4. **Structured Output**: Standardized markdown format for style guides
5. **Database Integration**: Ready-made connections to knowledge sources