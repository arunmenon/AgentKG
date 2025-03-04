# Style Guide Agent Analysis

## Overview
The style-guide-agent repository contains a specialized CrewAI implementation for automatically generating style guides for e-commerce product descriptions. The system is designed to create structured, compliant style guides for product titles, short descriptions, and long descriptions in various retail categories.

## Core Components

### StyleGuideCrew
- Multi-agent workflow that handles the end-to-end process of style guide generation
- Retrieves baseline guidelines and legal constraints from knowledge sources
- Processes multiple fields (title, shortDesc, longDesc) through construction, legal review, and refinement
- Stores final style guides in a database

### TitleStyleFlow 
- Alternative implementation using CrewAI's Flow paradigm
- Event-driven feedback loop for iterative refinement of style guides
- Includes validation and human approval steps

## Agent Structure

The system employs 7 specialized agents working in sequence:

1. **Knowledge Aggregator Agent**: Retrieves baseline and legal guidelines from knowledge sources
2. **Domain Breakdown Agent**: Analyzes domain-level constraints for categories (e.g., Fashion)
3. **Product Type Agent**: Refines guidelines for specific product types
4. **Schema Inference Agent**: Proposes structured output format for style guides
5. **Style Guide Construction Agent**: Creates draft guidelines for each field
6. **Legal Review Agent**: Checks for brand/IP compliance and legal issues
7. **Final Refinement Agent**: Produces polished, markdown-formatted guidelines

## Process Flow

1. Knowledge retrieval from database
2. Domain-level analysis (e.g., Fashion category rules)
3. Product-type analysis (e.g., specific rules for T-Shirts)
4. Schema definition
5. For each field (title, shortDesc, longDesc):
   - Draft creation
   - Legal compliance review
   - Final refinement
6. Storage of final style guides in database

## Technical Implementation

- Built with CrewAI for agent orchestration
- Uses SQLite database for knowledge storage and persistence
- Provides both a sequential Crew implementation and an event-driven Flow implementation
- Includes FastAPI endpoints for integration

## Integration Points

- Database connections for retrieving baseline guidelines
- API endpoints for triggering style guide generation
- Output persistence in SQL database

## Capabilities

- Generates compliant style guides for e-commerce product descriptions
- Ensures legal and brand compliance
- Handles specific requirements for different retail categories and product types
- Adapts to different field requirements (title, shortDesc, longDesc)
- Provides structured, markdown-formatted output

This system demonstrates a sophisticated approach to using orchestrated AI agents to solve a specific business problem in the e-commerce domain, with particular emphasis on maintaining compliance and consistency across product descriptions.