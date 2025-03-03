# AgentKG - Agent Knowledge Graph

AgentKG is a comprehensive knowledge graph framework for agent orchestration. It provides a Neo4j-based graph schema for representing business processes, agents, crews, and their relationships.

## Project Overview

This project implements a knowledge graph that organizes:

- Business processes and sub-processes
- Agents (human or AI) and crews (teams)
- Business functions and technical expertise taxonomies
- Tasks and assignments
- Performance metrics and records

The graph schema is designed to help answer questions like:
- Who is responsible for a specific process?
- What skills and expertise does an agent have?
- Which agents are best suited for a particular task?
- How are processes structured and what are their dependencies?

## Project Structure

- `/AgentKG/`: Main package
  - `/src/`: Source code
    - `neo4j_connector.py`: Neo4j database connection handler
    - `schema_creator.py`: Creates the Neo4j schema with constraints and indexes
    - `knowledge_ingestion.py`: Uses LLMs to ingest knowledge from text
    - `agent_augmentation.py`: Uses search tools to augment the knowledge graph
    - `graph_queries.py`: Common queries for interacting with the graph
    - `schema_visualizer.py`: Visualizes the schema design
    - `knowledge_ingestion_demo.py`: Demonstrates knowledge ingestion
    - `main.py`: Main script with commands for initialization, loading, and querying
  - `requirements.txt`: Dependencies for the project
  - `.env.example`: Example environment file

## Schema Design

### Node Labels

1. **Process**: Business processes or sub-processes
2. **Agent**: Individual agents (human or AI)
3. **Crew**: Teams or groups of agents
4. **BusinessFunction**: Business functions or role categories
5. **TechExpertise**: Technical specialties or skill categories
6. **Domain**: Top-level or sub-level domain areas
7. **Task**: Operational tasks within processes
8. **PerformanceRecord**: Historical performance records

### Relationship Types

1. `(:Process)-[:PART_OF]->(:Process)`: Sub-process hierarchy
2. `(:Process)-[:DEPENDS_ON]->(:Process)`: Process dependencies
3. `(:Crew)-[:SUPPORTS]->(:Process)`: Crew supports a process
4. `(:Agent)-[:SUPPORTS]->(:Process)`: Agent supports a process
5. `(:Agent)-[:MEMBER_OF {role:...}]->(:Crew)`: Agent membership in a crew
6. `(:Crew)-[:SUBTEAM_OF]->(:Crew)`: Crew hierarchy
7. `(:Agent)-[:HAS_FUNCTION]->(:BusinessFunction)`: Agent has a business function
8. `(:Agent)-[:HAS_EXPERTISE]->(:TechExpertise)`: Agent has technical expertise
9. `(:Process)-[:IN_DOMAIN]->(:Domain)`: Process belongs to a domain
10. `(:Agent)-[:ASSIGNED_TO]->(:Task)`: Agent is assigned to a task
11. `(:Task)-[:PART_OF]->(:Process)`: Task is part of a process
12. `(:Agent)-[:HAS_PERFORMANCE_RECORD]->(:PerformanceRecord)`: Agent performance records
13. `(:Crew)-[:HAS_PERFORMANCE_RECORD]->(:PerformanceRecord)`: Crew performance records

## Setup and Usage

### Prerequisites

- Python 3.7+
- Neo4j Database
- OpenAI API key (for LLM-based knowledge ingestion)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AgentKG.git
cd AgentKG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r AgentKG/requirements.txt
```

4. Create a `.env` file based on `.env.example` with your Neo4j and OpenAI credentials.

### Usage

#### Visualize the Schema

To visualize the schema design:

```bash
python AgentKG/src/schema_visualizer.py
```

#### Knowledge Ingestion Demo

To see a demonstration of LLM-based knowledge ingestion:

```bash
python AgentKG/src/knowledge_ingestion_demo.py
```

#### Populate Process Hierarchies

To populate the Neo4j database with detailed process hierarchies:

1. Install the necessary dependencies:
```bash
pip install neo4j python-dotenv openai
```

2. Run the populate script:
```bash
python AgentKG/populate_graph.py
```

The script will guide you through the process of selecting which domains to populate and whether to clear the existing database first.

#### Database Operations (Requires Neo4j)

Initialize the database schema:
```bash
python -m AgentKG.src.main init
```

Load example data:
```bash
python -m AgentKG.src.main load
```

Query the graph:
```bash
python -m AgentKG.src.main query --type processes
python -m AgentKG.src.main query --type agents
python -m AgentKG.src.main query --type crews
python -m AgentKG.src.main query --type tasks
python -m AgentKG.src.main query --type experts --filter "Catalog Management"
```

## License

MIT