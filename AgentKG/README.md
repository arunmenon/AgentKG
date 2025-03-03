# AgentKG - Agent Knowledge Graph

This project implements a comprehensive knowledge graph for agent orchestration using Neo4j. The graph schema organizes:

- Business processes and sub-processes
- Agents and crews
- Business functions and technical expertise taxonomies
- Success metrics and performance records

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with Neo4j credentials:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

3. Run the schema creation script:
```
python -m src.schema_creator
```

4. Run the knowledge ingestion script:
```
python -m src.knowledge_ingestion
```

## Components

- `neo4j_connector.py`: Handles database connections and transactions
- `schema_creator.py`: Creates the Neo4j schema with constraints and indexes
- `knowledge_ingestion.py`: Uses LLMs to populate the graph with domain knowledge