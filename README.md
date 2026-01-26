# Academic Affairs Chatbot Multi-Agent System

> Intelligent Q&A chatbot for university academic affairs inquiries

## Overview

[TODO: Project description]

![System Architecture](docs/images/architecture.png)

## Tech Stack

| Category | Technology |
|----------|------------|
| Language |  |
| Framework |  |
| Vector DB |  |
| LLM |  |
| Embedding |  |

## Directory Structure

```
├── src/
│   ├── agents/
│   │   ├── query_refiner/      # Assignee A
│   │   ├── retriever/          # Assignee B
│   │   └── responder/          # Assignee C
│   ├── core/
│   └── api/
├── configs/
├── data/
├── tests/
└── docs/
```

## Installation

```bash
git clone [repository-url]
cd [project-name]
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

```env
LLM_API_KEY=
VECTOR_DB_HOST=
```

## Usage

```bash
# TODO: Add run command
```

## Module Interfaces

### Query Refiner → Retriever

```
Input:  user query (str)
Output: keywords (list), search_query (str)
```

### Retriever → Responder

```
Input:  keywords, search_query
Output: regulations (list)
```

### Responder → Final

```
Input:  regulations
Output: response_type, answer, cited_regulations
```

## Collaboration

### Team

| Module | Assignee |
|--------|----------|
| Query Refiner |  |
| Retriever |  |
| Responder |  |

### Branch Strategy

```
main
 └── develop
      ├── feature/refiner/*
      ├── feature/retriever/*
      └── feature/responder/*
```

### Commit Convention

```
feat(module): description
fix(module): description
docs: description
```

## Testing

```bash
pytest tests/
```

## License

[TODO]
