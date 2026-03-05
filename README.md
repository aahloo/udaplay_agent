# UdaPlay - AI Game Research Agent 🎮🤖

> **Agentic AI Capstone Project**: An intelligent research agent for the video game industry that combines local knowledge retrieval (RAG) with web search capabilities and autonomous decision-making.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange.svg)](https://www.trychroma.com/)

---

## 🎯 Overview

**UdaPlay** is an advanced AI research agent that specializes in video game industry knowledge. It intelligently combines:

- 🗄️ **Local Vector Database**: ChromaDB with 15 curated video game entries
- 🔍 **Web Search Integration**: Tavily API for up-to-date information
- 🧠 **LLM-as-Judge**: Autonomous evaluation of retrieval quality
- 💾 **Long-term Memory**: Learns from web searches and improves over time
- 💬 **Stateful Conversations**: Maintains context across multiple queries

### What Makes It Special?

UdaPlay doesn't just retrieve information—it **learns**:

1. **Retrieves** from local knowledge base
2. **Evaluates** if the information is sufficient
3. **Decides** whether to use local knowledge or search the web
4. **Learns** from web searches, storing results for future queries
5. **Cites** sources appropriately based on information origin

This is **Agentic RAG** in action—autonomous decision-making with retrieval-augmented generation.

---

## ✨ Key Features

### Core Capabilities

✅ **Persistent Vector Database**
- ChromaDB with OpenAI embeddings
- 15 pre-loaded video game entries
- Semantic search with metadata preservation

✅ **Three Intelligent Tools**
- `retrieve_game`: Searches local database and learned knowledge
- `evaluate_retrieval`: LLM-as-judge for quality assessment
- `game_web_search`: Web search with automatic knowledge storage

✅ **Stateful Agent**
- Built on custom StateMachine implementation
- Short-term memory for conversation context
- Session-based state management

✅ **Structured Outputs**
- Pydantic models for reliable parsing
- Type-safe evaluation results
- Consistent response formats

### Advanced Features

🚀 **Long-term Memory System**
- Dual ChromaDB collections (original + learned)
- Automatic storage of web search results
- Knowledge persists across sessions
- Reduces redundant web searches

🎯 **Citation Validation**
- Programmatic verification of source attribution
- Detects database, web, and memory citations
- Tracks tool usage and response quality

📊 **Comprehensive Testing**
- 6+ test scenarios covering all use cases
- Citation validation framework
- Conversation continuity tests
- Performance metrics and logging

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UdaPlay Agent                                 │
│                  (GPT-4o-mini + StateMachine)                    │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ├──────────────┬──────────────────┬─────────────────┐
               ▼              ▼                  ▼                 ▼
    ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────┐
    │retrieve_game │  │evaluate_       │  │game_web_     │  │ Memory   │
    │   (Tool 1)   │  │retrieval       │  │search        │  │ System   │
    │              │  │   (Tool 2)     │  │   (Tool 3)   │  │          │
    └──────┬───────┘  └────────┬───────┘  └──────┬───────┘  └────┬─────┘
           │                   │                  │               │
           ▼                   ▼                  ▼               │
    ┌──────────────────────────────────────────────────┐         │
    │          ChromaDB (Persistent Storage)           │         │
    │  ┌────────────────┐    ┌────────────────────┐   │         │
    │  │ udaplay        │    │ udaplay_learned    │   │◄────────┘
    │  │ (15 games)     │    │ (web results)      │   │
    │  └────────────────┘    └────────────────────┘   │
    └──────────────────────────────────────────────────┘
                              │
                              │ (if insufficient)
                              ▼
                    ┌──────────────────┐
                    │   Tavily API     │
                    │  (Web Search)    │
                    └──────────────────┘
```

### Prerequisites

- **Python 3.13** (NOT 3.14 - ChromaDB compatibility)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Tavily API Key** ([Get one here](https://tavily.com/))

### Running the Project

#### Part 1: Set up Vector Database
1. Open `Udaplay_01_solution_project.ipynb`
2. Run all cells to create and populate ChromaDB
3. Verify: Should see "Documents in collection: 15"

#### Part 2: Run the Agent
1. Open `Udaplay_02_solution_project.ipynb`
2. Run all cells to initialize agent and run tests
3. Explore: Try your own queries with the agent!

---

## 💡 Usage Examples

### Example 1: Database Query

```python
# Query internal knowledge
run = agent.invoke(
    query="When was Pokémon Gold and Silver released?",
    session_id="demo"
)

# Output:
# "Pokémon Gold and Silver were released in 1999 for the Game Boy Color..."
```

**Agent Behavior**:
- ✅ Searches local database
- ✅ Evaluates sufficiency
- ✅ Answers without web search

---

### Example 2: Web Search Fallback

```python
# Query requiring web search
run = agent.invoke(
    query="Was Mortal Kombat X released for PlayStation 5?",
    session_id="demo"
)

# Output:
# "Mortal Kombat X was not originally released for the PlayStation 5...
#  [source: PlayStation, YouTube]"
```

**Agent Behavior**:
- ✅ Searches local database (not found)
- ✅ Evaluates (insufficient)
- ✅ Falls back to web search
- ✅ Stores results in learned collection
- ✅ Provides cited answer

---

### Example 3: Conversation Continuity

```python
# First query
run1 = agent.invoke(
    query="What platform was GoldenEye 007 released on?",
    session_id="conversation"
)
# Output: "GoldenEye 007 was originally released on the Nintendo 64..."

# Follow-up query (same session)
run2 = agent.invoke(
    query="And what is its Metacritic score?",
    session_id="conversation"
)
# Output: "The original GoldenEye 007 for the Nintendo 64 has a
#          Metacritic score of 96..."
```

**Agent Behavior**:
- ✅ Understands "its" refers to GoldenEye 007
- ✅ Maintains conversation context
- ✅ No need to re-explain

---

### Example 4: Long-term Memory

```python
# First query on new topic
run1 = agent.invoke(
    query="What is The Witcher 3's Metacritic score?",
    session_id="learning"
)
# → Web search → Store in learned collection

# Later query on same topic (even different session!)
run2 = agent.invoke(
    query="Tell me about Witcher 3's critical reception",
    session_id="new_session"
)
# → Retrieves from learned collection (no web search needed!)
```

**Agent Behavior**:
- ✅ Learns from web searches
- ✅ Knowledge persists across sessions
- ✅ Faster responses for repeated topics
- ✅ Reduces API costs

---

## 🧪 Testing

### Test Coverage

The project includes comprehensive testing across multiple scenarios:

#### Required Tests (Rubric)
- ✅ **Query 1**: Internal database retrieval
- ✅ **Query 2**: Specific detail queries
- ✅ **Query 3**: Web search fallback

#### Additional Tests
- ✅ **Conversation continuity**: Context maintenance across queries
- ✅ **Long-term memory**: Knowledge storage and retrieval
- ✅ **Citation validation**: Source attribution verification

### Running Tests

```bash
# Open Part 2 notebook
jupyter notebook Udaplay_02_solution_project.ipynb

# Run test cells:
# - Main Agent Testing (3 required queries)
# - Conversation Memory Testing (2 queries)
# - Long-term Memory Testing (2 queries)
```

### Test Results Summary

| Test Category | Queries | Pass Rate |
|--------------|---------|-----------|
| Basic Retrieval | 3 | 100% ✅ |
| Web Search Fallback | 1 | 100% ✅ |
| Conversation Memory | 2 | 100% ✅ |
| Long-term Memory | 2 | 100% ✅ |
| Citation Validation | 6 | 67% ⚠️ |

**Note**: Citation detection could be improved with additional prompt engineering.


