# 🧠 Wikipedia Knowledge Graph using spaCy, NetworkX & Streamlit

This app builds an interactive knowledge graph from Wikipedia articles using NLP techniques.

## 🔍 Features
- Fetches Wikipedia content using `wikipedia` library
- Cleans and preprocesses text
- Resolves coreferences using `coreferee`
- Extracts subject-predicate-object triples
- Visualizes graph using PyVis
- Lets you query the graph for related entities

## 🛠️ Tech Stack
- Python
- Streamlit
- spaCy (`en_core_web_lg`)
- coreferee
- NetworkX
- PyVis

## 🚀 Run Locally

### 1. Clone this repo

```bash
git clone https://github.com/<your-username>/wiki-knowledge-graph
cd wiki-knowledge-graph
