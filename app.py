# Import necessary libraries
import wikipedia as wp 
import re
import requests
import spacy
import spacy_transformers
from spacy import displacy
from spacy.matcher import Matcher
import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile
import os
from streamlit.components.v1 import html
# Extract subject-predicate-object triples from a sentence
def extract_relationship(sentence):
    doc = nlp(sentence)

    first, last = None, None

    for chunk in doc.noun_chunks:
        if not first:
            first = chunk
        else:
            last = chunk

    if first and last:
        return (first.text.strip(), last.text.strip(), str(doc[first.end:last.start]).strip())

    return (None, None, None)

#define a helper function for summarizing relationship text
def print_five_words(text):
    words = text.split()
    return " ".join(words[:5]) + ("..." if len(words) > 5 else "")

# Streamlit app starts here
st.title("Knowledge Graph from Wikipedia Text")
st.markdown("Enter a Wikipedia page title to extract knowledge and build a graph.")

# User input for Wikipedia page title
user_input = st.text_input("Enter Wikipedia Page Title", value="New York City")

# Load data when button is clicked
#Clean and simplify the Wikipedia content.
#Display it in a collapsible section for user inspection.
if st.button("Fetch Wikipedia Content"):
    try:
        wp.set_lang("en")
        data = wp.page(user_input).content
        
        st.success(f"Successfully fetched content for '{user_input}'")

        # Preprocess the text
        data = data.lower().replace('\n', "")
        data = re.sub('== see also ==.*|[@#:&\"]|===.*?===|==.*?==|\(.*?\)', '', data)

        # Show preprocessed data
        with st.expander("🧹 Show Preprocessed Text"):
            st.write(data)

        # Load spaCy model
        nlp = spacy.load("en_core_web_lg")

        # Add coreferee to pipeline
        try:
            nlp.add_pipe('coreferee')
        except ValueError:
            st.warning("Coreferee is already in the pipeline or failed to add.")
        
        #Run NLP pipeline
        doc = nlp(data)

        # Display named entities
        with st.expander("🧠 Named Entity Recognition (NER)"):
            html_content = displacy.render(doc, style="ent", page=True)
            st.components.v1.html(html_content, scrolling=True, height=500)
        
        # Display coreference chains
        with st.expander("🔁 Coreference Resolution"):
            try:
                coref_output = []
                for chain in doc._.coref_chains:
                    mentions = [doc[span.start:span.end].text for span in chain]
                    coref_output.append(" → ".join(mentions))
                if coref_output:
                    for i, item in enumerate(coref_output):
                        st.markdown(f"**Chain {i+1}:** {item}")
                else:
                    st.info("No coreference chains found.")
            except Exception as e:
                st.error(f"Coreference resolution failed: {e}")
        
        # Resolve coreferences in the text
        resolved_data = ""
        try:
            for token in doc:
                resolved_coref = doc._.coref_chains.resolve(token)
                if resolved_coref:
                    resolved_data += " " + " and ".join(r.text for r in resolved_coref)
                elif token.dep_ == "punct":
                    resolved_data += token.text
                else:
                    resolved_data += " " + token.text
            # Display resolved data
            with st.expander("🧾 Coreference-Resolved Text"):
                st.write(resolved_data.strip())
        except Exception as e:
            st.error(f"Error while resolving coreferences: {e}")

        # Build Knowledge Graph
        try:
            st.subheader("🧠 Knowledge Graph")

            graph_doc = nlp(resolved_data)
            nx_graph = nx.DiGraph()

            for sent in graph_doc.sents:
                if len(sent) > 3:
                    a, b, c = extract_relationship(str(sent))
                    if a and b:
                        nx_graph.add_node(a, size=5)
                        nx_graph.add_node(b, size=5)
                        nx_graph.add_edge(a, b, weight=1, title=print_five_words(c), arrows="to")

            # Create PyVis Network
            g = Network(height="500px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
            g.from_nx(nx_graph)

            # Save and display in Streamlit
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            g.show(tmp_file.name)

            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_graph = f.read()
                st.components.v1.html(html_graph, height=550, scrolling=True)

        except Exception as e:
            st.error(f"Error while building the knowledge graph: {e}")

        # Query related entities
        st.subheader("🔍 Query Related Entities in the Graph")
        node_query = st.text_input("Enter an entity to explore its connections", value="manhattan")

        if st.button("Show Related Entities"):
            if node_query in nx_graph.nodes:
                neighbors = list(nx_graph.edges([node_query]))
                if neighbors:
                    st.write(f"Entities connected to **{node_query}**:")
                    for edge in neighbors:
                        st.markdown(f"- {edge[1]}")
                else:
                    st.info(f"No related entities found for '{node_query}'.")
            else:
                st.warning(f"'{node_query}' does not exist in the graph.")


    except Exception as e:
        st.error(f"Failed to process Wikipedia content: {e}")



