# app.py

import os
import gradio as gr

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field

# --- API & DB Setup ---
os.environ["GROQ_API_KEY"] = "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"
NEO4J_URI = "neo4j+s://491b8299.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "W3i8UiePw9QyaSJxK9l_apbzUnzh10YWxZQtnpSS02I"

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
llm = ChatGroq(model="llama3-8b-8192")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm_transformer = LLMGraphTransformer(llm=llm)

# --- Entity Extraction Schema ---
class Entities(BaseModel):
    names: list[str] = Field(..., description="All person, org, or business names")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are extracting organization and person entities from the text"),
    ("human", "Use the given format to extract entities:\ninput: {question}")
])
entity_chain = entity_prompt | llm.with_structured_output(Entities)

# --- Helpers ---
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return " AND ".join([f"{word}~2" for word in words])

def structured_retriever(question: str) -> str:
    entities = entity_chain.invoke({"question": question})
    result = ""
    for entity in entities.names:
        cypher = """
        CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
        YIELD node,score
        CALL {
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + '-' + type(r) + '->' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + '-' + type(r) + '->' + node.id AS output
        }
        RETURN output LIMIT 50
        """
        response = graph.query(cypher, {"query": generate_full_text_query(entity)})
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str) -> str:
    structured = structured_retriever(question)
    unstructured = [el.page_content for el in vector_index.similarity_search(question)]
    return f"Structured Data:\n{structured}\n\nUnstructured Data:\n" + "\n---\n".join(unstructured)

# --- RAG Chain ---
template = """Answer the question based only on the context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

qa_prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel({
        "context": RunnableLambda(lambda x: retriever(x["question"])),
        "question": RunnableLambda(lambda x: x["question"]),
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# --- Gradio Pipeline ---
vector_index = None

def process_pdf(pdf_file):
    global vector_index
    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(docs)

    graph_docs = []
    for i in range(0, len(docs_split), 2):
        try:
            graph_docs.extend(llm_transformer.convert_to_graph_documents(docs_split[i:i+2]))
        except Exception as e:
            print(f"Error: {e}")

    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    vector_index = Neo4jVector.from_existing_graph(
        embedding_model,
        search_type="hybrid",
        graph=graph,
        node_label="Document",
        embedding_node_property="embedding",
        text_node_properties=["text"]
    )
    return "PDF uploaded and processed successfully!"

def chat_with_doc(question):
    if vector_index is None:
        return "Please upload and process a PDF first."
    return chain.invoke({"question": question})

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Graph RAG PDF Q&A")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")
    output_info = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer")

    upload_btn.click(process_pdf, inputs=[pdf_input], outputs=[output_info])
    ask_btn.click(chat_with_doc, inputs=[question_input], outputs=[answer_output])

demo.launch()
