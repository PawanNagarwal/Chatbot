import streamlit as st
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver


# Page configuration
st.set_page_config(
    page_title="Sarvika Technologies Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── State ──────────────────────────────────────────────────────────────────────
class AgenticRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    retrieval_attempts: int


# ── Grading / Classification models ───────────────────────────────────────────
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class ClassifyIntent(BaseModel):
    intent: Literal["conversational", "rag"] = Field(
        description=(
            "'conversational' for greetings, small talk, thanks, or general chat. "
            "'rag' for questions that need specific company/product knowledge."
        )
    )


# ── Chatbot class ──────────────────────────────────────────────────────────────
class AgenticRAGChatbot2:
    def __init__(self, vector_db_path: str = "./sarvika_faiss"):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        self.grader_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.graph = self._build_graph()

    # ── NEW: Intent classification node ───────────────────────────────────────
    def classify_intent_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Dummy state-pass node; routing logic lives in classify_intent_router."""
        return {}

    def classify_intent_router(self, state: AgenticRAGState) -> Literal["conversational", "retrieve"]:
        question = state["question"]

        classify_prompt = f"""Classify the user's message intent into one of two categories:
- 'conversational': greetings, small talk, expressions of thanks, or any general casual message
  (e.g., "hi", "good morning", "thanks", "how are you", "ok", "bye")
- 'rag': any question or request that requires specific knowledge about Sarvika Technologies,
  its services, team, projects, or capabilities

Message: {question}"""

        response = self.grader_llm.with_structured_output(ClassifyIntent).invoke(
            [{"role": "user", "content": classify_prompt}]
        )
        return "conversational" if response.intent == "conversational" else "retrieve"

    # ── NEW: Direct conversational response node ───────────────────────────────
    def respond_directly(self, state: AgenticRAGState) -> AgenticRAGState:
        question = state["question"]

        direct_prompt = f"""You are a friendly assistant for Sarvika Technologies, an IT outsourcing and AI services company.
Respond naturally and warmly to this conversational message.
Keep it brief and professional. Do NOT invent or assume any company-specific information.

Message: {question}
Response:"""

        response = self.llm.invoke([{"role": "user", "content": direct_prompt}])
        return {"messages": [AIMessage(content=response.content)]}

    # ── Existing nodes (unchanged) ─────────────────────────────────────────────
    def retrieve_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        question = state["question"]
        documents = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in documents])
        return {
            "context": context,
            "messages": [HumanMessage(content="Retrieved context")]
        }

    def grade_documents(self, state: AgenticRAGState) -> Literal["generate", "rewrite"]:
        attempts = state.get("retrieval_attempts", 0)
        if attempts >= 2:
            return "generate"

        question = state["question"]
        context = state["context"]

        grade_prompt = f"""You are a grader assessing relevance of retrieved documents to a user question.

Question: {question}

Retrieved Context: {context}

Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question."""

        response = self.grader_llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": grade_prompt}]
        )
        return "generate" if response.binary_score == "yes" else "rewrite"

    def rewrite_question(self, state: AgenticRAGState) -> AgenticRAGState:
        question = state["question"]
        attempts = state.get("retrieval_attempts", 0)

        rewrite_prompt = f"""Look at the input question and reason about the underlying semantic intent.

Original question: {question}

Generate an improved, more specific question that will retrieve better results about Sarvika Technologies.

Improved question:"""

        response = self.llm.invoke([{"role": "user", "content": rewrite_prompt}])
        return {
            "question": response.content,
            "retrieval_attempts": attempts + 1,
            "messages": [HumanMessage(content=f"Rewritten query: {response.content}")]
        }

    def generate_answer(self, state: AgenticRAGState) -> AgenticRAGState:
        question = state["question"]
        context = state["context"]
        conversation_history = state.get("messages", [])

        chat_history = [
            msg for msg in conversation_history
            if isinstance(msg, (HumanMessage, AIMessage))
        ]

        history_text = ""
        if len(chat_history) > 1:
            history_text = "\n\nPrevious conversation:\n"
            for msg in chat_history[:-1]:
                if isinstance(msg, HumanMessage) and not msg.content.startswith(("Retrieved", "Rewritten")):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"

        generate_prompt = f"""You are a helpful assistant for Sarvika Technologies, an IT outsourcing and AI services company.

Use the following context to answer the question. Be specific, professional, and helpful.
If the context doesn't contain relevant information, say: "I don't have specific information about that. Please contact Sarvika directly."
{history_text}

Context: {context}

Question: {question}

Answer:"""

        response = self.llm.invoke([{"role": "user", "content": generate_prompt}])
        return {"messages": [AIMessage(content=response.content)]}

    # ── Graph ──────────────────────────────────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgenticRAGState)

        # Register nodes
        workflow.add_node("classify", self.classify_intent_node)
        workflow.add_node("respond_directly", self.respond_directly)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("rewrite", self.rewrite_question)
        workflow.add_node("generate", self.generate_answer)

        # START → classify, then branch based on intent
        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self.classify_intent_router,
            {
                "conversational": "respond_directly",
                "retrieve": "retrieve"
            }
        )

        # Conversational path ends immediately
        workflow.add_edge("respond_directly", END)

        # RAG path (unchanged)
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite"
            }
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def query(self, question: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "retrieval_attempts": 0
        }
        result = self.graph.invoke(initial_state, config)

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I apologize, but I couldn't find relevant information to answer your question."


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "sarvika_chat_001"


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 Sarvika Technologies Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Chatbot</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("⚙️ Configuration")

    vector_db_path = st.text_input(
        "Vector DB Path",
        value="./sarvika_faiss",
        help="Path to your FAISS vector database"
    )

    if st.button("🔄 Initialize Chatbot"):
        with st.spinner("Loading chatbot..."):
            try:
                st.session_state.chatbot = AgenticRAGChatbot2(vector_db_path=vector_db_path)
                st.success("✅ Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {str(e)}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        import random
        st.session_state.thread_id = f"sarvika_chat_{random.randint(1000, 9999)}"
        st.rerun()

    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    - What services does Sarvika offer?
    - Tell me about your AI capabilities
    - How can I contact Sarvika?
    - What industries do you serve?
    """)


# ── Chat interface ─────────────────────────────────────────────────────────────
if st.session_state.chatbot is None:
    st.markdown('<div class="info-box">⚠️ Please initialize the chatbot using the sidebar configuration.</div>', unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)

    user_question = st.chat_input("Ask me anything about Sarvika Technologies...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("🔍 Thinking..."):
            try:
                response = st.session_state.chatbot.query(
                    user_question,
                    thread_id=st.session_state.thread_id
                )
                st.session_state.messages.append({"role": "assistant", "content": response})

                with st.chat_message("assistant"):
                    st.write(response)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
