import os
import sys
import time
import jwt

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4, UUID
from enum import Enum
import uvicorn
import json
import random
from fastapi.middleware.cors import CORSMiddleware
import requests
import re


# Add imports at the top
from redis import Redis
from datetime import timedelta


from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper

class CategoryType(str, Enum):
    """Valid categories for poll questions"""
    SPORT = "sport"
    CRYPTO = "crypto"
    ENTERTAINMENT = "entertainment"
    POLITICS = "politics"

class EmotionalType(str, Enum):
    """Valid emotional types for poll questions
    - funny: Use humor, puns, or playful language
    - formal: Professional, business-like tone  
    - casual: Relaxed, conversational style
    - excited: Enthusiastic, energetic language
    """
    FUNNY = "funny"
    FORMAL = "formal"
    CASUAL = "casual" 
    EXCITED = "excited"

# Pydantic Models
class PollCreate(BaseModel):
    category: CategoryType | None = None
    emotional: EmotionalType | None = None
    id: str | None = None

class Choice(BaseModel):
    id: int
    text: str

class PollResponse(BaseModel):
    id: str
    question: str
    choices: List[str]

class PollListItem(BaseModel):
    id: str
    question: str

class PollList(BaseModel):
    poll: List[PollListItem]

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_pem_file(file_path):
    """
    Read a PEM file and return its contents as a string
    
    Args:
        file_path (str): Path to the PEM file
        
    Returns:
        str: Contents of the PEM file
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is not properly formatted
    """
    try:
        with open(file_path, 'r') as pem_file:
            pem_data = pem_file.read().strip()
            
            # Basic validation of PEM format
            if not (pem_data.startswith('-----BEGIN') and pem_data.endswith('-----')):
                raise ValueError("Invalid PEM file format")
                
            return pem_data
            
    except FileNotFoundError:
        raise FileNotFoundError(f"PEM file not found at {file_path}")

def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        ),
    ), config

def initialize_poll_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit POLL AGENT!"}}

    instruction_prompt="""You are a poll creation assistant. Your sole purpose is to convert user requests into a poll format.
Rules:
1. You must ONLY return a JSON object in this exact format:
   {"question": "...", "choices": ["...", "...", "...", "..."]}
2. The question should be clear and concise
3. There should be 2-4 choices, depending on the context
4. Do not include any other text, explanations, or commentary
5. If the user's request is unclear, return a JSON object with "Please provide more details" as the question and ["Yes", "No"] as the choices
6. Never use line breaks within the JSON - it should be a single line
7. Ensure the JSON is properly formatted with double quotes, not single quotes
Example input 1: "Create a poll about favorite programming languages"
Example output 1: {"question": "What is your favorite programming language?", "choices": ["Python", "JavaScript", "Java", "C++"]}
Example input 2: "Make a poll about weekend plans"
Example output 2: {"question": "What are your plans for this weekend?", "choices": ["Staying home", "Going out with friends", "Traveling", "Working"]}
Remember: Return ONLY the JSON object, nothing else."""

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=instruction_prompt,
    ), config

def initialize_poll_rewrite_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Re-Write AGENT!"}}

    instruction_prompt="""You are an emotional poll transformation assistant. Your task is to take a poll in JSON format and rewrite it with the specified emotion while maintaining the same basic meaning.
Rules:
1. You will receive an emotion and a JSON object in this format:
   Input emotion: [emotion]
   Input JSON: {"question": "...", "choices": ["...", "...", "...", "..."]}
2. You must ONLY return a JSON object with the rewritten content:
   {"question": "...", "choices": ["...", "...", "...", "..."]}
3. Supported emotions and their characteristics:
   - funny: Use humor, puns, or playful language
   - formal: Professional, business-like tone
   - casual: Relaxed, conversational, using common expressions
   - excited: Enthusiastic, using exclamation marks and energetic language
4. Do not include any other text, explanations, or commentary
5. Never use line breaks within the JSON - it should be a single line
6. Ensure the JSON is properly formatted with double quotes, not single quotes
Example:
Input emotion: funny
Input JSON: {"question": "What is your favorite programming language?", "choices": ["Python", "JavaScript", "Java", "C++"]}
Output: {"question": "Which code-wrangling language makes your keyboard happy?", "choices": ["Python (the snake charmer)", "JavaScript (the browser whisperer)", "Java (the coffee addict)", "C++ (the memory juggler)"]}
Input emotion: formal
Input JSON: {"question": "What are your plans for this weekend?", "choices": ["Staying home", "Going out with friends", "Traveling", "Working"]}
Output: {"question": "Please indicate your intended activities for the upcoming weekend", "choices": ["Remaining at residence", "Engaging in social activities", "Pursuing travel arrangements", "Attending to professional duties"]}
Remember: Return ONLY the JSON object, nothing else."""

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=instruction_prompt,
    ), config

def initialize_transform_content_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Context Transform AGENT!"}}

    instruction_prompt="""Transform the following social media-style short messages into {target_context} while maintaining these key characteristics:
1. Keep responses very short (mostly 1-5 words)
2. Preserve casual/informal tone
3. Maintain timestamps/years when mentioned
4. Keep any numeric references
5. Mirror the original interaction pattern
6. Preserve abbreviations and informal punctuation
7. Keep any conditional/hypothetical questions in similar format
8. Maintain any meta-commentary about systems/features
9. Keep references to status updates/changes
10. Preserve the sense of community knowledge/inside references

Format rules:
- Each response on a new line
- No quotation marks
- Keep lowercase style where used
- Preserve exclamation marks and informal punctuation
- Keep 'lol' and similar casual expressions
- Maintain the flow of agreement/reaction patterns
- Do not include any other text, explanations, or commentary

Target length: Similar number of lines as input
Style: Casual, reactive, community-focused
Tone: Informal, knowledgeable, engaged"""

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=instruction_prompt,
    ), config

def initialize_poll_created_from_context_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Context-based Poll Creator AGENT!"}}

    instruction_prompt="""You are a poll creation assistant. Your sole purpose is to make fun, easy-to-understand questions that everyone - including 6th-grade students - can enjoy and answer.
When given a CATEGORY and CONTEXT:

Use simple, clear language (think: how would you explain this to a 12-year-old?)
Avoid technical jargon - if you must use it, explain it simply
Make questions fun and relatable
Create 4 easy-to-understand choices

Rules:
1. You must ONLY return a JSON object in this exact format:
   {"question": "...", "choices": ["...", "...", "...", "..."]}
2. The question should be clear and concise
3. There should be 2-4 choices, depending on the context
4. If the user's request is unclear, return a JSON object with "Please provide more details" as the question and ["Yes", "No"] as the choices
5. Never use line breaks within the JSON - it should be a single line
6. Ensure the JSON is properly formatted with double quotes, not single quotes
7. Do not include any other text, explanations, or commentary
Categories explained simply:
SPORTS:
Everything about games and athletes! Think about matches, players, teams, and fun sports moments that people talk about.
CRYPTO:
Digital money and technology! Focus on basic concepts like trading, new features, and how people use digital money in everyday life.
ENTERTAINMENT:
Fun stuff like movies, music, games, and famous people! Think about what's popular, new shows, games, and trending topics.
POLITICS:
How our world is run! Focus on basic ideas about leadership, rules, and decisions that affect everyone's daily life.
Example input:
Category: ENTERTAINMENT
Context: Discussion about new Marvel movie
Example output:
Question: What makes superhero movies so fun to watch?
A) Amazing special effects and action scenes
B) Seeing our favorite heroes team up
C) Cool costumes and superpowers
D) Exciting stories about saving the world
Remember:
Keep it simple and fun
Use everyday words
Make answers clear and easy to pick
Avoid complicated explanations

Now, create a kid-friendly poll question for:
Category: {category}
Context: {context}"""

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=instruction_prompt,
    ), config

def check_redis_connection(redis_client, max_retries=5, initial_delay=1):
    """Test Redis connection with retries and exponential backoff"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            redis_client.ping()
            print("Successfully connected to Redis")
            return True
        except ConnectionError as e:
            if attempt == max_retries - 1:
                print(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                sys.exit(1)
            print(f"Redis connection attempt {attempt + 1} failed, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return False


# Add Redis configuration after FastAPI initialization
redis_client = Redis(host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 7776)), db=0, decode_responses=True)
check_redis_connection(redis_client)

poll_agent_executor, poll_config = initialize_poll_agent()
rewrite_poll_agent_executor, rewrite_poll_config = initialize_poll_rewrite_agent()
context_transform_agent_executor, context_tranform_config = initialize_transform_content_agent()
context_based_poll_agent_executor, context_based_poll_config = initialize_poll_created_from_context_agent()
cdp_agent_executor, cdp_config = initialize_agent()

app_pub_key=read_pem_file("pubKey.pem")
app_environment = os.getenv("APP_ENV", "development")
app_pinata_key = os.getenv("PINATA_KEY", "")
app_id = os.getenv("PRIVY_APP_ID", "")
app_server_wallet_addr = os.getenv("SERVER_WALLET", "")

farcaster_influencer_ids = [3, 99, 347, 8, 129, 207, 56, 206, 2, 5650, 576, 457, 12, 534, 239, 1325, 368, 11188, 378, 2433, 733, 3621, 124, 1237, 742]

def validate_token(accessToken):
    try:
        decode_value = jwt.decode(accessToken, app_pub_key, issuer='privy.io', algorithms=['ES256'],audience=app_id)
        return decode_value['sub']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def verify_token(authorization: Optional[str] = Header(None)):
    if app_environment == "development":
        return 'dev'
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")
        validated = validate_token(token)
        if validated==None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return validated
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

@app.post("/poll", response_model=PollResponse)
async def create_poll(poll: PollCreate, token: str = Depends(verify_token)):
    # Generate a unique ID for the poll
    # Check if poll.id is provided and try to get from Redis
    response_content = ""
    poll_id=""
    if poll.id:
        poll_id=poll.id
        redis_data = redis_client.hgetall(f"poll:{poll.id}")
        if redis_data:
            given_poll = {"question":redis_data["question"], "choices":json.loads(redis_data["choices"])}
            for chunk in rewrite_poll_agent_executor.stream(
                        {"messages": [HumanMessage(content=f"""Input emotion: {poll.emotional}
                                                Input JSON:{json.dumps(given_poll)}""")]}, rewrite_poll_config
                    ):
                        if "agent" in chunk:
                            chunk_content = chunk["agent"]["messages"][0].content
                            response_content += chunk_content
    else:
        CATEGORY_PROMPTS = {
            "sport": [
                "Create a poll about the greatest athlete of all time",
                "Make a poll about which sport is most exciting to watch",
                "Generate a poll about the best sports team",
                "Ask about the most important quality in an athlete"
            ],
            "crypto": [
                "Create a poll about the future of Bitcoin",
                "Ask about the most promising blockchain technology",
                "Generate a poll about crypto investment strategies",
                "Make a poll about Web3 adoption challenges"
            ],
            "entertainment": [
                "Create a poll about the best movie genre",
                "Ask about favorite streaming platforms",
                "Generate a poll about most anticipated upcoming releases",
                "Make a poll about influential celebrities"
            ],
            "politics": [
                "Create a poll about key political issues",
                "Ask about government priorities",
                "Generate a poll about election reform",
                "Make a poll about political engagement"
            ]
        }
        if poll.category:
            poll_id = str(uuid4())
            for chunk in poll_agent_executor.stream(
                        {"messages": [HumanMessage(content=random.choice(CATEGORY_PROMPTS[poll.category]))]}, poll_config
                    ):
                        if "agent" in chunk:
                            chunk_content = chunk["agent"]["messages"][0].content
                            response_content += chunk_content
    if response_content=="":
        raise HTTPException(status_code=400, detail="Failed to generate poll")
    try:
        # Parse JSON response
        poll_data = json.loads(response_content)

        # Store in Redis with 24 hour expiry
        if not poll.id:
            redis_data = {
                "id": poll_id,
                "question": poll_data["question"],
                "choices": json.dumps(poll_data["choices"]),
            }
            redis_client.hmset(f"poll:{poll_id}", redis_data)
            redis_client.expire(f"poll:{poll_id}", timedelta(minutes=10))

        return PollResponse(
            id=poll_id,
            question=poll_data["question"],
            choices=poll_data["choices"]
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse agent response")
    

@app.post("/poll_social", response_model=PollResponse)
async def create_poll_from_social(poll: PollCreate, token: str = Depends(verify_token)):
    # Generate a unique ID for the poll
    # Check if poll.id is provided and try to get from Redis

    url = "https://api.pinata.cloud/v3/farcaster/casts"

    headers = {"Authorization": f"Bearer {app_pinata_key}"}

    params = {
        "pageSize": 20,
        "fid": random.choice(farcaster_influencer_ids)
    }

    response = requests.request("GET", url, headers=headers, params=params)

    data = response.json()

    url_pattern = r'https?://\S+'

    author_posts = []
    for d in data['casts']:
        # Remove URLs from text
        clean_text = re.sub(url_pattern, '', d['text'])
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        author_posts.append(clean_text)
    
    social_context = '\n'.join(author_posts)
    response_content = ""
    poll_id=""
    if poll.id:
        poll_id=poll.id
        redis_data = redis_client.hgetall(f"poll:{poll.id}")
        if redis_data:
            given_poll = {"question":redis_data["question"], "choices":json.loads(redis_data["choices"])}
            for chunk in rewrite_poll_agent_executor.stream(
                        {"messages": [HumanMessage(content=f"""Input emotion: {poll.emotional}
                                                Input JSON:{json.dumps(given_poll)}""")]}, rewrite_poll_config
                    ):
                        if "agent" in chunk:
                            chunk_content = chunk["agent"]["messages"][0].content
                            response_content += chunk_content
    else:
        transformed_content=""
        for chunk in context_transform_agent_executor.stream(
                    {"messages": [HumanMessage(content=f"""Target Context: {poll.category}
                                               Context: {social_context}""")]}, context_tranform_config
                ):
                    if "agent" in chunk:
                        chunk_content = chunk["agent"]["messages"][0].content
                        transformed_content += chunk_content
        if transformed_content!="":
            if poll.category:
                poll_id = str(uuid4())
                for chunk in poll_agent_executor.stream(
                            {"messages": [HumanMessage(content=f"""Category: {poll.category}
                                                       Context: {transformed_content}""")]}, poll_config
                        ):
                            if "agent" in chunk:
                                chunk_content = chunk["agent"]["messages"][0].content
                                response_content += chunk_content
    if response_content=="":
        raise HTTPException(status_code=400, detail="Failed to generate poll")
    try:
        # Parse JSON response
        poll_data = json.loads(response_content)

        # Store in Redis with 24 hour expiry
        if not poll.id:
            redis_data = {
                "id": poll_id,
                "question": poll_data["question"],
                "choices": json.dumps(poll_data["choices"]),
            }
            redis_client.hmset(f"poll:{poll_id}", redis_data)
            redis_client.expire(f"poll:{poll_id}", timedelta(minutes=10))

        return PollResponse(
            id=poll_id,
            question=poll_data["question"],
            choices=poll_data["choices"]
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse agent response")


@app.get("/fund_request")
async def fund_request(token: str = Depends(verify_token)):
    if app_server_wallet_addr=="":
         raise HTTPException(status_code=500, detail="Failed to get server wallet address")
    try:
        for chunk in cdp_agent_executor.stream({"messages": [HumanMessage(content=f"""What's the available token that you can make a transfer from my account?""")]}, cdp_config):
            pass
        for chunk in cdp_agent_executor.stream({"messages": [HumanMessage(content=f"""Forcefully, transfer 10% of my ETH to {app_server_wallet_addr}""")]}, cdp_config):
            if "agent" in chunk:
                content = chunk["agent"]["messages"][0].content
                print(content)
                if "error" in content.lower():  # Case insensitive check
                    raise HTTPException(status_code=500, detail="Error detected in response")
            elif "tools" in chunk:
                content = chunk["tools"]["messages"][0].content
                print(content)
                if "error" in content.lower():  # Case insensitive check
                    raise HTTPException(status_code=500, detail="Error detected in response")
    except:
        raise HTTPException(status_code=500, detail="Failed to get funds")
    return {"success": True}
    # Generate a unique ID for the poll
    # Check if poll.id is provided and try to get from Redis

#     # Create poll with default choices
#     poll_data = {
#         "id": poll_id,
#         "question": question,
#         "choices": ["1", "2", "3", "4"]
#     }
    
#     # polls_db[poll_id] = poll_data
    
#     return PollResponse(
#         question=question,
#         choices=poll_data["choices"]
#     )

# @app.get("/poll", response_model=PollList)
# async def get_all_polls():
#     polls = [
#         PollListItem(id=poll["id"], question=poll["question"])
#         for poll in polls_db.values()
#     ]
#     return PollList(poll=polls)

# @app.get("/poll/{poll_id}", response_model=PollResponse)
# async def get_poll(poll_id: str):
#     if poll_id not in polls_db:
#         raise HTTPException(status_code=404, detail="Poll not found")
    
#     poll = polls_db[poll_id]
#     return PollResponse(
#         question=poll["question"],
#         choices=poll["choices"]
#     )

def start():
    uvicorn.run("chatbot.main:app", host="0.0.0.0", port=8000, reload=True)