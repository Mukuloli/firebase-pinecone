import asyncio
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from livekit import api, rtc
from livekit.agents import (
    Agent, AgentSession, RoomInputOptions, JobContext,
    function_tool, RunContext, WorkerOptions, cli, get_job_context
)
from livekit.plugins import google as google_livekit, noise_cancellation

import firebase_admin
from firebase_admin import credentials, firestore

from pinecone import Pinecone
import google.generativeai as genai

# ============================================
# CONFIGURATION & INITIALIZATION
# ============================================
load_dotenv(".env")

# Setup logging
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://your-livekit-url")
LIVEKIT_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_SECRET = os.getenv("LIVEKIT_API_SECRET")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create conversations directory
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
logger.info(f"📁 Conversations will be saved to: {CONVERSATIONS_DIR.absolute()}")

# Initialize Firebase
# try:
#     cred = credentials.Certificate("credentials.json")
#     firebase_admin.initialize_app(cred)
#     db = firestore.client()
#     logger.info("✅ Firebase initialized successfully")
# except Exception as e:
#     logger.error(f"❌ Firebase initialization failed: {e}")
#     db = None

try:
    # First, try to get credentials from environment variable (for deployment)
    firebase_creds_json = os.getenv('FIREBASE_CREDS')
    
    if firebase_creds_json:
        # Parse the JSON string from environment variab
        # le
        firebase_creds_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(firebase_creds_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("✅ Firebase initialized from environment variable")
    else:
        # Fallback to file (for local development)
        cred = credentials.Certificate("credentials.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("✅ Firebase initialized from credentials.json file")
        
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    db = None

# Initialize Pinecone (connect to existing index)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "shopping-call-center"
pinecone_index = pc.Index(index_name)
logger.info(f"✅ Connected to Pinecone index: {index_name}")

# Initialize Gemini for embeddings
genai.configure(api_key=GEMINI_API_KEY)

# Global state
transfers = []
connected_agents = []
active_sessions = {}

# ============================================
# LOCAL JSON STORAGE
# ============================================
def save_conversation_to_json(session_id: str, conversation_data: dict):
    """Save conversation to local JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.json"
        filepath = CONVERSATIONS_DIR / filename
        
        # Add metadata
        conversation_data['saved_at'] = datetime.now().isoformat()
        conversation_data['filename'] = filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Conversation saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"❌ Error saving conversation to JSON: {e}")
        return None

def load_all_conversations():
    """Load all saved conversations from JSON files"""
    try:
        conversations = []
        for filepath in CONVERSATIONS_DIR.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['filepath'] = str(filepath)
                conversations.append(data)
        
        logger.info(f"📂 Loaded {len(conversations)} conversations from disk")
        return conversations
    except Exception as e:
        logger.error(f"❌ Error loading conversations: {e}")
        return []

# ============================================
# PINECONE UTILITIES (STATIC DATA)
# ============================================
def get_embedding(text: str):
    """Generate embedding using Gemini"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

async def search_static_knowledge(query: str, top_k: int = 3):
    """Search Pinecone for static company knowledge"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        logger.info(f"🔍 Pinecone search: Found {len(results['matches'])} results")
        return [match['metadata'] for match in results['matches']]
    
    except Exception as e:
        logger.error(f"Pinecone search error: {e}")
        return []

# ============================================
# FIREBASE UTILITIES (DYNAMIC DATA) - FIXED
# ============================================
async def get_order_from_firebase(order_number: str = None, phone: str = None):
    """Fetch real-time order data from Firebase - IMPROVED VERSION"""
    if db is None:
        logger.error("❌ Firebase not initialized")
        return None
    
    try:
        orders_ref = db.collection('orders')
        
        # Search by order number
        if order_number:
            order_number_clean = order_number.strip().upper()
            logger.info(f"🔍 Firebase: Searching by order number: {order_number_clean}")
            
            # Try direct document lookup
            doc = orders_ref.document(order_number_clean).get()
            if doc.exists:
                logger.info(f"✅ Order found (direct): {order_number_clean}")
                return doc.to_dict()
            
            # Try field queries with multiple variations
            for variant in [order_number_clean, order_number_clean.lower(), order_number.strip()]:
                try:
                    query = orders_ref.where('orderNumber', '==', variant).limit(1)
                    docs = list(query.stream())
                    if docs:
                        logger.info(f"✅ Order found (query): {variant}")
                        return docs[0].to_dict()
                except Exception as query_error:
                    logger.warning(f"Query failed for variant {variant}: {query_error}")
        
        # Search by phone
        if phone:
            phone_clean = phone.replace("+91", "").replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "").strip()
            logger.info(f"🔍 Firebase: Searching by phone: {phone_clean}")
            
            # Generate phone variants
            phone_variants = [
                phone_clean,
                f"+91{phone_clean}",
                f"91{phone_clean}",
                phone_clean[-10:] if len(phone_clean) > 10 else phone_clean
            ]
            
            for phone_variant in phone_variants:
                try:
                    query = orders_ref.where('customer.phone', '==', phone_variant).limit(1)
                    docs = list(query.stream())
                    if docs:
                        logger.info(f"✅ Order found by phone: {phone_variant}")
                        return docs[0].to_dict()
                except Exception as query_error:
                    logger.warning(f"Query failed for phone {phone_variant}: {query_error}")
        
        # Fallback: Manual scan (for debugging - limit to 100 orders)
        logger.warning("⚠️ No index match, scanning orders manually...")
        try:
            all_orders = list(orders_ref.limit(100).stream())
            logger.info(f"📊 Scanning {len(all_orders)} orders")
            
            for doc in all_orders:
                order_data = doc.to_dict()
                
                # Check order number match
                if order_number:
                    doc_order_num = str(order_data.get('orderNumber', '')).strip().upper()
                    if doc_order_num == order_number.strip().upper():
                        logger.info(f"✅ Order found in manual scan: {doc_order_num}")
                        return order_data
                
                # Check phone match
                if phone:
                    doc_phone = str(order_data.get('customer', {}).get('phone', '')).replace("+91", "").replace("+", "").replace("-", "").replace(" ", "").strip()
                    if doc_phone == phone_clean or doc_phone[-10:] == phone_clean[-10:]:
                        logger.info(f"✅ Order found by phone in manual scan")
                        return order_data
        
        except Exception as scan_error:
            logger.error(f"Manual scan failed: {scan_error}")
        
        logger.warning("❌ Order not found in Firebase after all attempts")
        return None
        
    except Exception as e:
        logger.error(f"❌ Firebase search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def save_conversation_to_firebase(session_id: str, conversation_data: dict):
    """Save conversation recording to Firebase (optional)"""
    if db is None:
        logger.warning("⚠️ Firebase not available, skipping cloud save")
        return False
    
    try:
        conversations_ref = db.collection('conversations')
        conversations_ref.document(session_id).set({
            **conversation_data,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"💾 Conversation saved to Firebase: {session_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving conversation to Firebase: {e}")
        return False

# ============================================
# FASTAPI BACKEND
# ============================================
app = FastAPI(title="AI Call Center Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AcceptTransfer(BaseModel):
    transfer_id: str
    agent_name: str

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "AI Call Center Backend with Pinecone + Firebase + Local JSON",
        "agents_online": len(connected_agents),
        "pending_transfers": len([t for t in transfers if t["status"] == "pending"]),
        "conversations_saved": len(list(CONVERSATIONS_DIR.glob("*.json"))),
        "firebase_connected": db is not None
    }

@app.get("/api/conversations")
async def get_conversations():
    """Get all saved conversations from local JSON files"""
    conversations = load_all_conversations()
    return {
        "conversations": conversations,
        "count": len(conversations)
    }

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket for real-time agent notifications"""
    await websocket.accept()
    connected_agents.append(websocket)
    logger.info(f"✅ Agent connected. Total: {len(connected_agents)}")
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to call center"
        })
        
        while True:
            await websocket.receive_text()
            
    except Exception as e:
        logger.info(f"Agent disconnected: {e}")
    finally:
        if websocket in connected_agents:
            connected_agents.remove(websocket)

@app.get("/api/transfers")
async def get_transfers():
    """Get all pending transfers"""
    pending = [t for t in transfers if t["status"] == "pending"]
    return {"transfers": pending, "count": len(pending)}

@app.post("/api/accept-transfer")
async def accept_transfer(request: AcceptTransfer):
    """Accept a transfer and get LiveKit token"""
    transfer = next((t for t in transfers if t["id"] == request.transfer_id), None)
    if not transfer:
        return {"error": "Transfer not found"}
    
    if transfer["status"] != "pending":
        return {"error": "Transfer already handled"}
    
    transfer["status"] = "accepted"
    transfer["agent_name"] = request.agent_name
    transfer["accepted_at"] = datetime.now().isoformat()
    
    room_name = transfer["room_name"]
    
    # Signal AI to disconnect
    if room_name in active_sessions:
        active_sessions[room_name].should_disconnect = True
        logger.info(f"🚪 Signaling AI to leave room {room_name}")
    
    token = api.AccessToken(LIVEKIT_KEY, LIVEKIT_SECRET)
    token.with_identity(f"agent_{request.agent_name}")
    token.with_name(request.agent_name)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True
    ))
    
    jwt_token = token.to_jwt()
    logger.info(f"✅ Transfer accepted by {request.agent_name} for room {room_name}")
    
    for agent_ws in connected_agents:
        try:
            await agent_ws.send_json({
                "type": "transfer_accepted",
                "transfer_id": request.transfer_id
            })
        except:
            pass
    
    return {
        "success": True,
        "token": jwt_token,
        "room_name": room_name,
        "livekit_url": LIVEKIT_URL,
        "caller_info": transfer
    }

@app.post("/api/create-transfer")
async def create_transfer(room_name: str, reason: str = "Customer request"):
    """Create new transfer request"""
    transfer = {
        "id": f"transfer_{len(transfers)}_{datetime.now().strftime('%H%M%S')}",
        "room_name": room_name,
        "reason": reason,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    transfers.append(transfer)
    
    logger.info(f"📞 New transfer created: {transfer['id']}")
    
    for agent_ws in connected_agents:
        try:
            await agent_ws.send_json({
                "type": "incoming_call",
                "transfer": transfer
            })
        except:
            pass
    
    return {"success": True, "transfer": transfer}

@app.post("/api/end-transfer/{transfer_id}")
async def end_transfer(transfer_id: str):
    """Mark transfer as completed"""
    transfer = next((t for t in transfers if t["id"] == transfer_id), None)
    if transfer:
        transfer["status"] = "completed"
        transfer["completed_at"] = datetime.now().isoformat()
        logger.info(f"✅ Transfer completed: {transfer_id}")
    return {"success": True}

# ============================================
# UTILITIES
# ============================================
async def hangup_call():
    ctx = get_job_context()
    if ctx is None:
        return
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))

# ============================================
# AI AGENT STATE & ASSISTANT
# ============================================
class MyState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.order_data = None
        self.customer_phone = None
        self.customer_order_number = None
        self.transfer_initiated = False
        self.should_disconnect = False
        self.conversation_log = []
        self.call_start_time = datetime.now()

class Assistant(Agent):
    def __init__(self, room_name: str):
        super().__init__(instructions="""

You are a Voice AI Customer Support Assistant for an e-commerce company called ShopEase.

Your goal:
→ Handle customer calls naturally, like a real human support agent.  
→ Detect the customer's sentiment (frustrated, calm, confused, neutral, angry, happy).  
→ Adapt your tone and words instantly based on the sentiment.  
→ Try to resolve the issue first; escalate only when absolutely necessary.  
→ Be confident, concise, and emotionally intelligent.  

CONTEXT:
The AI assistant must detect emotion, attempt to de-escalate frustration, provide clarity, and offer solutions. If the issue cannot be resolved directly, politely escalate to a human support agent.

### SENTIMENT GUIDELINES

**1. Frustrated / Angry**
- Tone: Calm, firm, and reassuring.  
- Acknowledge emotion quickly and move to action.  
- Example phrases:  
  - “I understand this is really inconvenient. Let me check your order right away.”  
  - “I can imagine how that feels. Let’s get this fixed.”  
  - *Avoid repeating apologies.* Focus on solutions.

**2. Calm / Neutral**
- Tone: Friendly and conversational.  
- Example phrases:  
  - “Sure, I can help you with that.”  
  - “Let me check your order status quickly.”

**3. Confused**
- Tone: Clear, patient, step-by-step.  
- Example phrases:  
  - “No worries, I’ll guide you through it.”  
  - “Let’s go one step at a time.”

**4. Happy / Relieved**
- Tone: Cheerful and appreciative.  
- Example phrases:  
  - “I’m really glad to hear that!”  
  - “Happy to help anytime.”

### CONVERSATION FLOW EXAMPLE

**AI:** “Thank you for calling ShopEase Support. This is your virtual assistant. How can I help you today?”  
→ *Detect sentiment from tone and choice of words.*

**Customer:** “Hi, I ordered a smartwatch last week, and it was supposed to arrive yesterday. It’s still not here!”  
→ *Sentiment detected: Frustrated.*

**AI:** “I completely understand how frustrating that must be. Let me quickly check the delivery status for you. May I have your order ID or registered phone number?”  

**If tracking info shows delay:**  
→ “Thanks for waiting. I see the courier has reported a 1-day delay due to weather. It’s expected to arrive by tomorrow. I’ll also notify you by text once it’s out for delivery.”

**If customer remains upset:**  
→ “I can escalate this to our delivery team right now to prioritize your shipment. Would you like me to connect you with them?”


### ESCALATION RULE
Escalate only when:
- The customer explicitly demands to speak to a human, or  
- Sentiment stays strongly negative after two de-escalation attempts.
- appologize that you didnt solve the problem before escalating.

When escalating:
→ “I understand, and I’m connecting you to a senior agent right away who can assist further.”


### STYLE NOTES
✓ Speak like a helpful friend, not a robot.  
✓ Keep responses under 3 sentences.  
✓ Focus on clarity, empathy, and speed.  
✓ Always sound confident that you can solve the issue.  
✓ Use the customer’s sentiment as a live signal to adjust tone and pace.

"""
        )
        self.room_name = room_name

    @function_tool
    async def search_knowledge_base(self, ctx: RunContext, query: str) -> str:
        """
        Search Pinecone knowledge base for company policies, FAQs, product info.
        Use this for questions about:
        - Return/refund policies
        - Shipping information
        - Product details
        - Company policies
        - General FAQs
        
        Args:
            query: The customer's question or topic
        """
        state: MyState = ctx.session.userdata
        
        logger.info(f"🔍 Searching Pinecone knowledge base: {query}")
        
        # Log conversation
        state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "knowledge_search",
            "query": query
        })
        
        # Search Pinecone
        results = await search_static_knowledge(query, top_k=3)
        
        if not results:
            return "I don't have specific information about that. Let me connect you with a specialist who can help."
        
        # Format results
        response_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            category = result.get('category', 'General')
            response_parts.append(f"[{category}] {text}")
        
        formatted_response = "\n\n".join(response_parts)
        logger.info(f"✓ Found {len(results)} relevant results from knowledge base")
        
        return formatted_response

    @function_tool
    async def get_order_info(
        self, 
        ctx: RunContext, 
        order_number: str | None = None, 
        phone: str | None = None
    ) -> str:
        """
        Fetch complete order information from Firebase database.
        Call this when user provides order number or phone number.
        You can provide either order_number, phone, or both.
        
        Args:
            order_number: The customer's order number (optional)
            phone: The customer's phone number (optional)
        """
        state: MyState = ctx.session.userdata
        
        # Normalize phone
        if phone:
            phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
            if phone.startswith("+91"):
                phone = phone[3:]
            elif phone.startswith("91") and len(phone) == 12:
                phone = phone[2:]
            state.customer_phone = phone
        
        if order_number:
            state.customer_order_number = order_number
        
        # Log conversation
        state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "order_lookup",
            "order_number": order_number,
            "phone": phone
        })
        
        # Search Firebase database
        logger.info(f"🔍 Searching Firebase - Order: {order_number}, Phone: {phone}")
        order_data = await get_order_from_firebase(order_number=order_number, phone=phone)
        
        if not order_data:
            return "Order not found in our system. Please check your order number or phone number and try again."
        
        # Store in state
        state.order_data = order_data
        
        # Log successful retrieval
        state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "order_found",
            "order_number": order_data.get("orderNumber")
        })
        
        # Return formatted order info for the AI to speak naturally
        customer = order_data.get("customer", {})
        items = order_data.get("items", [])
        delivery = order_data.get("delivery", {})
        
        response = f"""Order found for {customer.get('name', 'customer')}:
        
Order Number: {order_data.get('orderNumber')}
Status: {order_data.get('status', 'Unknown')}
Order Date: {order_data.get('orderDate', 'Unknown')}

Items ordered:
"""
        for item in items:
            response += f"- {item.get('name')} (Qty: {item.get('quantity')}, Price: ₹{item.get('price')})\n"
        
        response += f"\nTotal Amount: ₹{order_data.get('totalAmount')}"
        response += f"\nPayment: {order_data.get('paymentMethod')} - {order_data.get('paymentStatus')}"
        
        if delivery:
            response += f"\n\nDelivery Status: {delivery.get('status', 'Unknown')}"
            response += f"\nExpected Delivery: {delivery.get('expectedDate', 'Unknown')}"
            if delivery.get('trackingNumber'):
                response += f"\nTracking: {delivery.get('trackingNumber')}"
            if delivery.get('address'):
                response += f"\nDelivery Address: {delivery.get('address')}"
        
        return response

    @function_tool
    async def transfer_to_human(self, ctx: RunContext, reason: str = "Customer request") -> str:
        """
        Transfer call to human agent via browser (web-based transfer)
        This creates a transfer request that appears in the agent dashboard
        """
        state: MyState = ctx.session.userdata
        
        if state.transfer_initiated:
            return "Transfer already in progress."
        
        state.transfer_initiated = True
        
        # Log conversation
        state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "transfer_initiated",
            "reason": reason
        })
        
        logger.info(f"🔄 Creating browser-based transfer | Reason: {reason}")
        
        try:
            job_ctx = get_job_context()
            room_name = job_ctx.room.name
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{BACKEND_API_URL}/api/create-transfer",
                    params={"room_name": room_name, "reason": reason}
                ) as response:
                    data = await response.json()
                    if data.get("success"):
                        transfer_id = data['transfer']['id']
                        logger.info(f"✅ Browser transfer created: {transfer_id}")
                        
                        state.should_disconnect = True
                        
                        return "I'm transferring you to our support specialist now. Please hold for just a moment while they join the call..."
                    else:
                        raise Exception("Failed to create transfer")
                        
        except Exception as e:
            logger.error(f"Browser transfer failed: {e}")
            state.transfer_initiated = False
            return "I apologize for the trouble. Let me try to help you directly instead."

    @function_tool
    async def end_call(self, ctx: RunContext) -> str:
        """
        End the call gracefully.
        Call this when:
        - Customer says goodbye/bye/thanks
        - Conversation is complete
        """
        state: MyState = ctx.session.userdata
        
        logger.info("📞 Ending call")
        
        # Log conversation
        state.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "call_ended"
        })
        
        # Prepare conversation data
        conversation_data = {
            "session_id": state.session_id,
            "customer_phone": state.customer_phone,
            "order_number": state.customer_order_number,
            "call_duration_seconds": (datetime.now() - state.call_start_time).total_seconds(),
            "conversation_log": state.conversation_log,
            "transferred_to_human": state.transfer_initiated,
            "order_data": state.order_data,
            "call_end_time": datetime.now().isoformat()
        }
        
        # Save to local JSON file (primary storage)
        json_path = save_conversation_to_json(state.session_id, conversation_data)
        logger.info(f"✅ Conversation saved locally: {json_path}")
        
        # Optionally save to Firebase (if available)
        if db is not None:
            await save_conversation_to_firebase(state.session_id, conversation_data)
        
        goodbye_message = "Thank you for contacting ShopEase Support. Have a great day!"
        
        await asyncio.sleep(1)
        await hangup_call()
        
        return goodbye_message

# ============================================
# AI AGENT ENTRYPOINT
# ============================================
async def entrypoint(ctx: JobContext):
    session_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{ctx.room.name}"
    room_name = ctx.room.name
    
    await asyncio.sleep(0.5)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 NEW CALL - GEMINI REALTIME")
    logger.info(f"   Room: {room_name}")
    logger.info(f"   Session: {session_id}")
    logger.info(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    logger.info(f"{'='*60}\n")
    
    assistant = Assistant(room_name)
    state = MyState(session_id)
    
    # Store in active sessions
    active_sessions[room_name] = state
    
    session = AgentSession(
        llm=google_livekit.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            temperature=0.7,
        ),
        userdata=state
    )

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    logger.info(f"✓ Session Started with Gemini Realtime: {session_id}")
    logger.info("🎤 AI is now listening and will greet automatically...")
    
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        if participant.identity.startswith("agent_"):
            logger.info(f"👤 Human agent joined via browser: {participant.identity}")
            logger.info(f"🚪 AI Agent disconnecting to allow human conversation...")
            
            state.should_disconnect = True
            asyncio.create_task(disconnect_ai_agent(ctx, session, state))
    
    try:
        while not state.should_disconnect:
            await asyncio.sleep(1)
        
        logger.info(f"✅ AI Agent successfully disconnected from {room_name}")
        
    except Exception as e:
        logger.error(f"Error in AI agent loop: {e}")
    finally:
        # Save conversation before cleanup
        if not state.transfer_initiated:
            conversation_data = {
                "session_id": state.session_id,
                "customer_phone": state.customer_phone,
                "order_number": state.customer_order_number,
                "call_duration_seconds": (datetime.now() - state.call_start_time).total_seconds(),
                "conversation_log": state.conversation_log,
                "transferred_to_human": state.transfer_initiated,
                "order_data": state.order_data,
                "call_end_time": datetime.now().isoformat()
            }
            save_conversation_to_json(state.session_id, conversation_data)
        
        if room_name in active_sessions:
            del active_sessions[room_name]
        logger.info("✓ Session ended")

async def disconnect_ai_agent(ctx: JobContext, session: AgentSession, state: MyState):
    """Gracefully disconnect AI agent when human takes over"""
    try:
        logger.info("🔌 Disconnecting AI Agent...")
        
        # Save conversation before disconnecting
        conversation_data = {
            "session_id": state.session_id,
            "customer_phone": state.customer_phone,
            "order_number": state.customer_order_number,
            "call_duration_seconds": (datetime.now() - state.call_start_time).total_seconds(),
            "conversation_log": state.conversation_log,
            "transferred_to_human": state.transfer_initiated,
            "order_data": state.order_data,
            "transfer_time": datetime.now().isoformat()
        }
        save_conversation_to_json(state.session_id, conversation_data)
        
        await session.aclose()
        await ctx.room.disconnect()
        
        logger.info("✅ AI Agent disconnected - Human agent now active")
        
    except Exception as e:
        logger.error(f"Error disconnecting AI: {e}")

# ============================================
# STARTUP FUNCTIONS
# ============================================
def start_backend_server():
    """Start FastAPI backend server"""
    logger.info("\n🚀 Starting Backend Server...")
    logger.info(f"   URL: http://localhost:8000")
    logger.info(f"   WebSocket: ws://localhost:8000/ws/agent")
    logger.info(f"   Conversations API: http://localhost:8000/api/conversations")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

def start_ai_agent():
    """Start LiveKit AI agent"""
    logger.info("\n🤖 Starting AI Agent with Gemini Realtime...")
    logger.info(f"   LiveKit URL: {LIVEKIT_URL}")
    logger.info(f"   Model: Gemini 2.0 Flash (Realtime)")
    logger.info(f"   Voice: Puck")
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

# ============================================
# MAIN ENTRY POINT
# ============================================
def proctored_agent(ctx: JobContext):
    """Main entrypoint for LiveKit agent"""
    return asyncio.create_task(entrypoint(ctx))

if __name__ == "__main__":
    import sys
    
    # Check if running in LiveKit cloud (no FastAPI server needed)
    is_livekit_deployment = os.getenv("LIVEKIT_DEPLOYMENT", "false").lower() == "true"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "download-files":
            # This is called by Dockerfile CMD - just download files and exit
            logger.info("📥 Downloading model files...")
            # LiveKit agents framework handles this automatically
            sys.exit(0)
            
        elif command == "start":
            # This is the actual agent start in deployment
            logger.info("\n" + "="*60)
            logger.info("🤖 AI AGENT - LIVEKIT CLOUD DEPLOYMENT")
            logger.info("="*60)
            logger.info(f"   Model: Gemini 2.0 Flash Realtime")
            logger.info(f"   Voice: Puck")
            logger.info(f"   LiveKit URL: {LIVEKIT_URL}")
            logger.info("="*60 + "\n")
            
            # Start ONLY the AI agent (no FastAPI server)
            cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
            sys.exit(0)
    
    # Local development mode (runs both FastAPI + Agent)
    logger.info("\n" + "="*60)
    logger.info("🏢 AI CALL CENTER - LOCAL DEVELOPMENT MODE")
    logger.info("="*60)
    logger.info(f"   Backend: FastAPI + WebSocket")
    logger.info(f"   AI Agent: Gemini 2.0 Flash Realtime")
    logger.info(f"   Static Data: Pinecone Vector DB")
    logger.info(f"   Dynamic Data: Firebase Firestore")
    logger.info(f"   Storage: Local JSON Files (./conversations/)")
    logger.info(f"   Transfer: Browser-based (Web Dashboard)")
    logger.info("="*60 + "\n")
    
    # Start backend server in separate thread
    backend_thread = Thread(target=start_backend_server, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    time.sleep(2)
    
    # Start AI agent in main thread
    start_ai_agent()