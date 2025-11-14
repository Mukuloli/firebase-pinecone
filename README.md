# AI Call Center - WebRTC Application

A WebRTC-based AI call center application using LiveKit, Gemini Realtime, and FastAPI.

## Project Structure

```
.
├── src/
│   ├── api/           # FastAPI backend routes
│   ├── agents/        # AI agent implementation
│   ├── models/        # Data models and schemas
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── data/              # Data files (orders, company info)
├── instructions/      # Agent instructions
├── venv/              # Virtual environment (created after setup)
├── main.py           # Main entry point
├── requirements.txt  # Python dependencies
└── .env.example     # Environment variables template
```

## Setup Instructions

### 1. Create Virtual Environment

The virtual environment has already been created. If you need to recreate it:

**Windows (Command Prompt):**
```bash
python -m venv venv
```

**Windows (PowerShell):**
```powershell
python -m venv venv
```

**Or use the setup script:**
```bash
setup.bat
```

### 2. Activate Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Or use the activation script:**
```bash
activate.bat
```

### 3. Install Dependencies

Once the virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env
```

Edit `.env` and add your:
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `BACKEND_API_URL`

### 5. Firebase Credentials (Optional)

If you want to use Firebase, place your `credentials.json` file in the root directory.

### 6. Run the Application

```bash
python main.py dev
```

## Features

- AI-powered voice customer support using Gemini Realtime
- Order lookup by order number or phone number
- Human agent transfer via browser-based dashboard
- Real-time WebSocket notifications
- Firebase integration (commented code available)

## Notes

- All commented code is preserved exactly as it was in the original file
- To use Firebase search, uncomment the relevant code in `src/utils/firebase.py` and `src/agents/assistant.py`
- To use SIP transfer, uncomment the relevant code in `src/agents/assistant.py`
- Instructions are loaded from `instructions/agent_instructions.yml` (YAML format for faster loading)

## Deactivate Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```


