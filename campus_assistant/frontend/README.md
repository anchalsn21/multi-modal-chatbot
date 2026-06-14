# Campus Assistant — Frontend

Next.js 15 frontend for the Greenfield Campus Orientation Assistant.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Next.js 15 (App Router) | React framework |
| TypeScript | Type safety |
| Tailwind CSS v4 | Utility-first styling |
| Lucide React | Icons |
| shadcn/ui (minimal primitives) | UI base |

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Set up environment variables
cp .env.example .env.local
# Edit .env.local: set NEXT_PUBLIC_API_URL=http://localhost:8000

# 3. Start the dev server
npm run dev
# → Opens at http://localhost:3000
```

Make sure the **backend is running** on port 8000 before testing the chat.

## Pages

| Route  | Description |
|--------|-------------|
| `/`    | Landing page — hero, features, how it works |
| `/chat`| Full-screen chat UI — sidebar + message window + input bar |

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout, fonts, metadata
│   │   ├── page.tsx            # Landing page (/)
│   │   └── chat/page.tsx       # Chat page (/chat)
│   ├── components/
│   │   ├── landing/            # Hero, Navbar, Features, HowItWorks, Footer
│   │   └── chat/               # ChatWindow, MessageBubble, InputBar,
│   │                           #   VoiceButton, ImageUpload
│   ├── hooks/
│   │   └── useChat.ts          # All chat state + API calls
│   ├── lib/
│   │   ├── api.ts              # Typed fetch wrappers for backend
│   │   └── utils.ts            # cn(), formatTime(), generateId()
│   └── types/
│       └── chat.ts             # Message, ChatResponse, etc.
├── .env.local                  # NEXT_PUBLIC_API_URL (not committed)
├── .env.example
└── package.json
```

## Chat Features

- **Text input** — auto-resizing textarea, Enter to send
- **Voice input** — MediaRecorder API, sends audio/webm to `/chat/voice`
- **Image upload** — file picker with thumbnail preview, sends to `/chat/image`
- **Suggested prompts** — sidebar buttons to try one query per intent
- **Clear conversation** — reset to welcome message
- **Typing indicator** — three-dot animation while loading
- **Intent badges** — assistant messages show the predicted intent label

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | FastAPI backend URL | `http://localhost:8000` |
