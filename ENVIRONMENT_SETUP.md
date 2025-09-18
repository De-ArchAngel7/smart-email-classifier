# Environment Setup Guide

## Required Environment Variables

### Backend (.env or environment variables)
```bash
HF_API_TOKEN=your_hugging_face_token_here
PORT=8000
HOST=0.0.0.0
```

### Frontend (frontend/.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Getting Your Hugging Face Token

1. Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token and set it in your environment variables

## Deployment

### Backend (Railway/Render)
Set these environment variables in your deployment platform:
- `HF_API_TOKEN`: Your Hugging Face token
- `PORT`: (automatically set by platform)

### Frontend (Vercel/Netlify)
Set these environment variables:
- `NEXT_PUBLIC_API_URL`: Your deployed backend URL

## Local Development

1. Create a `.env` file in the root directory
2. Add your environment variables
3. Run the application

The application will automatically load environment variables and provide helpful error messages if any are missing.
