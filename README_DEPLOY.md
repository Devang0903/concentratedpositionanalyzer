# Quick Deployment Guide

## Deploy to Streamlit Cloud (Easiest - Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/Devang0903/concentratedpositionanalyzer.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `Devang0903/concentratedpositionanalyzer`
   - Main file: `streamlit_app.py`
   - Click "Deploy!"

3. **Done!** Your app will be live in ~2 minutes.

## What Gets Deployed

- ✅ Streamlit web app (`streamlit_app.py`)
- ✅ All dependencies (`requirements.txt`)
- ✅ Configuration (`.streamlit/config.toml`)

## Local Testing

Before deploying, test locally:
```bash
streamlit run streamlit_app.py
```

