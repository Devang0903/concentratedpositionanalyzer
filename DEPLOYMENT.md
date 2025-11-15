# Deployment Guide

This guide will help you deploy the Concentrated Position Analyzer Streamlit app.

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Prerequisites
1. GitHub account (you have: Devang0903)
2. Repository pushed to GitHub: https://github.com/Devang0903/concentratedpositionanalyzer

### Steps

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Concentrated Position Analyzer"
   git branch -M main
   git remote add origin https://github.com/Devang0903/concentratedpositionanalyzer.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `Devang0903/concentratedpositionanalyzer`
   - Set Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live at:**
   `https://[your-app-name].streamlit.app`

## Option 2: Heroku

1. **Create a Procfile:**
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create setup.sh:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Option 3: Docker

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t position-analyzer .
   docker run -p 8501:8501 position-analyzer
   ```

## Files Needed for Deployment

- ✅ `streamlit_app.py` - Main app file
- ✅ `requirements.txt` - Dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Git ignore rules

## Notes

- The app uses yfinance which fetches data from Yahoo Finance
- No API keys required
- Cache directory will be created automatically on first run
- The app is stateless - no database needed

