#!/bin/bash
# Setup script for GitHub deployment

echo "🚀 Setting up Git repository for deployment..."

# Initialize git if not already
if [ ! -d ".git" ]; then
    git init
    echo "✓ Git repository initialized"
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Concentrated Position Analyzer with Streamlit app"

# Set main branch
git branch -M main

# Add remote (if not already added)
if ! git remote | grep -q "origin"; then
    git remote add origin https://github.com/Devang0903/concentratedpositionanalyzer.git
    echo "✓ Remote added"
fi

echo ""
echo "✅ Ready to push! Run:"
echo "   git push -u origin main"
echo ""
echo "Then deploy on Streamlit Cloud:"
echo "   1. Go to https://share.streamlit.io"
echo "   2. Sign in with GitHub"
echo "   3. Click 'New app'"
echo "   4. Select repository: Devang0903/concentratedpositionanalyzer"
echo "   5. Main file: streamlit_app.py"
echo "   6. Click 'Deploy!'"

