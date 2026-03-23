#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# push_to_github.sh
# Run this ONCE in your Mac Terminal to publish your CM3070 repo to GitHub.
# Prerequisites: git installed (comes with macOS Xcode Command Line Tools)
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  CM3070 — Push to GitHub"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: collect credentials ──────────────────────────────────────────────
read -p "  Your GitHub username: " GH_USER
echo ""
echo "  You need a Personal Access Token (PAT) with 'repo' scope."
echo "  Create one at: https://github.com/settings/tokens/new"
echo "  → Select 'repo' checkbox → click 'Generate token' → copy it"
echo ""
read -s -p "  Paste your PAT here (input hidden): " GH_TOKEN
echo ""

REPO_NAME="cm3070-ai-study-assistant"
REPO_URL="https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git"

# ── Step 2: create the public repo via GitHub API ────────────────────────────
echo ""
echo "  Creating public repo '${REPO_NAME}' on GitHub..."
HTTP_CODE=$(curl -s -o /tmp/gh_create_response.json -w "%{http_code}" \
  -X POST "https://api.github.com/user/repos" \
  -H "Authorization: token ${GH_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  -d "{\"name\":\"${REPO_NAME}\",\"description\":\"CM3070 Final Project — AI Study Assistant V4: lecture audio → transcription, topic classification, keywords, hierarchical summarisation\",\"private\":false,\"auto_init\":false}")

if [ "$HTTP_CODE" = "201" ]; then
    echo "  ✓ Repo created: https://github.com/${GH_USER}/${REPO_NAME}"
elif [ "$HTTP_CODE" = "422" ]; then
    echo "  ⚠ Repo already exists — will push to existing repo."
else
    echo "  ✗ Failed to create repo (HTTP $HTTP_CODE). Response:"
    cat /tmp/gh_create_response.json
    exit 1
fi

# ── Step 3: push ─────────────────────────────────────────────────────────────
echo ""
echo "  Pushing code..."
cd "$(dirname "$0")"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ SUCCESS!"
echo "  Your public repo URL:"
echo "  https://github.com/${GH_USER}/${REPO_NAME}"
echo ""
echo "  Copy this URL into your CM3070 submission."
echo "═══════════════════════════════════════════════════════════"
echo ""
