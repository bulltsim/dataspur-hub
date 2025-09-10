#!/usr/bin/env bash
set -euo pipefail

# Helper script to create a GitHub project board for DataSpur Phase 1 using GitHub CLI.
# Requires gh CLI to be installed and authenticated.

PROJECT_TITLE="DataSpur Phase 1"
PROJECT_DESCRIPTION="Tasks and progress tracking for DataSpur motion tracking Phase 1"

# Determine owner from current repository
OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null || echo "")

if [ -z "$OWNER" ]; then
  echo "Unable to determine owner. Pass owner as first argument."
  exit 1
fi

# Create the project
gh project create "$OWNER/$PROJECT_TITLE" \
  --title "$PROJECT_TITLE" \
  --description "$PROJECT_DESCRIPTION" \
  --public || true
