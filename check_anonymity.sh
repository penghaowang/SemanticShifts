#!/bin/bash
# Script to check for any remaining personal information in the repo

echo "Checking for personal username (phwang)..."
grep -r "phwang" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Checking for organization reference (cscs)..."
grep -r "cscs" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Checking for personal paths..."
grep -r "/users/" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .
grep -r "/iopsstor/" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .
grep -r "/capstor/" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Checking for Hugging Face token..."
grep -r "hf_" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Checking for personal account ID..."
grep -r "a-a05" --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Checking for email addresses..."
grep -r -E '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" .

echo "Anonymity check completed" 