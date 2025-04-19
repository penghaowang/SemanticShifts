#!/bin/bash

# Function to parse simple key: value lines from YAML and export them as uppercase environment variables.
# Usage: load_yaml_config <yaml_file_path>

load_yaml_config() {
    local yaml_file="$1"

    if [ ! -f "$yaml_file" ]; then
        echo "Error: YAML config file not found at $yaml_file" >&2
        return 1
    fi

    # Use grep to find lines with a colon (key: value)
    # Use sed to:
    # 1. Remove comments (#...)
    # 2. Remove leading/trailing whitespace
    # 3. Extract key (part before first ':')
    # 4. Extract value (part after first ':')
    # 5. Remove leading/trailing whitespace from key and value
    # 6. Remove optional quotes (' or ") surrounding the value
    # Use awk for final processing and export
    
    # Define function within awk to trim whitespace
    local awk_script='
    function trim(s) {
        sub(/^[ \t\r\n]+/, "", s);
        sub(/[ \t\r\n]+$/, "", s);
        return s;
    }
    {
        # Remove comments
        sub(/#.*/, "");
        # Basic key-value split on first colon
        if (match($0, /:[ \t]*/)) {
            key = substr($0, 1, RSTART - 1);
            value = substr($0, RSTART + RLENGTH);
            
            key = trim(key);
            value = trim(value);
            
            # Remove quotes from value
            gsub(/^["\']|["\']$/, "", value);
            
            # Convert key to uppercase ENV_VAR style
            gsub(/[^a-zA-Z0-9_]/, "_", key);
            key = toupper(key);
            
            # Print export command
            if (key != "") {
                # Escape potential special characters in value for export
                gsub(/\$/, "\\$", value); 
                gsub(/`/, "\\`", value); 
                gsub(/"/, "\\\"", value); 
                printf "export %s=\"%s\"\n", key, value;
            }
        }
    }'
    
    # Execute the awk script and evaluate the output to perform exports
    eval $(awk "$awk_script" "$yaml_file")

    # Optional: Check if essential variables were loaded
    # if [ -z "$BASE_HS_DIR" ]; then
    #     echo "Warning: BASE_HS_DIR not loaded from config." >&2
    # fi

    return 0
} 