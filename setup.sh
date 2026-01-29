#!/bin/bash
mkdir -p ~/.streamlit/
echo "[theme]
primaryColor = \"#FF6B35\"
backgroundColor = \"#FFFFFF\"
secondaryBackgroundColor = \"#F0F0F0\"
textColor = \"#000000\"
font = \"sans serif\"

[client]
showErrorDetails = false
" > ~/.streamlit/config.toml
