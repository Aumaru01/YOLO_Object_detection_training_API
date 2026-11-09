# 0. open the label-studio web interface in the default browser
label-studio

# 1A. run the API server in terminal
./venv/bin/python -m api_main
# 1B. run the API server in background and redirect logs to a file
nohup ./venv/bin/python -m api_main > logs/api.log 2>&1 &