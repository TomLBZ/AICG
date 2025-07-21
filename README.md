# AICG
AI Generic Code Generator

This is a simple library / script that generates code based on structured instructions in yaml or json format, or based on specifications programmatically provided to it.

### Usage as a library

See [test_lib.py](test_lib.py) for usage as a library.

### Usage as a script
Running the script directly depends on a correct OpenAI-Compatible API endpoint. Please open the [agcg.py](agcg.py) file and set the `BASE_URL` variable to point to your OpenAI-compatible API endpoint. For example:

```
BASE_URL = "https://ai.my.server.com/engines/v1"
MODEL = "ai/phi4"
KEY = "IGNORED"
```

Then, run the script with the following command:

```bash
python3 agcg.py
```

The script listens for input on stdin and outputs the generated code to stdout. To mark the end of input, send the `0x04` character (EOT, End Of Transmission). To exit the script, press `Ctrl + C`.

### Run Your Own Local LLM with Docker
You can use the `docker model` commands to run a local LLM server. For example, to run a local instance of the Phi-4 model, you can use:

```bash
docker model pull ai/phi4:latest
docker model run ai/phi4:latest
```

Note that you must enable `docker model` first by following the instructions at [Docker Model Runner](https://docs.docker.com/ai/model-runner/).

After running the model, you can use the [test_api.sh](test_api.sh) script to test the API endpoint. Make sure to edit the `curl` command to point to your local model server url first. Then, run the script with:

```bash
./test_api.sh
```