# AICG
AI Generic Code Generator

This is a simple library / script that generates code based on structured instructions in yaml or json format.

### Usage as a library

See [test_lib.py](test_lib.py) for usage as a library.

### Usage as a script
Run the script with the following command:

```bash
python3 agcg.py
```

The script listens for input on stdin and outputs the generated code to stdout. To mark the end of input, send the `0x04` character (EOT, End Of Transmission). To exit the script, press `Ctrl + C`.