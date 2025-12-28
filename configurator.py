import sys 
from ast import literal_eval
# literal_eval for evaluating the string into python objects

# sys.argv for python[ignored] script.py[0] --flag[1]
for arg in sys.argv[1:]:
    if '=' not in arg: 
        # this is a file path  
        assert not arg.startswith('--'), (f'Expected a config file path, got {arg!r}')
        config_file = arg
        print(f'Overriding config with {config_file}:')
        with open(config_file) as f:
            code = f.read()
            print(code)
        exec(code, globals()) # runs everything inside the file 
    # this file is a config file with all hyperparameters only
    else:
        # assume it's a --key=value argument 
        assert arg.startswith('--'), (f'Expected --key=value argument, got {arg!r}')
        key, val = arg.split('=', 1) # split only once
        key = key[2:].strip() 
        val = val.strip() # remove the spaces
        if key in globals():
            try:
                # use the bool, int, float etc
                # --batch_size=32, val=32
                attempt = literal_eval(val) 
            except (ValueError, SyntaxError):
                # use the string
                attempt = val
            # key=batch_size, globals()[key] = value, attempt = value
            assert type(attempt) == type(globals()[key])
            print(f'Overriding: {key} = {attempt}')
            globals()[key] = attempt 
        else:
            raise ValueError(f"Unknown config key: {key!r}")




    
