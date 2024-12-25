# utility module

# Determine which divice to use.
def determine_device():
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"

    if torch.has_mps:
        device = "mps"
    
    print(f"{device} is found.") 

    return device

device = determine_device()