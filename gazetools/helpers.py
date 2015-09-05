def loadProgram(filename):
    with open(filename, 'r') as f:
        return "".join(f.readlines())
