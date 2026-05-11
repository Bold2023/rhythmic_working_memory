
def list_of_vals(example):
    return (lambda arg: list(map(type(example), arg.split(','))))

