import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def find_occurrences(l, pattern):
    indexes = []
    initial_indexes = [index for index, x in enumerate(l) if x == pattern[0]]    
    for initial_index in initial_indexes:
        if l[initial_index:initial_index+len(pattern)] == pattern:
            indexes.append(initial_index)
    return indexes    