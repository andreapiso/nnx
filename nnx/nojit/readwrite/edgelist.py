import nnx

def parse_edgelist(lines, comments='#', delimiter=None,
                   create_using=None):
    """Parse lines of an edge list representation of a graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in edgelist format
    comments : string, optional
       Marker for comment lines
    delimiter : string, optional
       Separator for node labels
    create_using: NetworkX graph container, optional

    """

    from ast import literal_eval
    if create_using is None:
        G=nnx.SimpleGraphArray()
    else:
        G=create_using()

    for line in lines:
        p=line.find(comments)
        if p>=0:
            line = line[:p]
        if not len(line):
            continue
        # split line, should have 2 or more
        s=line.strip().split(delimiter)
        if len(s)<2:
            continue
        u=int(s.pop(0))
        v=int(s.pop(0))
        d=s #TODO edge weights
        G.add_edge(u, v, True)
    return G

def read_edgelist(path, comments="#", delimiter=None, create_using=None, encoding='utf-8'):
    """Read a graph from a list of edges.

    Parameters
    ----------
    path : file or string
       File or filename to write. If a file is provided, it must be
       opened in 'rb' mode.
       Filenames ending in .gz or .bz2 will be uncompressed.
    comments : string, optional
       The character used to indicate the start of a comment. 
    delimiter : string, optional
       The string used to separate values.  The default is whitespace.
    create_using : Graph container, optional, 
       Use specified container to build graph.  The default is networkx.Graph,
       an undirected graph.
    nodetype : int, float, str, Python type, optional
       Convert node data from strings to specified type
    data : bool or list of (label,type) tuples
       Tuples specifying dictionary key names and types for edge data
    edgetype : int, float, str, Python type, optional OBSOLETE
       Convert edge data from strings to specified type and use as 'weight'
    encoding: string, optional
       Specify which encoding to use when reading file.

    
    lines = (line.decode(encoding) for line in path)
    """
    with open(path, 'rb') as f:
        lines = (line.decode(encoding) for line in f)
        return parse_edgelist(lines,comments=comments, delimiter=delimiter,
                            create_using=create_using)