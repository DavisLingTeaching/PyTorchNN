import nbformat

def makeSolution():
    nb = nbformat.read('../key/Intro2NN.ipynb', as_version=4)
    notes_index = []
    for i, cell in enumerate(nb['cells']):
        try:
            tags = cell['metadata']['tags']
            if 'notes-only' in tags:
                notes_index.append(i)
        except:
            pass

    for i, index in enumerate(notes_index):
        print(index)
        nb.cells.pop(index-i)

    nbformat.write(nb, "../key/Intro2NN_key.ipynb")

def makeNotes():
    nb = nbformat.read('../key/Intro2NN.ipynb', as_version=4)
    solutions_index = []
    for i, cell in enumerate(nb['cells']):
        try:
            tags = cell['metadata']['tags']
            if 'solution-only' in tags:
                solutions_index.append(i)
        except:
            pass

    for i, index in enumerate(solutions_index):
        print(index)
        nb.cells.pop(index-i)

    nbformat.write(nb, "../Intro2NN.ipynb")

makeSolution()
makeNotes()
