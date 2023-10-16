NOTEBOOK='../key/Intro2NN.ipynb'
NOTESMD='../README.md'
NOTESPDF='../Intro2NN.pdf'
SOLUTIONSMD='../key/Intro2NN_key.md'
SOLUTIONSPDF='../key/Intro2NN_key.pdf'

# To Notes Markdown
jupyter nbconvert ${NOTEBOOK} --TagRemovePreprocessor.remove_cell_tags "solution-only" --to markdown --output=${NOTESMD}

# To Solution Markdown
jupyter nbconvert ${NOTEBOOK} --TagRemovePreprocessor.remove_cell_tags "notes-only" --to markdown --output=${SOLUTIONSMD}

# To Notes ipynb and Solution ipynb
python ./toNotebooks.py

