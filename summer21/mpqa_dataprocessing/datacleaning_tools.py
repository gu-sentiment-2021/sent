import os
import clipboard as cb

def adjustSpansInRange(start_byte=0, end_byte=1e9, delta=0, filename="gateman.mpqa.lre.3.0"):
    """
    It adjusts the spans of the annotation lines in the 'filename' that are
    between 'start_byte' and 'end_byte', and adds 'delta' to them. It stores
    the modified file in the clipboard and prints the modified line ids.
    """
    modified_file = '' # Store the entire modified file
    ids = [] # Store the modified annotation line ids (to use them in the new README file)
    with open(os.path.join(filename)) as doc_file:
        doc_text = doc_file.readlines()
        for line in doc_text:
            if line[0] == '#': # Skip comment lines
                modified_file += line
                continue
            id, span, anno_type, attr = line.split('\t')
            x, y = span.split(',')
            x, y = int(x), int(y)
            modified = False # Remember if we've modified this annotation line
            if x >= start_byte and x <= end_byte:
                x += delta
                modified = True
            if y >= start_byte and y <= end_byte:
                y += delta
                modified = True
            modified_file += '{}\t{},{}\t{}\t{}'.format(id, x, y, anno_type, attr)
            if modified: # Store its id, if we've modified this annotation line
                ids.append(int(id))
    cb.copy(modified_file) # Copy the complete modified file into the clipboard
    print("len(ids):", len(ids))
    print("ids:", ids)

# Sample:
# adjustSpansInRange(start_byte=2318, delta=+2)