from sklearn.tree import export_text

def get_rules(tree, tree_id: int , features: list, sas_table: str, max_depth=100, spacing=2):
    """ 
    Extract the rules of a decision tree and translate them to SAS code.
    Create a SAS dataset representing those rules.
    
    
    Parameters:
    -----------
    tree: sklearn DecisionTreeClassifier
        the tree whose decision rules we want to extract
    tree_id: int
        tree identifier (0 to numbers of trees -1)
    features: list
        list of model features
    sas_table: str
        name of the SAS dataset containing the decision tree features
    max_depth: int
        number of levels of the tree considered (default is 100 - must be greater than 1)
        more information : https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
    spacing: int
        number of spaces between edges (default is 2)
        more information : https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html      
    """
    if spacing < 2:
        raise ValueError('spacing must be > 1')
    # export decision tree to text, using sklearn.tree function
    rules = export_text(tree, feature_names=features, 
                        max_depth=max_depth,
                        decimals=6,
                        spacing=spacing-1)
    # translate text to SAS code
    rules_in_sas = translate_text_to_sas(tree, tree_id, sas_table, features, rules, spacing)
    return rules_in_sas

def translate_text_to_sas(tree, tree_id: int, sas_table: str, features: list, text: str, spacing=2):
    """ 
    Translate tree rules to SAS code, into a dataset.
    
    
    Parameters:
    -----------
    tree: sklearn DecisionTreeClassifier
        the tree whose decision rules we want to extract
    tree_id: int
        tree identifier (0 to numbers of trees -1)
    features: list
        list of model features
    sas_table: str
        name of the SAS dataset containing the decision tree features
        a column "PREDICTED_VALUE_i" will be added to this dataset (i matches tree identifier)
    text: str
        rules obtained with export_text function
    spacing: int
        number of spaces between edges (default is 2)
        more information : https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html      
    """
    skip, dash = ' '*spacing, '-'*(spacing-1) # handling spacing
    sas_rules = 'DATA DECISION_TREE_' + str(tree_id + 1) + ';\n'
    sas_rules += 'SET {};\n'.format(sas_table)
    splitted_rules = text.split('\n') # Make a list of rules
    dict_space_count = {} # keys: number of eft spaces, values: number of rows with this number
    current_elseif_count = [] # for handling "END;" ASSOCIATED WITH "ELSE IF" conditions. Add as many "END;" as nested "ELSE IFs"
    # Iterate through rules
    for line in splitted_rules:
        line = line.rstrip().replace('|',' ').replace('-', ' ') # replace | and - by spaces
        n_spaces = len(line) - len(line.lstrip(' ')) # get spaces from left
        # Update dictionary for handling whether IF or ELSE IF
        if str(n_spaces) not in dict_space_count.keys():
            dict_space_count[str(n_spaces)] = 1
        else: 
            dict_space_count[str(n_spaces)] += 1
        # If count of spaces is even --> ELSE condition
        if 'class' in line:
            dict_space_count[str(n_spaces)] -= 1 # do not count rows where predicted value is computed
        front_add = get_front_add(dict_space_count[str(n_spaces)], n_spaces)
        add_end_front = ''
        if len(current_elseif_count):
            if n_spaces < current_elseif_count[-1]:
                add_end_front = ''
                while(n_spaces < current_elseif_count[-1]):
                    add_end_front += 'END;\n'
                    current_elseif_count.pop() # last element corresponds to current front spaces in line
                    if not len(current_elseif_count):
                        break
        if 'ELSE' in front_add and 'class' not in line:
            current_elseif_count.append(n_spaces)
        # Handling rows for IF conditions
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace(' '*n_spaces, ' '*n_spaces + front_add)
            line = '{} {:g} THEN DO;'.format(line, float(val))
        # Handling rows for PREDICTED_VALUE_i
        if 'class' in line:
            line = line.replace('class:', 'PREDICTED_VALUE_' + str(tree_id + 1) + ' =')
            line += ';'
        line = add_end_front + line
        sas_rules += skip + line + '\n'
    sas_rules = sas_rules[:-1]
    sas_rules += 'RUN;'
    return sas_rules

# String to add before row for IF or ELSE IF conditions
def get_front_add(count, n_spaces):
    """
    Get string to add before condition, for IF or ELSE IF nested condition
    
    Parameters:
    -----------
    count: counter of the occurences of the current line spacing
    n_spaces: current line spacing    
    """
    if count%2 == 0:
        toRet = 'END;\n'+ ' '*(n_spaces+2)+ 'ELSE IF '
    else:
        toRet = 'IF '
    return toRet

