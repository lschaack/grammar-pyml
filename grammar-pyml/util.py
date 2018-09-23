import os

# Draws a box around a nearly-arbitrary input text with optional title
#  currently text with no spaces is unsupported
def box(text, title=None, alignment='center', width=80):
    """
    <nlc>   'new line char'
    <arc>   'alignment replacement char'
    <wid>   'width'
    """
    stride_len = width-4
    bottom_line = '+{}+<nlc>'.format('-' * (width - 2))
    top_line = bottom_line if title is None else box(title, None, 'center', width)
    mid_lines = []

    alignment_char = get_alignment_char(alignment)

    line_text = ''
    curr_len = 0
    for index in range(len(text)): # loop through string char by char
        # break on newline or string reaching max width of line
        if curr_len < stride_len and text[index] != '\n':
            line_text += text[index] # might be slow
            curr_len += 1
        else:
            # implicitly handle special chars
            line_text = line_text.strip().replace('\t', '    ')
            line_text += text[index]
            line_text, leftover = split_last_full_word(line_text)
            mid_lines.append(line_from_text(line_text, alignment_char, stride_len))
            # only add current char if not a newline
            line_text = leftover
            curr_len = len(line_text)

    if line_text: # flush
        mid_lines.append(line_from_text(line_text, alignment_char, stride_len))

    full_box = top_line + ''.join(mid_lines) + bottom_line
    return full_box.replace('<nlc>', os.linesep)

# Gets the proper character from the alignment param, throws ValueError if not found
def get_alignment_char(alignment):
    if alignment == 'center':
        return '^'
    elif alignment == 'left':
        return '<'
    elif alignment == 'right':
        return '>'
    else:
        raise ValueError('Alignment should be one of "left", "right", or "center"')

# Handles the application of settings like alignment and line width
def line_from_text(line_text, alignment_char, stride_len):
    mid_line = '| {:<arc><wid>} |<nlc>'
    mid_line = mid_line.replace('<arc>', alignment_char) # ensure proper alignment
    mid_line = mid_line.replace('<wid>', str(stride_len)) # ensure proper width
    return mid_line.format(line_text)

# Splits the line on the last full word if necessary
def split_last_full_word(line_text):
    if line_text[-1] != '\n':
        # try/except quicker and cheaper than conditional since lines w/o ' ' are so rare
        try:
            final_space_index = line_text.rindex(' ')
            return line_text[:final_space_index], line_text[final_space_index:]
        except ValueError:
            return line_text, ''
    else:
        return line_text[:-1], '' # don't return the newline itself

# Only do tests if called from terminal rather than imported
if __name__ == '__main__':
    print(box("""CLEOPATRA:
No, the prince they are as many boy.

CRESSIDA:
A' had been a prince what I should queen. By my troth,
He would be young John Falstaff for them you;
Then tell me I offend you who will follow thy chamber.

First Hurder:
And I have praised to your good lord, from this
From her at April are, ha! must not come?""",
    "Excerpt from Shakespeare, left aligned",
    alignment="left"))

    print(box("""CLEOPATRA:
No, the prince they are as many boy.

CRESSIDA:
A' had been a prince what I should queen. By my troth,
He would be young John Falstaff for them you;
Then tell me I offend you who will follow thy chamber.

First Hurder:
And I have praised to your good lord, from this
From her at April are, ha! must not come?""",
    "Excerpt from Shakespeare, center aligned",
    alignment="center"))

    print(box("""CLEOPATRA:
No, the prince they are as many boy.

CRESSIDA:
A' had been a prince what I should queen. By my troth,
He would be young John Falstaff for them you;
Then tell me I offend you who will follow thy chamber.

First Hurder:
And I have praised to your good lord, from this
From her at April are, ha! must not come?""",
    "Excerpt from Shakespeare, right aligned",
    alignment="right"))

    print(box("""It was in one of the vast and gloomy chambers of this remaining tower that I, Antoine, last of the unhappy and accursed Counts de C-, first saw the light of day, ninety long years ago. Within these walls and amongst the dark and shadowy forests, the wild ravines and grottos of the hillside below, were spent the first years of my troubled life. My parents I never knew. My father had been killed at the age of thirty-two, a month before I was born, by the fall of a stone somehow dislodged from one of the deserted parapets of the castle. And my mother having died at my birth, my care and education devolved solely upon one remaining servitor, an old and trusted man of considerable intelligence, whose name I remember as Pierre. I was an only child and the lack of companionship which this fact entailed upon me was augmented by the strange care exercised by my aged guardian, in excluding me from the society of the peasant children whose abodes were scattered here and there upon the plains that surround the base of the hill. At that time, Pierre said that this restriction was imposed upon me because my noble birth placed me above association with such plebeian company. Now I know that its real object was to keep from my ears the idle tales of the dread curse upon our line that were nightly told and magnified by the simple tenantry as they conversed in hushed accents in the glow of their cottage hearths.

    Thus isolated, and thrown upon my own resources, I spent the hours of my childhood in poring over the ancient tomes that filled the shadow haunted library of the chateau, and in roaming without aim or purpose through the perpetual dust of the spectral wood that clothes the side of the hill near its foot. It was perhaps an effect of such surroundings that my mind early acquired a shade of melancholy. Those studies and pursuits which partake of the dark and occult in nature most strongly claimed my attention.""", 
    "Excerpt from Lovecraft, left aligned",
    alignment="left"))

    print(box("""It was in one of the vast and gloomy chambers of this remaining tower that I, Antoine, last of the unhappy and accursed Counts de C-, first saw the light of day, ninety long years ago. Within these walls and amongst the dark and shadowy forests, the wild ravines and grottos of the hillside below, were spent the first years of my troubled life. My parents I never knew. My father had been killed at the age of thirty-two, a month before I was born, by the fall of a stone somehow dislodged from one of the deserted parapets of the castle. And my mother having died at my birth, my care and education devolved solely upon one remaining servitor, an old and trusted man of considerable intelligence, whose name I remember as Pierre. I was an only child and the lack of companionship which this fact entailed upon me was augmented by the strange care exercised by my aged guardian, in excluding me from the society of the peasant children whose abodes were scattered here and there upon the plains that surround the base of the hill. At that time, Pierre said that this restriction was imposed upon me because my noble birth placed me above association with such plebeian company. Now I know that its real object was to keep from my ears the idle tales of the dread curse upon our line that were nightly told and magnified by the simple tenantry as they conversed in hushed accents in the glow of their cottage hearths.

    Thus isolated, and thrown upon my own resources, I spent the hours of my childhood in poring over the ancient tomes that filled the shadow haunted library of the chateau, and in roaming without aim or purpose through the perpetual dust of the spectral wood that clothes the side of the hill near its foot. It was perhaps an effect of such surroundings that my mind early acquired a shade of melancholy. Those studies and pursuits which partake of the dark and occult in nature most strongly claimed my attention.""", 
    "Excerpt from Lovecraft, center aligned",
    alignment="center"))

    print(box("""It was in one of the vast and gloomy chambers of this remaining tower that I, Antoine, last of the unhappy and accursed Counts de C-, first saw the light of day, ninety long years ago. Within these walls and amongst the dark and shadowy forests, the wild ravines and grottos of the hillside below, were spent the first years of my troubled life. My parents I never knew. My father had been killed at the age of thirty-two, a month before I was born, by the fall of a stone somehow dislodged from one of the deserted parapets of the castle. And my mother having died at my birth, my care and education devolved solely upon one remaining servitor, an old and trusted man of considerable intelligence, whose name I remember as Pierre. I was an only child and the lack of companionship which this fact entailed upon me was augmented by the strange care exercised by my aged guardian, in excluding me from the society of the peasant children whose abodes were scattered here and there upon the plains that surround the base of the hill. At that time, Pierre said that this restriction was imposed upon me because my noble birth placed me above association with such plebeian company. Now I know that its real object was to keep from my ears the idle tales of the dread curse upon our line that were nightly told and magnified by the simple tenantry as they conversed in hushed accents in the glow of their cottage hearths.

    Thus isolated, and thrown upon my own resources, I spent the hours of my childhood in poring over the ancient tomes that filled the shadow haunted library of the chateau, and in roaming without aim or purpose through the perpetual dust of the spectral wood that clothes the side of the hill near its foot. It was perhaps an effect of such surroundings that my mind early acquired a shade of melancholy. Those studies and pursuits which partake of the dark and occult in nature most strongly claimed my attention.""", 
    "Excerpt from Lovecraft, right aligned",
    alignment="right"))