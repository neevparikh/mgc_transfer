def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    return ((size[-2] - (kernel_size[-2] - 1) - 1) // stride + 1,
            (size[-1] - (kernel_size[-1] - 1) - 1) // stride + 1)


genre_list = [
        'Electronic',
        'Experimental',
        'Folk',
        'Hip-Hop',
        'Instrumental',
        'International',
        'Pop',
        'Rock',
        'Blues',
        'Classical',
        'Country',
        'Easy, Listening',
        'Jazz',
        'Old-Time, / Historic',
        'Soul-RnB',
        'Spoken',
    ]

def genre_to_label(genre):
    return genre_list.index(genre)
