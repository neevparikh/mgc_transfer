def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    return ((size[-2] - (kernel_size[-2] - 1) - 1) // stride + 1,
            (size[-1] - (kernel_size[-1] - 1) - 1) // stride + 1)


fma_genre_list = {
    'Electronic': 0,
    'Experimental': 1,
    'Folk': 2,
    'Hip-Hop': 3,
    'hiphop': 3,
    'Instrumental': 4,
    'International': 5,
    'Pop': 6,
    'pop': 6,
    'Rock': 7,
    'rock': 7,
    'Blues': 8,
    'blues': 8,
    'Classical': 9,
    'classical': 9,
    'Country': 10,
    'country': 10,
    'Easy Listening': 11,
    'Jazz': 12,
    'jazz': 12,
    'Old-Time / Historic': 13,
    'Soul-RnB': 14,
    'Spoken': 15,
    'disco': 16,
    'metal': 17,
    'reggae': 18,
}

gtzan_genre_list = {
    'country': 0,
    'jazz': 1,
    'disco': 2,
    'hiphop': 3,
    'metal': 4,
    'reggae': 5,
    'pop': 6,
    'rock': 7,
    'blues': 8,
    'classical': 9,
}


def genre_to_label(genre, dataset):
    if 'fma' in dataset:
        return fma_genre_list[genre]
    elif 'gtzan' in dataset:
        return gtzan_genre_list[genre]
    else:
        raise ValueError("Unknown dataset")
