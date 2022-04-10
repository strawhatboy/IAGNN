
from typing import Dict, List
import numpy as np

def data_statistics(all_data: Dict[str, List[List[List]]]):
    '''
    get statistics from data

    Args:
        data (List[List[List]]): list of [0, [item], [next_item](1), [category], [next_category](1)]
    '''
    data = all_data['train'] + all_data['test']
    items = set()
    total_session_length = 0
    cats = set()
    total_cat_per_session = 0

    for x in data:
        total_session_length += len(x[1])
        for i in x[1]:
            items.add(i)
        items.add(x[2][0])
        for c in x[3]:
            cats.add(c)
        cats.add(x[4][0])

        total_cat_per_session += len(np.unique(x[3]))
    
    print('')
    print('* dataset statistics:')
    print('=====================')
    print('No. of items: {}'.format(len(items)))
    print('No. of sessions: {}'.format(len(data)))
    print('Avg. of session length: {}'.format(total_session_length / len(data)))
    print('No. of categories: {}'.format(len(cats)))
    print('No. of cats/session: {}'.format(total_cat_per_session / len(data)))
    print('')



