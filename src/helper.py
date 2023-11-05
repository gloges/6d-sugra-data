import numpy as np
from src.clique import IrreducibleClique


dtype_hypers = np.dtype([('nR', int), ('irrep', str, 255)])
dtype_vertex = np.dtype([('ID', str, 255), ('delta', int), ('bibi', int), ('b0bi', int), ('hypers', object)])


def display_irreps(group, rank, folder):

    print(f'\nIrreps for {group}[{rank}]:\n')

    group_ID = group + str(rank).rjust(2, '0')
    filepath = folder + f'/irreps/{group}/{group_ID}-irreps.tsv'

    if group_ID in ['B03', 'D04']:
        print(f"    {'ID':14}{'H':>10}{'A':>10}{'B':>10}{'C+3B/4':>10}    {'quat?':8}{'hw vector'}")
    else:
        print(f"    {'ID':14}{'H':>10}{'A':>10}{'B':>10}{'C':>10}    {'quat?':8}{'hw vector'}")
    print('    ' + max(75, 67+2*rank)*'—')

    with open(filepath, 'r') as file:
        for line in file:
            # Remove '\n' and split by tabs
            data = line[:-1].split('\t')
            id = data[0]
            H = int(data[1])
            A = int(data[2])
            B = int(data[3])
            C = int(data[4])
            quat = data[5]
            hw_vec = data[6]
            print(f'    {id:14}{H:>10,}{A:>10,}{B:>10,}{C:>10,}    {quat:8}{hw_vec}')

    print()

def get_irrep_data(group, rank, folder):

    group_ID = group + str(rank).rjust(2, '0')
    filepath = folder + f'/irreps/{group}/{group_ID}-irreps.tsv'

    irrep_data = []

    with open(filepath, 'r') as file:
        for line in file:
            # Remove '\n' and split by tabs
            data = line[:-1].split('\t')
            id = data[0]
            H = int(data[1])
            A = int(data[2])
            B = int(data[3])
            C = int(data[4])
            quat = (data[5] == 'True')
            hw_vec = tuple([int(aa) for aa in data[6][1:-1].split(',')])

            irrep_data.append([id, H, A, B, C, quat, hw_vec])

    return irrep_data

def get_vertex_data(group, rank, folder):

    group_ID = group + str(rank).rjust(2, '0')
    filepath = folder + f'/vertices/{group}/{group_ID}-vertices.tsv'

    vertex_data = []

    with open(filepath, 'r') as file:
        for line in file:
            # Remove '\n' and split by tabs
            data = line[:-1].split('\t')
            id = data[0]
            delta = int(data[1][4:])
            bibi = int(data[2][8:])
            b0bi = int(data[3][8:])
            hypers = [hyp.split(' x ') for hyp in data[4].split(' + ')]
            hypers = [(int(nR), irr[1:-1]) for nR, irr in hypers]
            hypers = np.array(hypers, dtype=dtype_hypers)

            vertex_data.append((id, delta, bibi, b0bi, hypers))

    vertex_data = np.array(vertex_data, dtype=dtype_vertex)

    return vertex_data

def get_irreducible_cliques(filepath):

    cliques_irreducible = []

    with open(filepath, 'r') as file:
        for line in file:
            clq = IrreducibleClique(line)
            cliques_irreducible.append(clq)

    print(f'{len(cliques_irreducible):10,} cliques loaded from {filepath}')

    return cliques_irreducible

def filter_irreducible_cliques_by_vertex(cliques, vertex_IDs, match):

    if match not in ['any', 'all']:
        raise Exception("Expecting match='any' or match='all'.")
    
    cliques_filtered = []

    for clique in cliques:
        if match == 'any':
            if any([vertex_ID in clique.vertices for vertex_ID in vertex_IDs]):
                cliques_filtered.append(clique)
                
        elif match == 'all':
            # Check multiplicities too!
            to_add = True
            for vertex_ID, count in zip(*np.unique(vertex_IDs, return_counts=True)):
                if count > sum([vtx == vertex_ID for vtx in clique.vertices]):
                    to_add = False
                    break

            if to_add:
                cliques_filtered.append(clique)

    return cliques_filtered

def filter_irreducible_cliques_by_gauge_group(cliques, simple_factors, exact=True):

    simple_factors = simple_factors.split(' x ')

    cliques_filtered = []

    for clique in cliques:
        if exact:
            if sorted(simple_factors) == sorted(clique.gauge_group.split(' x ')):
                cliques_filtered.append(clique)

        else:
            # Check if subgroup
            to_add = True
            for simple_factor, count in zip(*np.unique(simple_factors, return_counts=True)):
                if count > sum([grp == simple_factor for grp in clique.gauge_group.split(' x ')]):
                    to_add = False
                    break

            if to_add:
                cliques_filtered.append(clique)


    return cliques_filtered
