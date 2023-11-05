import numpy as np
from src.gramMatrix import *


class IrreducibleClique():

    def __init__(self, string_tsv):

        # Remove '\n' and split by tabs
        data = string_tsv[:-1].split('\t')

        # Unique identifier
        self.ID = data[0]

        # Number of type-A and type-B vertices
        self.num_AB = [int(xx) for xx in self.ID.split('-')[1][1:-1].split(',')]

        # List of vertex IDs
        self.vertices = data[1][1:-1].split(', ')

        # Gauge group string
        self.gauge_group = []
        for vertex in self.vertices:
            group_ID = vertex.split('-')[1]

            group_series = group_ID[0]
            group_rank = int(group_ID[1:])

            if   group_series == 'A': group_string = f'SU({group_rank+1})'
            elif group_series == 'B': group_string = f'SO({2*group_rank+1})'
            elif group_series == 'C': group_string = f'Sp({group_rank})'
            elif group_series == 'D': group_string = f'SO({2*group_rank})'
            else:                     group_string = f'{group_series}({group_rank})'

            self.gauge_group.append(group_string)

        self.gauge_group = ' x '.join(self.gauge_group)

        # Δ and Δ+28nneg
        self.delta = int(data[2][4:])
        self.delta_28n = int(data[3][8:])

        # Tmin
        self.T_min = int(data[4][7:])

        # Signature of g = bi.bj (npos, nneg)
        sign = data[5].split('[')[1].split(']')[0].split(', ')
        self.n_pos_bibj = int(sign[0])
        self.n_neg_bibj = int(sign[1])

        # bi.bj and b0.bi
        self.gram_bibj = np.array([[int(aa) for aa in row.split(', ')] for row in data[6][10:-2].split('], [')])
        self.gram_b0bi = np.array([int(aa) for aa in data[7][9:-1].split(', ')])

        self.hyper_strings = data[8:]

        self.hypers = {}

        for irrep_data in data[8:]:
            key, irreps = irrep_data.split(' : ')

            key = tuple([int(aa) for aa in key[1:-1].split(',')])
            irreps = irreps.split(' + ')
            irreps = [irr.split(' x ') for irr in irreps]
            irreps = [[int(nR), irr_ID[1:-1].split(', ')] for nR, irr_ID in irreps]

            self.hypers[key] = irreps

    def display(self):

        print()
        print(f'           ID : {self.ID}')
        print('     vertices :', ', '.join(self.vertices))
        print(f'            G : {self.gauge_group}')


        print('       hypers : ', end='')
        pads = [hyper_string.find(':') for hyper_string in self.hyper_strings]
        maxpad = max(pads)

        for ii, [hyper_string, pad] in enumerate(zip(self.hyper_strings, pads)):
            if ii != 0:
                print(16*' ', end='')
            print((maxpad - pad)*' ' + hyper_string)

        print(f'        b0·bi : {self.gram_b0bi}')

        if np.max(np.abs(self.gram_bibj)) == 0:
            ndigits = 1
        else:
            ndigits = int(np.log10(np.max(np.abs(self.gram_bibj)))) + 1
        if np.min(self.gram_bibj) < 0:
            ndigits += 1
        print('        bi·bj : [[', end='')
        for ii, row in enumerate(self.gram_bibj):
            if ii != 0:
                print(17*' ', end='[')
            print(' '.join([f'{val:{ndigits}}' for val in row]), end='')
            if ii == len(self.gram_bibj) - 1:
                print(']', end='')
            print(']')

        print(f'  sign(bi·bj) : ({self.n_pos_bibj}, {self.n_neg_bibj})')
        
        print(f'            Δ : {self.delta}')
        print(f'        Δ+28n : {self.delta_28n}')
        print(f'        T_min : {self.T_min}')
        print()

    def get_nontrivial_adjacency_matrix(self):

        adjacency_matrix = self.gram_bibj.copy()
        adjacency_matrix -= np.diag(np.diag(adjacency_matrix))
        adjacency_matrix = np.minimum(adjacency_matrix, 1)

        return adjacency_matrix

class Clique():

    def __init__(self, clique_irr=None):

        self.irreducible_cliques = []

        self.num_AB = [0, 0]
        self.delta = 0
        self.delta_28n = 0

        self.n_pos_bibj = 0
        self.n_neg_bibj = 0

        if clique_irr is not None:
            self.add_irreducible_clique(clique_irr)

    def display(self):

        print('Irreducible components:')
        for clique_irr in self.irreducible_cliques:
            print('  ', clique_irr.ID)

        print(f'\n    (NA,NB) = ({self.num_AB[0]},{self.num_AB[1]})')
        print('          Δ =', self.delta)
        print('   Δ+28nneg =', self.delta + 28*self.n_neg_bibj)
        print(f'sign(bi.bj) = ({self.n_pos_bibj},{self.n_neg_bibj})')

    def add_irreducible_clique(self, clique_irr):

        self.irreducible_cliques.append(clique_irr)

        self.num_AB[0] += clique_irr.num_AB[0]
        self.num_AB[1] += clique_irr.num_AB[1]

        self.delta += clique_irr.delta
        self.delta_28n += clique_irr.delta_28n

        self.n_pos_bibj += clique_irr.n_pos_bibj
        self.n_neg_bibj += clique_irr.n_neg_bibj

    def clone(self):

        clone = Clique()

        for clique_irr in self.irreducible_cliques:
            clone.add_irreducible_clique(clique_irr)

        return clone

    def get_gram(self):
        
        # Build up b0.bi and block-diagonal bi.bj
        gram_bibj = np.zeros([0, 0], dtype=int)
        gram_b0bi = np.zeros(0, dtype=int)

        for clique_irr in self.irreducible_cliques:
            k = sum(clique_irr.num_AB)
            gram_bibj = np.pad(gram_bibj, (0,k))
            gram_bibj[-k:][:,-k:] = clique_irr.gram_bibj.copy()

            gram_b0bi = np.append(gram_b0bi, clique_irr.gram_b0bi)

        return gram_bibj, gram_b0bi
