##########################
# Uniform Spanning Trees #
##########################


import numpy as np
from itertools import chain  # create graph edges from path

import networkx as nx
import matplotlib.pyplot as plt

def ust_sampler_wilson(list_of_neighbors, list_of_neighbor_probs, root=None,
                       random_state=None):
    try:
        import networkx as nx
    except ImportError:
        raise ValueError('The networkx package is required to sample spanning trees (see setup.py).')

    rng = check_random_state(random_state)

    # Initialize the tree
    wilson_tree_graph = nx.Graph()
    nb_nodes = len(list_of_neighbors)

    # Initialize the root, if root not specified start from any node
    n0 = root if root else rng.choice(nb_nodes)  # size=1)[0]
    # -1 = not visited / 0 = in path / 1 = in tree
    state = -np.ones(nb_nodes, dtype=int)
    state[n0] = 1
    nb_nodes_in_tree = 1

    path, branches = [], []  # branches of tree, temporary path

    while nb_nodes_in_tree < nb_nodes:  # |Tree| = |V| - 1

        # visit a neighbor of n0 uniformly at random
        probs = list_of_neighbor_probs[n0]

        eps=1e-6
        probs = [p+eps for p in probs]
        probs = [p/sum(probs) for p in probs]

        n1 = rng.choice(list_of_neighbors[n0], p=probs)  # size=1)[0]

        if state[n1] == -1:  # not visited => continue the walk

            path.append(n1)  # add it to the path
            state[n1] = 0  # mark it as in the path
            n0 = n1  # continue the walk

        if state[n1] == 0:  # loop on the path => erase the loop

            knot = path.index(n1)  # find 1st appearence of n1 in the path
            nodes_loop = path[knot + 1:]  # identify nodes forming the loop
            del path[knot + 1:]  # erase the loop
            state[nodes_loop] = -1  # mark loopy nodes as not visited
            n0 = n1  # continue the walk

        elif state[n1] == 1:  # hits the tree => new branch

            if nb_nodes_in_tree == 1:
                branches.append([n1] + path)  # initial branch of the tree
            else:
                branches.append(path + [n1])  # path as a new branch

            state[path] = 1  # mark nodes in path as in the tree
            nb_nodes_in_tree += len(path)

            # Restart the walk from a random node among those not visited
            nodes_not_visited = np.where(state == -1)[0]
            if nodes_not_visited.size:
                n0 = rng.choice(nodes_not_visited)  # size=1)[0]
                path = [n0]

    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]),
                                              branches)))
    wilson_tree_graph.add_edges_from(tree_edges)

    return wilson_tree_graph


class UST:
    """ DPP on edges of a connected graph :math:`G` with correlation kernel the projection kernel onto the span of the rows of the incidence matrix :math:`\\text{Inc}` of :math:`G`.

    This DPP corresponds to the uniform measure on spanning trees (UST) of :math:`G`.

    :param graph:
        Connected undirected graph
    :type graph:
        networkx graph

    .. seealso::

        - :ref:`UST`
        - :ref:`Definition of DPP <finite_dpps_definition>`
    """

    def __init__(self, graph, edge_scores):
        # For Uniform Spanning Trees
        try:
            import networkx as nx
            self.nx = nx
        except ImportError:
            raise ValueError('The networkx package is required to sample spanning trees (see setup.py).')

        if nx.is_connected(graph):
            self.graph = graph
        else:
            raise ValueError('graph not connected')

        self.nodes = list(self.graph.nodes())
        self.nb_nodes = self.graph.number_of_nodes()  # len(self.graph)

        self.edges = list(self.graph.edges())
        self.nb_edges = self.graph.number_of_edges()  # len(self.edges)




        self.edge_labels = {edge: r'$e_{}$'.format(i)
                            for i, edge in enumerate(self.edges)}

        self.neighbors = [list(graph.neighbors(v))
                          for v in range(self.nb_nodes)]

        #self.neighbor_edge_probs = [len(list(graph.neighbors(v)))*[1/len(list(graph.neighbors(v)))]
        #                  for v in range(self.nb_nodes)]
        self.neighbor_edge_probs = [[float(edge_scores[u,v]) for v in graph.neighbors(u)]
                          for u in range(self.nb_nodes)]

        self.sampling_mode = 'Wilson'  # Default (avoid eig_vecs computation)
        self._sampling_modes = {'markov-chain': ['Wilson', 'Aldous-Broder'],
                                'spectral-method': ['GS'],
                                'projection-K-kernel': ['Schur', 'Chol']}
        self.list_of_samples = []

        self.kernel = None
        self.kernel_eig_vecs = None

    def __str__(self):

        str_info = ['Uniform Spanning Tree measure on a graph with:',
                    '- {} nodes'.format(self.nb_nodes),
                    '- {} edges'.format(self.nb_edges),
                    'Sampling mode = {}'.format(self.sampling_mode),
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    # def info(self):
    #     """ Print infos about the :class:`UST` object
    #     """
    #     print(self.__str__())

    def flush_samples(self):
        """ Empty the :py:attr:`list_of_samples` attribute.
        """
        self.list_of_samples = []

    def sample(self, mode='Wilson', root=None, random_state=None):
        """ Sample a spanning of the underlying graph uniformly at random.
        It generates a networkx graph object.

        :param mode:

            Markov-chain-based samplers:

            - ``'Wilson'``, ``'Aldous-Broder'``

            Chain-rule-based samplers:

            - ``'GS'``, ``'GS_bis'``, ``'KuTa12'`` from eigenvectors
            - ``'Schur'``, ``'Chol'``, from :math:`\\mathbf{K}` correlation kernel

        :type mode:
            string, default ``'Wilson'``

        :param root:
            Starting node of the random walk when using Markov-chain-based samplers
        :type root:
            int

        :param random_state:
        :type random_state:
            None, np.random, int, np.random.RandomState

        .. seealso::

            - Wilson :cite:`PrWi98`
            - Aldous-Broder :cite:`Ald90`
            - :py:meth:`~dppy.FiniteDPP.sample`
        """

        rng = check_random_state(random_state)

        self.sampling_mode = mode

        if self.sampling_mode in self._sampling_modes['markov-chain']:
            if self.sampling_mode == 'Wilson':
                sampl = ust_sampler_wilson(self.neighbors, self.neighbor_edge_probs,
                                           random_state=rng)

            elif self.sampling_mode == 'Aldous-Broder':
                sampl = ust_sampler_aldous_broder(self.neighbors,
                                                  random_state=rng)

        else:
            err_print = '\n'.join(
                'Invalid sampling mode',
                'Chose from: {}'.format(self._sampling_modes.values()),
                'Given {}'.format(mode))
            raise ValueError()

        self.list_of_samples.append(sampl)

    def compute_kernel(self):
        """ Compute the orthogonal projection kernel :math:`\\mathbf{K} = \\text{Inc}^+ \\text{Inc}` i.e. onto the span of the rows of the vertex-edge incidence matrix :math:`\\text{Inc}` of size :math:`|V| \\times |E|`.

        In fact, for a connected graph, :math:`\\text{Inc}` has rank :math:`|V|-1` and any row can be discarded to get an basis of row space. If we note :math:`A` the amputated version of :math:`\\text{Inc}`, then :math:`\\text{Inc}^+ = A^{\\top}[AA^{\\top}]^{-1}`.

        In practice, we orthogonalize the rows of :math:`A` to get the eigenvectors :math:`U` of :math:`\\mathbf{K}=UU^{\\top}`.

        .. seealso::

            - :py:meth:`plot_kernel`
        """

        if self.kernel is None:
            self.compute_kernel_eig_vecs()  # U = QR(Inc[:-1,:].T)
            # K = UU.T
            self.kernel = self.kernel_eig_vecs.dot(self.kernel_eig_vecs.T)

    def compute_kernel_eig_vecs(self):
        """ See explaination in :func:`compute_kernel <compute_kernel>`
        """

        if self.kernel_eig_vecs is None:

            inc_mat = self.nx.incidence_matrix(self.graph, oriented=True)
            # Discard any row e.g. the last one
            A = inc_mat[:-1, :].toarray()
            # Orthonormalize rows of A
            self.kernel_eig_vecs, _ = qr(A.T, mode='economic')

    def plot(self, title=''):
        """ Display the last realization (spanning tree) of the corresponding :class:`UST` object.

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :py:meth:`sample`
        """

        graph_to_plot = self.list_of_samples[-1]

        plt.figure(figsize=(4, 4))

        pos = self.nx.circular_layout(self.graph)
        self.nx.draw_networkx(graph_to_plot,
                         pos=pos,
                         node_color='orange',
                         with_labels=True,
                         width=3)

        edge_labs = {e: self.edge_labels[e if e in self.edges else e[::-1]]
                     for e in graph_to_plot.edges()}
        self.nx.draw_networkx_edge_labels(graph_to_plot,
                                     pos=pos,
                                     edge_labels=edge_labs,
                                     font_size=20)

        plt.axis('off')

        str_title = 'A realization of UST with {} procedure'.format(self.sampling_mode)
        plt.title(title if title else str_title)

    def plot_graph(self, title=''):
        """Display the original graph defining the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :func:`compute_kernel <compute_kernel>`
        """

        # edge_lab = [r'$e_{}$'.format(i) for i in range(self.nb_edges)]
        # edge_labels = dict(zip(self.edges, edge_lab))
        # node_labels = dict(zip(self.nodes, self.nodes))

        plt.figure(figsize=(4, 4))

        pos = self.nx.circular_layout(self.graph)
        self.nx.draw_networkx(self.graph,
                         pos=pos,
                         node_color='orange',
                         with_labels=True,
                         width=3)
        # nx.draw_networkx_labels(self.graph,
        #                         pos,
        #                         node_labels)
        self.nx.draw_networkx_edge_labels(self.graph,
                                     pos=pos,
                                     edge_labels=self.edge_labels,
                                     font_size=20)

        plt.axis('off')

        str_title = 'Original graph'
        plt.title(title if title else str_title)

    def plot_kernel(self, title=''):
        """Display a heatmap of the underlying orthogonal projection kernel :math:`\\mathbf{K}` associated to the DPP underlying the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :func:`compute_kernel <compute_kernel>`
        """

        self.compute_kernel()

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        heatmap = ax.pcolor(self.kernel, cmap='jet')

        ax.set_aspect('equal')

        ticks = np.arange(self.nb_edges)
        ticks_label = [r'${}$'.format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        str_title = 'Correlation K kernel: transfer current matrix'
        plt.title(title if title else str_title, y=1.08)

        # Adapt size of colbar to plot
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        cax = fig.add_axes([ax.get_position().x1 + 0.02,
                            ax.get_position().y0,
                            0.05,
                            ax.get_position().height])
        plt.colorbar(heatmap, cax=cax)



def check_random_state(seed: object) -> np.random.RandomState:
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState

    .. seealso::

        `Scikit learn source code <https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/utils/validation.py#L763>`_
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

