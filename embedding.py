from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.finite_rings.finite_field_constructor import FiniteField as GF
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.groups.matrix_gps.linear import GL

from spnutils import symb_ring, symb_element, symb_element_to_bit_matrix

class Embedding:
    """
    Represents an embedding from self.R_from into self.R_to
    """
    def __init__(self, R_from, R_to, f):
        self.R_from = R_from
        self.R_to = R_to
        self.R_base_from = PolynomialRing(R_from.base_ring(), R_from.gen())
        self.R_base_to = PolynomialRing(R_to.base_ring(), R_to.gen())
        self.f = self.R_base_to(f)
        self.f_inv = None
    
    def embed(self, x):
        """ Given x in self.R_from, returns its embedding in self.R_to """
        assert(x in self.R_from)
        y = self.R_base_from(x).subs(self.f)
        return self.R_to(y)

    def bit_embedding(self):
        """ 
            Returns the embedding self.R_from -> self.R_to as bit-level matrix.
            A 1 in row i and column j denotes that coefficient j of the input element should be part of the sum to yield coefficient i of the embedded element.
        """
        n_bits = self.R_from.modulus().degree()
        R_bit = GF(2**(n_bits+1), name='b_')
        R = symb_ring(n_bits)
        element = symb_element(R)
        symbolic_R_from_element = element.subs(self.f)
        modulus = self.R_base_to(self.R_to.modulus())
        bit_embedding = symbolic_R_from_element % modulus
        bit_embedding_matrix = symb_element_to_bit_matrix(bit_embedding)
        return bit_embedding_matrix

    def bit_embedding_xor_cost(self):
        """ Returns the number of XOR operations required to embedd an element """
        bit_embedding = matrix(ZZ, self.bit_embedding())
        xors = 0
        for row in bit_embedding:
            s = sum(row)
            if s > 1:
                xors += s-1
        return xors
    
    def _compute_inverse_embedding(self):
        if self.f_inv is not None:
            return
        n = self.R_from.modulus().degree()
        temp = matrix(GF(2), n, n, 0)
        self.f_inv_coefs = list()
        for i,row in enumerate(self.bit_embedding()):
            if any((x != 0 for x in row)) and len(self.f_inv_coefs) < n:
                temp[len(self.f_inv_coefs),:] = row
                self.f_inv_coefs.append(i)
        self.f_inv = temp**-1
    
    def inverse_bit_embedding(self):
        """ 
            Returns the- inverse embedding of self.R_from -> self.R_to as a sparse bit-level matrix.
            A 1 in row i and column j denotes that coefficient j of the input element should be part of the sum to yield coefficient i of the inverted element.
        """
        self._compute_inverse_embedding()
        return self.f_inv, self.f_inv_coefs
    
    def inverse_bit_embedding_xor_cost(self):
        """ Returns the number of XOR operations required to invert the embedding """
        self._compute_inverse_embedding()
        xors = 0
        for row in matrix(ZZ, self.f_inv):
            s = sum(row)
            if s > 1:
                xors += s-1
        return xors
    
    def print_impl_forward(self):
        """ Prints pseudo-code to compute the embedding """
        m = self.bit_embedding()
        for yi,y in enumerate(m):
            if any(xi != 0 for xi in y):
                x = [f'x{i}' for i,xi in enumerate(y) if xi != 0]
                rhs = ' + '.join(x)
                print(f'y{yi} = {rhs}')
    
    def print_impl_backward(self):
        """ Prints pseudo-code to invert the embedding """
        m, coefs = self.inverse_bit_embedding()
        for xi,x in enumerate(m):
            y = [f'y{coefs[i]}' for i,b in enumerate(x) if b != 0]
            rhs = ' + '.join(y)
            print(f'x{xi} = {rhs}')
    
    def __repr__(self):
        return f'Embedding from {self.R_from} to {self.R_to} via {self.f}'

def all_embeddings(R_from, R_to):
    """ Returns alist of all possible embedding from R_from to R_to """
    p_from = R_from.modulus()
    p = p_from.change_ring(R_to)
    return [Embedding(R_from, R_to, r) for r,_ in p.roots()]


def find_min_cost_embedding(F_from, F_to, weight=None, custom_cost=None):
    """
    Returns the embedding of F_from to F_to that has minimum XOR cost.
    weight['forward'] and weight['inverse'] can contain weight factors to the XOR cost of embedding and inverting an element.
    custom_cost can be a function to add customized cost
    
    Returns the embedding and the minimum cost
    """
    if weight is None:
        weight = {'forward': 1, 'inverse': 1}
    min_cost = -1
    min_cost_embedding = None
    for e in all_embeddings(F_from, F_to):
        cost = weight['forward'] * e.bit_embedding_xor_cost() + weight['inverse'] * e.inverse_bit_embedding_xor_cost()
        if custom_cost is not None:
            cost += custom_cost(e,weight)
        if min_cost_embedding is None or cost < min_cost:
            min_cost = cost
            min_cost_embedding = e
    return min_cost_embedding, min_cost