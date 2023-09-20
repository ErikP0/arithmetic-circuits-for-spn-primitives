from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.finite_rings.finite_field_constructor import FiniteField as GF
from sage.rings.finite_rings.integer_mod_ring import IntegerModRing
from sage.matrix.constructor import matrix, vector
from sage.rings.integer_ring import ZZ


def symb_ring(n_bits):
    """ Returns a ring to symbolicly compute over GF(2^nbits) """
    F_symb = GF(2**(n_bits+1), name='b_')
    R = PolynomialRing(F_symb, name='x')
    return R

def symb_element(R_symb):
    """ Returns a symbolic ring element of R_symb """
    x = R_symb.gen()
    b = R_symb.base_ring().gen()
    n = R_symb.base_ring().modulus().degree()-1
    return R_symb(sum([b**(i+1) * x**i for i in range(n)]))

def symb_element_to_bit_matrix(element):
    """ Returns a matrix that indicates the computation rule for each bit in the symbolic element.
    For example, the symbolic element (b_0 + b_1) + (b_1 + b_2) * x of the symbolic ring over 3 bits yields
        [1, 1, 0] // bit 0 of the element is b0 + b1
        [0, 1, 1] // bit 1 of the element is b1 + b2
        [0, 0, 0] // bit 2 of the element is always 0
    """ 
    R_symb = element.parent()
    n = R_symb.base_ring().modulus().degree()-1
    coefs = element.coefficients(sparse=False)
    impl_matrix = matrix(GF(2), len(coefs), n, 0, sparse=True)
    for i, bit_expr in enumerate(coefs):
            if bit_expr != 0:
                R = PolynomialRing(bit_expr.base_ring(), bit_expr.parent().gen())
                for j,c in enumerate(R(bit_expr).coefficients(sparse=False)):
                    if c != 0:
                        impl_matrix[i,j-1] = 1
    return impl_matrix

def gf_squaring(F):
    """
        Returns the implementation matrix M s.t. b = M * a where a is the vector of coefficients of a GF(2^k) element to be squared and b is the vector of coefficients of the resulting element
        
        F: the field to square in (must be binary extension field)
    """
    p = F.modulus()
    n = p.degree()
    R = symb_ring(n)
    b = symb_element(R)
    b_sq = R(sum([bi * R.gen()**(2*i) for i,bi in enumerate(b.coefficients(sparse=False))]))
    b_sq = b_sq % R(p.subs(R.gen()))
    impl_matrix = symb_element_to_bit_matrix(b_sq)
    return impl_matrix

def gf_squaring_k(F, k):
    """
    Compute the implementation matrix for squaring an element k times
    """
    assert k >= 0
    p = F.modulus()
    n = p.degree()
    R = symb_ring(n)
    b_sq = symb_element(R)
    for i in range(k):
        b_sq = R(sum([bi * R.gen()**((2*i)) for i,bi in enumerate(b_sq.coefficients(sparse=False))]))
        b_sq = b_sq % R(p.subs(R.gen()))
    impl_matrix = symb_element_to_bit_matrix(b_sq)
    return impl_matrix

def gf_squaring_xor_cost(F):
    """ Returns the number of XOR operations required to compute squaring in F """
    impl_matrix = matrix(ZZ, gf_squaring(F))
    xors = 0
    for row in impl_matrix:
        s = sum(row)
        if s > 1:
            xors += s-1
    return xors

def gf_cmul(F, c):
    """ 
        Returns the implementation matrix M s.t. b = M * a where a is the vector of coefficients of a GF(2^k) element and 
        b is the vector of coefficients of the resulting element multiplied with constant c (w.r.t. arithmetic in GF(2^k)
        
        F: the field to multiply in (must be binary extension field)
        c: element in F
    """
    p = F.modulus()
    n = p.degree()
    assert 0 <= c < 2**n
    R = symb_ring(n)
    x = symb_element(R)
    c = R(sum([R.gen()**i for i in range(n) if ((c >> i) & 0x1) > 0]))
    cx = (c * x) % R(p.subs(R.gen()))
    impl_matrix = symb_element_to_bit_matrix(cx)
    impl_matrix = matrix(ZZ, impl_matrix)
    return impl_matrix
    
def gf_cmul_xor_cost(F, c):
    """ Returns the number of XOR operations required to compute multiplication by c in F """
    impl_matrix = gf_cmul(F,c)
    xors = 0
    for row in impl_matrix:
        s = sum(row)
        if s > 1:
            xors += s-1
    return xors

def print_cmul_code(F, constants):
    """ Prints pseudo-code for multiplication by each c in constants """
    for c in constants:
        impl = gf_cmul(F, c)
        nrows = impl.nrows()
        ncols = impl.ncols()
        print(f'{c}: ')
        for i in range(nrows):
            rhs = ' + '.join(f'x{j}' for j in range(ncols) if impl[i,j] == 1)
            print(f'    y{i} = {rhs}')


def interpolate(F, x, y, n=None):
    """
        Returns the coefficients of the interpolating polynomial for points (x[0], y[0]), ... in F.
        n denotes the maximum degree of the polynomial
    """
    assert len(x) == len(y)
    if n is None:
        n = 2**F.modulus().degree()
    assert len(x) <= n, f'len(x) = {len(x)}, n = {n}'
    A = matrix(F, len(x), n, 0)
    for i,x in enumerate(x):
        for j in range(n):
            A[i,j] = x**j
    b = vector(F, len(y), y)
    c = A.solve_right(b)
    return c

def compute_gf_squaring_constants(F, k):
    """
    Returns a list of constants to use in
    ```
    x = element in F to square
    bd = coefficients of x
    c = compute_gf_squaring_constants(F, k)
    coefficient i of x**(2**k) =  c[i] * bd[i]
    ```
    """
    impl = gf_squaring_k(F,k)
    # build constant over columns
    cols = impl.columns()
    ncols = len(cols)
    constants = []
    for col in cols:
        constants.append(sum([2**i for i,c in enumerate(col) if c == 1]))
    return constants

def natural_encoding(F, x):
    """
    Returns an element in F s.t. bit i of x is i-th coefficient in the returned element
    """
    X = F.gen()
    n = F.modulus().degree()
    assert x < 2**n
    return F(sum([X**i for i in range(n) if ((x >> i) & 0x1) > 0]))

def natural_encoding_to_int(x):
    """
    Returns a number where the i-th coefficient of x is the i-th bit of the returned number
    """
    R = PolynomialRing(x.parent().base_ring(), x.parent().gen())
    return sum((2**i for i,c in enumerate(R(x).coefficients(sparse=False)) if c != 0))

class ShallowAdditionChainResult:
    """ Class to represent a result of a shallow addition chain for a given set of values to compute. Shallow means that as many additions can be performed in parallel as possible. """
    def __init__(self, values, levels, sq_bases, frees):
        self._values = values
        self._levels = levels
        self._sq_bases = sq_bases
        self._frees = frees
    def required_levels(self):
        """ Returns the number of levels, i.e., non-parallel additions. """
        return len(self._levels)
    def print_levels(self):
        for i,level in enumerate(self._levels):
            steps = [f'{z} = {x} + {y}' for x, y, z in level if z not in self._frees]
            free_square_steps = [f'{z} = {x} + {y}' for x, y, z in level if z in self._frees]
            print(f'Round {i+1} [free]: ' + ', '.join(free_square_steps))
            print(f'Round {i+1}: ' + ', '.join(steps))
    def required_additions(self):
        sums = sum(1 for level in self._levels for x, y, z in level if x != y and z not in self._frees)
        doubles = sum(1 for level in self._levels for x, y, z in level if x == y and z not in self._frees)
        return sums, doubles, len(self._sq_bases)


def shallow_dense_addition_chain(values, n, doubles=[]):
    """ 
    Computes a shallow addition chain to compute all numbers in values in GF(2^n).
    This assumes that we don't need to generate intermediary values that are not in values.
    doubles is a list of elements in values that generate more values via doubling, e.g., if a in doubles, then the chain will compute 2a, 4a, 8a, 16a, ...
    
    Returns a `ShallowAdditionChainResult`
    """
    def add_doubles(d, max_value):
        i = 1
        # doubles are cyclic
        l = list()
        while (2**(i-1)*d % (2**n-1), (2**i*d % (2**n-1))) not in l:
            l.append((2**(i-1)*d % (2**n-1), (2**i*d % (2**n-1))))
            i += 1
        # remove d
        for i in range(len(l)):
            if l[i][1] == d:
                l.pop(i)
        return l
        
    remaining = set(values) - {0,1}
    available = set([0,1])
    levels = list()
    used_doubles = list()
    max_value = max(values)
    next_doubles = list()
    frees = set()
    if 1 in doubles:
        next_doubles += add_doubles(1, max_value)
        new_level = []
        for x,z in next_doubles:
            new_level.append((x, x, z))
            frees.add(z)
            available.add(z)
            if z in remaining:
                remaining.remove(z)
        next_doubles.clear()
        levels.append(new_level)
    while len(remaining) > 0:
        # try to produce a value from values
        new_level = list()
        new_available = set()
        # add next doubles
        for x,z in next_doubles:
            new_level.append((x, x, z))
            available.add(z)
            frees.add(z)
            if z in remaining:
                remaining.remove(z)
        next_doubles.clear()
        for target in remaining:
            if target in frees or any(map(lambda x: x[1] == target, next_doubles)):
                continue
            for x in available:
                y = (target - x) % (2**n-1)
                if y > 0 and y in available:
                    new_level.append((x,y,target))
                    new_available.add(target)
                    if target in doubles:
                        next_doubles += add_doubles(target, max_value)
                    break
        if len(new_level) == 0:
            raise ValueError(f'Cannot produce any of {remaining} without introducing intermediary values')
        levels.append(new_level)
        # computed values become available
        available |= new_available
        # remove computed values from remaining
        remaining -= new_available
    # check consistency
    vals = set([0,1])
    for level in levels:
        for x,y,z in level:
            if x not in vals or y not in vals:
                raise AssertionError(f'Cannot compute {z} = {x} + {y} from {vals}')
            vals.add(z)
    if len(vals.intersection(values)) != len(values):
        raise AssertionError(f'Some powers in {values} are not computed in {vals}')
    return ShallowAdditionChainResult(set(values), levels, list(doubles), frees)

def find_field(size, cost_f, keep_summary=False):
    """
        Returns all fields GF(2**size) that minimize the cost function cost_f.
        If keep_summary=True, a list of all candidate fields and their cost is returned as well.
        
        Returns the minimum cost, the fields (, and all candidates with their cost)
    """
    cost = None
    best_fields = None
    R = IntegerModRing(2)['x']
    if keep_summary:
        summary = []
    for p in R.polynomials(of_degree=size):
        if p.is_irreducible():
            F = GF(2**size, name=R.gen(), modulus=p)
            c = cost_f(F)
            if cost == None or c < cost:
                cost = c
                best_fields = [F]
            elif c == cost:
                best_fields.append(F)
            if keep_summary:
                summary.append((F,c))
    if keep_summary:
        return cost, best_fields, summary
    else:
        return cost, best_fields

def find_field_for_sparse_interpolation(size, x, y, summary=False):
    """
        Returns all fields where the interpolation of the points (x[0], y[0]), ... is maximally sparse (when encoded with `natural_encoding`)
        
        Returns a list of fields, the number of non-zero coefficients (and if summary=True, all candidates with the number of non-zero coefficients)
    """
    def sparse_interpolation_cost(F):
        x_f = [natural_encoding(F, xi) for xi in x]
        y_f = [natural_encoding(F, yi) for yi in y]
        c = interpolate(F, x_f, y_f)
        # number of non-zeros
        return sum(1 for ci in c if ci != 0)
    res = find_field(size, sparse_interpolation_cost, keep_summary=summary)
    if summary:
        min_cost, fields, cost_summary = res
        return fields, min_cost, cost_summary
    else:
        min_cost, fields = res
        return fields, min_cost