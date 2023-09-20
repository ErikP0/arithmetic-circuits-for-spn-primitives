from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.matrix.constructor import matrix
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector
from multiprocessing import Pool
import itertools
import math

def compute_polynomial(coefs, powers):
    assert len(coefs) == len(powers), f'len({coefs}) == len({powers})'
    return sum(c * p for c,p in zip(coefs, powers))

class CRVResult:
    """
    Stores the parameters of a CRV decomposition
    """
    def __init__(self, n, t, alphas, L, q, p):
        assert(len(q) + 1 == len(p))
        self.n = n
        self.t = t
        self.alphas = alphas
        self.L = L
        self.q = q
        self.p = p
    
    def multiplication_cost(self):
        """ Returns the number of non-free multiplications needed to evaluate the polynomial """
        return len(self.alphas)-2 + len(self.q)
    
    def highest_L_power(self):
        """ Returns the largest power in L that is required to compute the p and q polynomials """
        i = len(self.L)-1
        while i >= 0:
            if any((qj[i] != 0 for qj in self.q)) or any((pj[i] != 0 for pj in self.p)):
                return self.L[i]
            i -= 1
        return self.L[0]
    
    def compute(self, x):
        """ Evaluate the polynomial on x """
        # compute powers of x
        powers = [x**p for p in self.L]
        res = 0
        for i in range(self.t-1):
            pi = compute_polynomial(self.p[i], powers)
            qi = compute_polynomial([self.q[i][p] for p in self.L], powers)
            res += pi * qi
        res += compute_polynomial(self.p[self.t-1], powers)
        return res

def _generate_cyclotomic_class(n, alpha):
    """
    Returns a set of powers of two of alpha mod (2**n - 1)
    """
    return set(((alpha << i) % (2**n-1) for i in range(2**n)))
    
def _integers_of_cyclotomic_classes(n, alphas):
    """
    Returns all powers of two of alpha in alphas mod (2**n - 1)
    """
    L = set()
    for alpha in alphas:
        L.update(_generate_cyclotomic_class(n, alpha))
    return L

def _into_F(F, x, n):
    return F(sum([((x >> i) & 0x1) * F.gen()**i for i in range(n)]))

class CRV:
    """ 
        Class to compute the poylnomial decomposition by Coron, Roy and Vivek (https://eprint.iacr.org/2014/890)
    """
    def __init__(self, F, cyclotomic_alphas):
        self.F = F
        self.R = PolynomialRing(F, name='z')
        self.n = F.modulus().degree()
        self.alphas = cyclotomic_alphas
        self.L = sorted(list(_integers_of_cyclotomic_classes(self.n, cyclotomic_alphas)))
        self.A = None
        self.q = None
        self.t = None
    
    def _generate_random_poly_q(self):
        coefs = [self.F.random_element() for i in range(len(self.L))]
        return sum((c * self.R.gen()**p for c,p in zip(coefs, self.L)))
    
    def _generate_A(self, q, t):
        R_elements = [_into_F(self.F, i, self.n) for i in range(2**self.n)]
        A = matrix(self.F, len(R_elements), t * len(self.L))
        for j, a_j in enumerate(R_elements):
            for i in range(t-1):
                for k,p in enumerate(self.L):
                    A[j,i*len(self.L)+k] = a_j**p * q[i](a_j)
            for k,p in enumerate(self.L):
                A[j,(t-1)*len(self.L)+k] = a_j**p
        return A
    
    def set_q_polynomials(self, q):
        """ Manually sets the heuristic q polynomials """
        assert all(len(qi) == len(self.L) for qi in q)
        assert (len(q)+1) * len(self.L) >= 2**self.n
        self.t = len(q)+1
        self.q = [sum(c * self.R.gen()**p for c,p in zip(qi, self.L)) for qi in q]
        A = self._generate_A(self.q, self.t)
        assert A.rank() >= 2**self.n
        self.A = A
    
    def find_q_polynomials(self, t=None, tries=10, check_full_rank=True, sbox=None):
        """ 
            Heuristically finds q polynomials for the given parameters self.L, t by chosing random polynomials
            t: number of q polynomials to generate
            tries: the number of tries before failing
            check_full_rank: if true, check that the resulting system has full rank. This only finds parameters that decompose **any** S-box
                if false, then sbox must be supplied. The found parameters may only decompose that specific S-box
            sbox: S-box to interpolate
        """
        if t is None:
            t = int(math.ceil(2**self.n/len(self.L)))
        assert t * len(self.L) >= 2**self.n, f't = {t}, len(L) = {len(self.L)}, 2**n = {2**self.n}'
        cnt = 0
        if not check_full_rank:
            assert sbox != None
            b = matrix(self.F, 2**self.n, 1, [x for x in sbox])
        while cnt < tries:
            q = [self._generate_random_poly_q() for i in range(t-1)]
            A = self._generate_A(q, t)
            if check_full_rank:
                if A.rank() >= 2**self.n:
                    # found polynomials
                    self.A = A
                    self.q = q
                    self.t = t
                    return q
                else:
                    cnt += 1
            else:
                try:
                    c = A.solve_right(b)
                    # found polynomials
                    self.A = A
                    self.q = q
                    self.t = t
                    return q
                except ValueError:
                    cnt += 1
        raise ValueError(f'Could not find q polynomials in {tries} tries. Try to increase t or pick different alpha values.')
    
    def polynomial_decomposition(self, sbox):
        """ Returns CRVResult for the decomposition of the specific S-box """
        assert(len(sbox) == 2**self.n), f'Provide complete S-box'
        assert(self.A is not None and self.q is not None and self.t is not None), f'Set q polynomials first: find_q_polynomials(t,tries)'
        b = matrix(self.F, 2**self.n, 1, [_into_F(self.F, x, self.n) for x in sbox])
        c = self.A.solve_right(b)
        p = [[c[i * len(self.L) + j] for j in range(len(self.L))] for i in range(self.t)]
        return CRVResult(self.n, self.t, self.alphas, self.L, self.q, p)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def _do_crv_starmap(arg):
    F, alphas, new_alphas, tries, sbox = arg
    i = CRV(F, alphas + list(new_alphas))
    print(f'Trying {new_alphas}')
    try:
        i.find_q_polynomials(tries=tries, check_full_rank=False, sbox=sbox)
        print(f'Trying {new_alphas}: Found')
        # found one :)
        return i
    except ValueError:
        print(f'Trying {new_alphas}: none')
        return None

def find_min_depth_crv(F, n, sbox, tries=10):
    alphas = [0,1,2,3, 5, 9, 17]
    L = set(x for alpha in alphas for x in _generate_cyclotomic_class(n, alpha))
    round = 1
    instances = []
    while len(instances) <= 0:
        round += 1
        # sum of two values in L may be the next alpha value if they are not in L already
        alpha_candidates = sorted(set((l1 + l2) % (2**n-1) for l1 in L for l2 in L if (l1+l2) % (2**n-1) not in L))
        unique_alpha_candidates = set(alpha_candidates)
        for alpha in alpha_candidates:
            if alpha in unique_alpha_candidates:
                c = _generate_cyclotomic_class(n, alpha)
                for c_a in c:
                    if c_a != alpha and c_a in unique_alpha_candidates:
                        unique_alpha_candidates.remove(c_a)
        # filter for useful cyclotomic classes
        filtered_candidates = []
        for alpha in unique_alpha_candidates:
            c = _generate_cyclotomic_class(n, alpha)
            if len(L & c) < len(c):
                # some new powers are added
                filtered_candidates.append(alpha)
        print(f'round {round}: trying new alphas: {filtered_candidates}')
        for new in powerset(filtered_candidates):
            if len(new) > 0:
                args = (F,alphas,new,tries,sbox)
                inst = _do_crv_starmap(args)
                if inst != None:
                    instances.append(inst)
        print(f'Instances: {instances}')
        alphas += filtered_candidates
    return instances