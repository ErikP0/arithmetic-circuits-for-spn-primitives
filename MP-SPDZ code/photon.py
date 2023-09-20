from Compiler.types import cgf2n
from Compiler.library import get_program
import functools
import operator

# we use the embedding code from AES for P_288
from Programs.Source.aes import Aes128

class VectorConstant:
    def __init__(self, f):
        self.v = dict()
        self.f = f
    def get_type(self, n):
        tape = get_program().curr_tape
        if (n,tape.name) not in self.v:
            self.v[(n,tape.name)] = self.f(n)
        return self.v[(n,tape.name)]

CEMBED_POWERS4 = [1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 30, 1 << 35]
CEMBED_POWERS8 = [1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 25, 1 << 30, 1 << 35]
EMBED_POWERS4 = VectorConstant(lambda n: [cgf2n(p, size=n) for p in CEMBED_POWERS4])
EMBED_POWERS8 = VectorConstant(lambda n: [cgf2n(p, size=n) for p in CEMBED_POWERS8])

def cembed4(x):
    x0,x1,x2,x3 = [(x >> i) & 0x1 for i in range(4)]
    y0 = x0
    y5 = x2 ^ x3
    y10 = x2
    y15 = x1 ^ x2 ^ x3
    y20 = x1
    y30 = x2
    y35 = x1
    
    return y0 ^ sum(p*y for p,y in zip([y5,y10,y15,y20,y30,y35], CEMBED_POWERS4))

def embed4(x):
    assert all(xi.size == x[0].size for xi in x)
    x0,x1,x2,x3 = x
    y0 = x0
    y5 = x2 + x3
    y10 = x2
    y15 = x1 + x2 + x3
    y20 = x1
    y30 = x2
    y35 = x1
    
    return y0 + sum(p*y for p,y in zip([y5,y10,y15,y20,y30,y35], EMBED_POWERS4.get_type(x0.size)))

def bit_decompose4(x):
    y0, y5, y10, y15 = x.bit_decompose(bit_length=20, step=5)
    x0 = y0
    x1 = y5 + y15
    x2 = y10
    x3 = y5 + y10
    return [x0,x1,x2,x3]

def square4(x):
    b0,b1,b2,b3 = x
    s0 = b0 + b2
    s1 = b2
    s2 = b1 + b3
    s3 = b3
    return [s0,s1,s2,s3]

def cembed8(x):
    in_bytes = [(x >> i) & 0x1 for i in range(8)]
    out_bytes = [None] * 8
    out_bytes[0] = functools.reduce(operator.xor, in_bytes[0:8])
    out_bytes[1] = functools.reduce(operator.xor, (in_bytes[idx] for idx in range(1, 8, 2)))
    out_bytes[2] = in_bytes[2] ^ in_bytes[3] ^ in_bytes[6] ^ in_bytes[7]
    out_bytes[3] = in_bytes[3] ^ in_bytes[7]
    out_bytes[4] = in_bytes[4] ^ in_bytes[5] ^ in_bytes[6] ^ in_bytes[7]
    out_bytes[5] = in_bytes[5] ^ in_bytes[7]
    out_bytes[6] = in_bytes[6] ^ in_bytes[7]
    out_bytes[7] = in_bytes[7]
    return out_bytes[0] + sum(p*y for p,y in zip(CEMBED_POWERS8, out_bytes[1:]))

def embed8(self, in_bytes):
    assert all(xi.size == x[0].size for xi in in_bytes)
    out_bytes = [None] * 8
    out_bytes[0] = sum(in_bytes[0:8])
    out_bytes[1] = sum(in_bytes[idx] for idx in range(1, 8, 2))
    out_bytes[2] = in_bytes[2] + in_bytes[3] + in_bytes[6] + in_bytes[7]
    out_bytes[3] = in_bytes[3] + in_bytes[7]
    out_bytes[4] = in_bytes[4] + in_bytes[5] + in_bytes[6] + in_bytes[7]
    out_bytes[5] = in_bytes[5] + in_bytes[7]
    out_bytes[6] = in_bytes[6] + in_bytes[7]
    out_bytes[7] = in_bytes[7]
    return out_bytes[0] + sum(p*y for p,y in zip(EMBED_POWERS8.get_type(in_bytes[0].size), out_bytes[1:]))

PLAIN_RC = [1, 3, 7, 14, 13, 11, 6, 12, 9, 2, 5, 10]
RC4 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in PLAIN_RC])
RC8 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in PLAIN_RC])

PLAIN_IC = {
    5: [0, 1, 3, 6, 4],
    6: [0, 1, 3, 7, 6, 4]
}

IC4 = {
    5: VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in PLAIN_IC[5]])
}
IC8 = {
    6: VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in PLAIN_IC[6]])
}
A_5 = [
    [1,2,9,9,2],
    [2,5,3,8,13],
    [13,11,10,12,1],
    [1,15,2,3,14],
    [14,14,8,5,12]
]
A_288 = [
    [2,3,1,2,1,4],
    [8, 14, 7, 9, 6, 17],
    [34, 59, 31, 37, 24, 66],
    [132, 228, 121, 155, 103, 11],
    [22, 153, 239, 111, 144, 75],
    [150, 203, 210, 121, 36, 167]
]
PLAIN_MIX_MATRIX4 = {
    5: A_5
}
PLAIN_MIX_MATRIX8 = {
    6: A_288
}
MIX_MATRIX4 = {
    5: VectorConstant(lambda n: [[cgf2n(cembed4(x), size=n) for x in row] for row in A_5])
}

MIX_MATRIX8 = {
    6: VectorConstant(lambda n: [[cgf2n(cembed8(x), size=n) for x in row] for row in A_288])
}

# CRV polynomials
Q4 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x9,0xf,0x6,0xb,0x7,0x6,0xc,0x8,0xa]])
P4_1 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0xa,0x2,0x0,0xe,0x1,0x1,0x4,0x0,0x3]])
P4_2 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x9,0xb,0x5,0xe,0x4,0x9,0x5,0x0,0x0]])

def compute_polynomial(coeffs, xpowers):
    assert len(coeffs) == len(xpowers) + 1, f'{len(coeffs)} == {len(xpowers) + 1}'
    return coeffs[0] + sum([c * x for c, x in zip(coeffs[1:], xpowers)])

def _s4_sbox_crv(cell):
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = cell * x2
    
    # round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    q0 = compute_polynomial(Q4.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    p0 = compute_polynomial(P4_1.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    p1 = compute_polynomial(P4_2.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    # round 4
    return p0*q0 + p1

def _s4_sbox_bin(cell, ONE):
    x0,x1,x2,x3 = cell

    f3 = x1 + x2
    tmp = x1 * f3
    f1 = x3 + tmp
    
    f3 = (f3 + f1) + ONE
    f2 = x0 + ONE
    tmp = f1 * f3
    f0 = x1 + tmp
    
    tmp = f0 * f2
    f1 = f1 + tmp
    f0 = f0 + f2
    f2 = f2 + f3
    f0 = f0 + f1
    tmp = f0 * f1
    f3 = f3 + tmp
    return [f2,f1,f3,f0]

class Photon:
    PHOTON_100 = (12, 5, 4)
    PHOTON_288 = (12, 6, 8)
    def __init__(self, variant, simd):
        rounds, d, cellsize = variant
        self.rounds = rounds
        self.d = d
        self.simd = simd
        self.cellsize = cellsize
        if self.cellsize == 8:
            self.aes = Aes128(simd)
    
    def _add_constant(self, state, r):
        if self.cellsize == 4:
            rc = RC4.get_type(self.simd)
            irc = IC4[self.d].get_type(self.simd)
        elif self.cellsize == 8:
            rc = RC8.get_type(self.simd)
            irc = IC8[self.d].get_type(self.simd)
        for i in range(self.d):
            state[i][0] += rc[r]
            state[i][0] += irc[i]
    
    def _sub_cells(self, state):
        for i in range(self.d):
            for j in range(self.d):
                if self.cellsize == 4:
                    state[i][j] = _s4_sbox_crv(state[i][j])
                elif self.cellsize == 8:
                    state[i][j] = self.aes.box.forward_bit_sbox(state[i][j])
    
    def _shift_rows(self, state):
        for i in range(self.d):
            state[i] = list(state[i][i:self.d]) + list(state[i][:i])
    
    def _mix_columns_serial(self, state):
        if self.cellsize == 4:
            A = MIX_MATRIX4[self.d].get_type(self.simd)
        elif self.cellsize == 8:
            A = MIX_MATRIX8[self.d].get_type(self.simd)
        new_state = [[None] * self.d for _ in range(self.d)]
        for i in range(self.d):
            for j in range(self.d):
                column = [state[k][j] for k in range(self.d)]
                new_state[i][j] = sum(ai*sj for ai,sj in zip(A[i], column))
        return new_state
    
    def forward(self, block):
        assert len(block) == self.d
        assert all(len(row) == self.d for row in block)
        assert all(cell.size == self.simd for row in block for cell in row)
        state = [list(row) for row in block]
        
        for r in range(self.rounds):
            self._add_constant(state, r)
            self._sub_cells(state)
            self._shift_rows(state)
            state = self._mix_columns_serial(state)
        return state

def test():
    from Compiler.library import print_ln
    from Compiler.types import sgf2n
    cipher = Photon(Photon.PHOTON_100, 1)
    message = [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,1],
        [4,1,4,1,0]
    ]
    message = [[sgf2n(cembed4(0),size=1) for x in row] for row in message]
    output = cipher.forward(message)
    output = [[sum(2**i*b for i,b in enumerate(bit_decompose4(x))).reveal() for x in row] for row in output]
    for i in range(5):
        print_ln('%s ' * 5, *output[i])
#test()

def mat_mul(M, x):
    res = [0] * len(M)
    for i in range(len(M)):
        res[i] = sum([x[j] for j in range(len(M[i])) if M[i][j] == 1])
    return res

U = [
    [0,0,0,0,0,0,0,1],
    [0,1,1,0,0,0,0,1],
    [1,1,1,0,0,0,0,1],
    [1,1,1,0,0,1,1,1],
    [0,1,1,1,0,0,0,1],
    [0,1,1,0,0,0,1,1],
    [1,0,0,1,1,0,1,1],
    [0,1,0,0,1,1,1,1],
    [1,0,0,0,0,1,0,0],
    [1,0,0,1,0,0,0,0],
    [1,1,1,1,1,0,1,0],
    [0,1,0,0,1,1,1,0],
    [1,0,0,1,0,1,1,0],
    [1,0,0,0,0,0,1,0],
    [0,0,0,1,0,1,0,0],
    [1,0,0,1,1,0,1,0],
    [0,0,1,0,1,1,1,0],
    [1,0,1,1,0,1,0,0],
    [1,0,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [1,1,0,1,1,1,1,0],
    [1,0,1,0,1,1,0,0]
]

B = [
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
]

def aes_sbox(x, ONE):
    assert len(x) == 8
    
    y = mat_mul(U, x[::-1])
    t2 = y[12] * y[15]
    t3 = y[3] * y[6]
    t4 = t2 + t3
    t5 = y[4] * y[0]
    t6 = t2 + t5
    t7 = y[13] * y[16]
    t8 = y[5] * y[1]
    t9 = t7 + t8
    t10 = y[2] * y[7]
    t11 = t7 + t10
    t12 = y[9] * y[11]
    t13 = y[14] * y[17]
    t14 = t12 + t13
    t15 = y[8] * y[10]
    t16 = t12 + t15
    t17 = t4 + t14
    t18 = t6 + t16
    t19 = t9 + t14
    t20 = t11 + t16
    t21 = t17 + y[20]
    t22 = t18 + y[19]
    t23 = t19 + y[21]
    t24 = t20 + y[18]
    
    t25 = t21 + t22
    t26 = t21 * t23
    t27 = t24 + t26
    t28 = t25 * t27
    t29 = t28 + t22
    t30 = t23 + t24
    t31 = t22 + t26
    t32 = t31 * t30
    t33 = t32 + t24
    t34 = t23 + t33
    t35 = t27 + t33
    t36 = t24 * t35
    t37 = t36 + t34
    t38 = t27 + t36
    t39 = t29 * t38
    t40 = t25 + t39
    
    t41 = t40 + t37
    t42 = t29 + t33
    t43 = t29 + t40
    t44 = t33 + t37
    t45 = t42 + t41
    z0 = t44 * y[15]
    z1 = t37 * y[6]
    z2 = t33 * y[0]
    z3 = t43 * y[16]
    z4 = t40 * y[1]
    z5 = t29 * y[7]
    z6 = t42 * y[11]
    z7 = t45 * y[17]
    z8 = t41 * y[10]
    z9 = t44 * y[12]
    z10 = t37 * y[3]
    z11 = t33 * y[4]
    z12 = t43 * y[13]
    z13 = t40 * y[5]
    z14 = t29 * y[2]
    z15 = t42 * y[9]
    z16 = t45 * y[14]
    z17 = t41 * y[8]
    
    z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17]
    s = mat_mul(B, z)
    return [ONE + s[7], ONE + s[6], s[5], s[4], s[3], ONE + s[2], ONE + s[1], s[0]]

class PhotonBin:
    PHOTON_100 = (12, 5, 4)
    PHOTON_288 = (12, 6, 8)
    def __init__(self, variant, simd):
        rounds, d, cellsize = variant
        self.rounds = rounds
        self.d = d
        self.cellsize = cellsize
        self.simd = simd
        self.ONE = cgf2n(1, size=self.simd)
    
    def _add_constant(self, state, r):
        rc = PLAIN_RC
        irc = PLAIN_IC[self.d]
        for i in range(self.d):
            state[i][0] = [b ^ self.ONE if c == 1 else b for c,b in zip([((rc[r] ^ irc[i]) >> j) & 0x1 for j in range(self.cellsize)], state[i][0])]
    
    def _sub_cells(self, state):
        for i in range(self.d):
            for j in range(self.d):
                if self.cellsize == 4:
                    state[i][j] = _s4_sbox_bin(state[i][j], self.ONE)
                elif self.cellsize == 8:
                    state[i][j] = aes_sbox(state[i][j], self.ONE)
    
    def _shift_rows(self, state):
        for i in range(self.d):
            state[i] = list(state[i][i:self.d]) + list(state[i][:i])
    
    def _cmul4(cell, constant):
        x0,x1,x2,x3 = cell
        if constant == 1:
            return list(cell)
        elif constant == 2: 
            y0 = x3
            y1 = x0 + x3
            y2 = x1
            y3 = x2
        elif constant == 3: 
            y0 = x0 + x3
            y1 = x0 + x1 + x3
            y2 = x1 + x2
            y3 = x2 + x3
        elif constant == 5: 
            y0 = x0 + x2
            y1 = x1 + x2 + x3
            y2 = x0 + x2 + x3
            y3 = x1 + x3
        elif constant == 8: 
            y0 = x1
            y1 = x1 + x2
            y2 = x2 + x3
            y3 = x0 + x3
        elif constant == 9: 
            y0 = x0 + x1
            y1 = x2
            y2 = x3
            y3 = x0
        elif constant == 10: 
            y0 = x1 + x3
            y1 = x0 + x1 + x2 + x3
            y2 = x1 + x2 + x3
            y3 = x0 + x2 + x3
        elif constant == 11: 
            y0 = x0 + x1 + x3
            y1 = x0 + x2 + x3
            y2 = x1 + x3
            y3 = x0 + x2
        elif constant == 12: 
            y0 = x1 + x2
            y1 = x1 + x3
            y2 = x0 + x2
            y3 = x0 + x1 + x3
        elif constant == 13: 
            y0 = x0 + x1 + x2
            y1 = x3
            y2 = x0
            y3 = x0 + x1
        elif constant == 14: 
            y0 = x1 + x2 + x3
            y1 = x0 + x1
            y2 = x0 + x1 + x2
            y3 = x0 + x1 + x2 + x3
        elif constant == 15: 
            y0 = x0 + x1 + x2 + x3
            y1 = x0
            y2 = x0 + x1
            y3 = x0 + x1 + x2
        else:
            assert False
        return [y0,y1,y2,y3]
    
    def _cmul8(cell, constant):
        x0,x1,x2,x3,x4,x5,x6,x7 = cell
        if constant == 1: 
            return list(cell)
        elif constant == 2: 
            y0 = x7
            y1 = x0 + x7
            y2 = x1
            y3 = x2 + x7
            y4 = x3 + x7
            y5 = x4
            y6 = x5
            y7 = x6
        elif constant == 3: 
            y0 = x0 + x7
            y1 = x0 + x1 + x7
            y2 = x1 + x2
            y3 = x2 + x3 + x7
            y4 = x3 + x4 + x7
            y5 = x4 + x5
            y6 = x5 + x6
            y7 = x6 + x7
        elif constant == 4: 
            y0 = x6
            y1 = x6 + x7
            y2 = x0 + x7
            y3 = x1 + x6
            y4 = x2 + x6 + x7
            y5 = x3 + x7
            y6 = x4
            y7 = x5
        elif constant == 6: 
            y0 = x6 + x7
            y1 = x0 + x6
            y2 = x0 + x1 + x7
            y3 = x1 + x2 + x6 + x7
            y4 = x2 + x3 + x6
            y5 = x3 + x4 + x7
            y6 = x4 + x5
            y7 = x5 + x6
        elif constant == 7: 
            y0 = x0 + x6 + x7
            y1 = x0 + x1 + x6
            y2 = x0 + x1 + x2 + x7
            y3 = x1 + x2 + x3 + x6 + x7
            y4 = x2 + x3 + x4 + x6
            y5 = x3 + x4 + x5 + x7
            y6 = x4 + x5 + x6
            y7 = x5 + x6 + x7
        elif constant == 8: 
            y0 = x5
            y1 = x5 + x6
            y2 = x6 + x7
            y3 = x0 + x5 + x7
            y4 = x1 + x5 + x6
            y5 = x2 + x6 + x7
            y6 = x3 + x7
            y7 = x4
        elif constant == 9: 
            y0 = x0 + x5
            y1 = x1 + x5 + x6
            y2 = x2 + x6 + x7
            y3 = x0 + x3 + x5 + x7
            y4 = x1 + x4 + x5 + x6
            y5 = x2 + x5 + x6 + x7
            y6 = x3 + x6 + x7
            y7 = x4 + x7
        elif constant == 11: 
            y0 = x0 + x5 + x7
            y1 = x0 + x1 + x5 + x6 + x7
            y2 = x1 + x2 + x6 + x7
            y3 = x0 + x2 + x3 + x5
            y4 = x1 + x3 + x4 + x5 + x6 + x7
            y5 = x2 + x4 + x5 + x6 + x7
            y6 = x3 + x5 + x6 + x7
            y7 = x4 + x6 + x7
        elif constant == 14: 
            y0 = x5 + x6 + x7
            y1 = x0 + x5
            y2 = x0 + x1 + x6
            y3 = x0 + x1 + x2 + x5 + x6
            y4 = x1 + x2 + x3 + x5
            y5 = x2 + x3 + x4 + x6
            y6 = x3 + x4 + x5 + x7
            y7 = x4 + x5 + x6
        elif constant == 17: 
            y0 = x0 + x4
            y1 = x1 + x4 + x5
            y2 = x2 + x5 + x6
            y3 = x3 + x4 + x6 + x7
            y4 = x0 + x5 + x7
            y5 = x1 + x6
            y6 = x2 + x7
            y7 = x3
        elif constant == 22: 
            y0 = x4 + x6 + x7
            y1 = x0 + x4 + x5 + x6
            y2 = x0 + x1 + x5 + x6 + x7
            y3 = x1 + x2 + x4
            y4 = x0 + x2 + x3 + x4 + x5 + x6 + x7
            y5 = x1 + x3 + x4 + x5 + x6 + x7
            y6 = x2 + x4 + x5 + x6 + x7
            y7 = x3 + x5 + x6 + x7
        elif constant == 24: 
            y0 = x4 + x5
            y1 = x4 + x6
            y2 = x5 + x7
            y3 = x0 + x4 + x5 + x6
            y4 = x0 + x1 + x4 + x6 + x7
            y5 = x1 + x2 + x5 + x7
            y6 = x2 + x3 + x6
            y7 = x3 + x4 + x7
        elif constant == 31: 
            y0 = x0 + x4 + x5 + x6 + x7
            y1 = x0 + x1 + x4
            y2 = x0 + x1 + x2 + x5
            y3 = x0 + x1 + x2 + x3 + x4 + x5 + x7
            y4 = x0 + x1 + x2 + x3 + x7
            y5 = x1 + x2 + x3 + x4
            y6 = x2 + x3 + x4 + x5
            y7 = x3 + x4 + x5 + x6
        elif constant == 34: 
            y0 = x3
            y1 = x0 + x3 + x4
            y2 = x1 + x4 + x5
            y3 = x2 + x3 + x5 + x6
            y4 = x4 + x6 + x7
            y5 = x0 + x5 + x7
            y6 = x1 + x6
            y7 = x2 + x7
        elif constant == 36: 
            y0 = x3 + x6 + x7
            y1 = x3 + x4 + x6
            y2 = x0 + x4 + x5 + x7
            y3 = x1 + x3 + x5 + x7
            y4 = x2 + x3 + x4 + x7
            y5 = x0 + x3 + x4 + x5
            y6 = x1 + x4 + x5 + x6
            y7 = x2 + x5 + x6 + x7
        elif constant == 37: 
            y0 = x0 + x3 + x6 + x7
            y1 = x1 + x3 + x4 + x6
            y2 = x0 + x2 + x4 + x5 + x7
            y3 = x1 + x5 + x7
            y4 = x2 + x3 + x7
            y5 = x0 + x3 + x4
            y6 = x1 + x4 + x5
            y7 = x2 + x5 + x6
        elif constant == 59: 
            y0 = x0 + x3 + x4 + x5
            y1 = x0 + x1 + x3 + x6
            y2 = x1 + x2 + x4 + x7
            y3 = x0 + x2 + x4
            y4 = x0 + x1 + x4
            y5 = x0 + x1 + x2 + x5
            y6 = x1 + x2 + x3 + x6
            y7 = x2 + x3 + x4 + x7
        elif constant == 66: 
            y0 = x2 + x6
            y1 = x0 + x2 + x3 + x6 + x7
            y2 = x1 + x3 + x4 + x7
            y3 = x4 + x5 + x6
            y4 = x2 + x5 + x7
            y5 = x3 + x6
            y6 = x0 + x4 + x7
            y7 = x1 + x5
        elif constant == 75: 
            y0 = x0 + x2 + x5 + x6
            y1 = x0 + x1 + x2 + x3 + x5 + x7
            y2 = x1 + x2 + x3 + x4 + x6
            y3 = x0 + x3 + x4 + x6 + x7
            y4 = x1 + x2 + x4 + x6 + x7
            y5 = x2 + x3 + x5 + x7
            y6 = x0 + x3 + x4 + x6
            y7 = x1 + x4 + x5 + x7
        elif constant == 103: 
            y0 = x0 + x2 + x3 + x7
            y1 = x0 + x1 + x2 + x4 + x7
            y2 = x0 + x1 + x2 + x3 + x5
            y3 = x1 + x4 + x6 + x7
            y4 = x3 + x5
            y5 = x0 + x4 + x6
            y6 = x0 + x1 + x5 + x7
            y7 = x1 + x2 + x6
        elif constant == 111: 
            y0 = x0 + x2 + x3 + x5 + x7
            y1 = x0 + x1 + x2 + x4 + x5 + x6 + x7
            y2 = x0 + x1 + x2 + x3 + x5 + x6 + x7
            y3 = x0 + x1 + x4 + x5 + x6
            y4 = x1 + x3 + x6
            y5 = x0 + x2 + x4 + x7
            y6 = x0 + x1 + x3 + x5
            y7 = x1 + x2 + x4 + x6
        elif constant == 121: 
            y0 = x0 + x2 + x3 + x4 + x5 + x6
            y1 = x1 + x2 + x7
            y2 = x2 + x3
            y3 = x0 + x2 + x5 + x6
            y4 = x0 + x1 + x2 + x4 + x5 + x7
            y5 = x0 + x1 + x2 + x3 + x5 + x6
            y6 = x0 + x1 + x2 + x3 + x4 + x6 + x7
            y7 = x1 + x2 + x3 + x4 + x5 + x7
        elif constant == 132: 
            y0 = x1 + x5
            y1 = x1 + x2 + x5 + x6
            y2 = x0 + x2 + x3 + x6 + x7
            y3 = x3 + x4 + x5 + x7
            y4 = x1 + x4 + x6
            y5 = x2 + x5 + x7
            y6 = x3 + x6
            y7 = x0 + x4 + x7
        elif constant == 144: 
            y0 = x1 + x4 + x5 + x6
            y1 = x1 + x2 + x4 + x7
            y2 = x2 + x3 + x5
            y3 = x1 + x3 + x5
            y4 = x0 + x1 + x2 + x5
            y5 = x1 + x2 + x3 + x6
            y6 = x2 + x3 + x4 + x7
            y7 = x0 + x3 + x4 + x5
        elif constant == 150: 
            y0 = x1 + x4 + x5 + x7
            y1 = x0 + x1 + x2 + x4 + x6 + x7
            y2 = x0 + x1 + x2 + x3 + x5 + x7
            y3 = x2 + x3 + x5 + x6 + x7
            y4 = x0 + x1 + x3 + x5 + x6
            y5 = x1 + x2 + x4 + x6 + x7
            y6 = x2 + x3 + x5 + x7
            y7 = x0 + x3 + x4 + x6
        elif constant == 153: 
            y0 = x0 + x1 + x4 + x6
            y1 = x2 + x4 + x5 + x6 + x7
            y2 = x3 + x5 + x6 + x7
            y3 = x0 + x1 + x7
            y4 = x0 + x2 + x4 + x6
            y5 = x1 + x3 + x5 + x7
            y6 = x2 + x4 + x6
            y7 = x0 + x3 + x5 + x7
        elif constant == 155: 
            y0 = x0 + x1 + x4 + x6 + x7
            y1 = x0 + x2 + x4 + x5 + x6
            y2 = x1 + x3 + x5 + x6 + x7
            y3 = x0 + x1 + x2
            y4 = x0 + x2 + x3 + x4 + x6 + x7
            y5 = x1 + x3 + x4 + x5 + x7
            y6 = x2 + x4 + x5 + x6
            y7 = x0 + x3 + x5 + x6 + x7
        elif constant == 167: 
            y0 = x0 + x1 + x3 + x5
            y1 = x0 + x2 + x3 + x4 + x5 + x6
            y2 = x0 + x1 + x3 + x4 + x5 + x6 + x7
            y3 = x2 + x3 + x4 + x6 + x7
            y4 = x1 + x4 + x7
            y5 = x0 + x2 + x5
            y6 = x1 + x3 + x6
            y7 = x0 + x2 + x4 + x7
        elif constant == 203: 
            y0 = x0 + x1 + x2
            y1 = x0 + x3
            y2 = x1 + x4
            y3 = x0 + x1 + x5
            y4 = x6
            y5 = x7
            y6 = x0
            y7 = x0 + x1
        elif constant == 210: 
            y0 = x1 + x2 + x4 + x5
            y1 = x0 + x1 + x3 + x4 + x6
            y2 = x1 + x2 + x4 + x5 + x7
            y3 = x1 + x3 + x4 + x6
            y4 = x0 + x1 + x7
            y5 = x1 + x2
            y6 = x0 + x2 + x3
            y7 = x0 + x1 + x3 + x4
        elif constant == 228: 
            y0 = x1 + x2 + x3 + x5 + x6
            y1 = x1 + x4 + x5 + x7
            y2 = x0 + x2 + x5 + x6
            y3 = x2 + x5 + x7
            y4 = x1 + x2 + x5
            y5 = x0 + x2 + x3 + x6
            y6 = x0 + x1 + x3 + x4 + x7
            y7 = x0 + x1 + x2 + x4 + x5
        elif constant == 239: 
            y0 = x0 + x1 + x2 + x3 + x6 + x7
            y1 = x0 + x4 + x6
            y2 = x0 + x1 + x5 + x7
            y3 = x0 + x3 + x7
            y4 = x2 + x3 + x4 + x6 + x7
            y5 = x0 + x3 + x4 + x5 + x7
            y6 = x0 + x1 + x4 + x5 + x6
            y7 = x0 + x1 + x2 + x5 + x6 + x7
        else:
            assert False
        return [y0,y1,y2,y3,y4,y5,y6,y7]
    
    def _mix_columns_serial(self, state):
        if self.cellsize == 4:
            A = PLAIN_MIX_MATRIX4[self.d]
            cmul = PhotonBin._cmul4
        elif self.cellsize == 8:
            A = PLAIN_MIX_MATRIX8[self.d]
            cmul = PhotonBin._cmul8
        new_state = [[None] * self.d for _ in range(self.d)]
        for i in range(self.d):
            for j in range(self.d):
                column = [state[k][j] for k in range(self.d)]
                parts = [cmul(sj, ai) for ai,sj in zip(A[i], column)]
                new_state[i][j] = functools.reduce(lambda acc,x: [acci + xi for acci,xi in zip(acc,x)], parts[1:], parts[0])
        return new_state
    
    def forward(self, block):
        assert len(block) == self.d
        assert all(len(row) == self.d for row in block)
        assert all(len(cell) == self.cellsize for row in block for cell in row)
        assert all(b.size == self.simd for row in block for cell in row for b in cell)
        state = [list(row) for row in block]
        
        for r in range(self.rounds):
            self._add_constant(state, r)
            self._sub_cells(state)
            self._shift_rows(state)
            state = self._mix_columns_serial(state)
        return state