from Compiler.types import sgf2n, cgf2n, Matrix, regint
from Compiler.library import get_program, print_ln

import math

def skinny_64_128_enc(message, tweakey_schedule):
    '''
    Computes SKINNY-64-128 forward direction.
    message: message block with dimension 16; each cell is embedded via embed4
    tweakey_schedule: list of round keys obtained from expand_key

    Returns
        ciphertext with dimension 16; each cell is embedded via embed4
    '''
    assert(len(message) == 16)
    assert all((message[i].size == message[0].size for i in range(16)))
    return SkinnyGF2n(SkinnyGF2n.SKINNY_64_128, vector_size=message[0].size).skinny_enc(message, tweakey_schedule)

def skinny_64_128_dec(ciphertext, tweakey_schedule):
    '''
    Computes SKINNY-64-128 inverse direction.
    ciphertext: ciphertext block with dimension 16; each cell is embedded via embed4
    tweakey_schedule: list of round keys obtained from expand_key

    Returns
        message with dimension 16; each cell is embedded via embed4
    '''
    assert(len(ciphertext) == 16), f'{len(ciphertext)}: {ciphertext}'
    assert all((ciphertext[i].size == ciphertext[0].size for i in range(16)))
    return SkinnyGF2n(SkinnyGF2n.SKINNY_64_128, vector_size=ciphertext[0].size).skinny_dec(ciphertext, tweakey_schedule)

def skinny_128_256_enc(message, tweakey_schedule):
    '''
    Computes SKINNY-128-256 forward direction.
    message: message block with dimension 16; each cell is embedded via embed8
    tweakey_schedule: list of round keys obtained from expand_key

    Returns
        ciphertext with dimension 16; each cell is embedded via embed8
    '''
    assert(len(message) == 16)
    assert all((message[i].size == message[0].size for i in range(16)))
    return SkinnyGF2n(SkinnyGF2n.SKINNY_128_256, vector_size=message[0].size).skinny_enc(message, tweakey_schedule)

def skinny_128_256_dec(ciphertext, tweakey_schedule):
    '''
    Computes SKINNY-128-256 inverse direction.
    ciphertext: ciphertext block with dimension 16; each cell is embedded via skinny_embed8
    tweakey_schedule: list of round keys obtained from expand_key

    Returns
        message with dimension 16; each cell is embedded via skinny_embed8
    '''
    assert(len(ciphertext) == 16), f'{len(ciphertext)}: {ciphertext}'
    assert all((ciphertext[i].size == ciphertext[0].size for i in range(16)))
    return SkinnyGF2n(SkinnyGF2n.SKINNY_128_256, vector_size=ciphertext[0].size).skinny_dec(ciphertext, tweakey_schedule)

def expand_key(key, tweak, variant):
    '''
    Computes and returns the key schedule of SKINNY
    key:        list of key bits encoded in groups of cellsize bits with least significant bit first
    tweak:      list of tweak bits encoded in groups of cellsize bits with least significant bit first
    variant:    SkinnyGF2n.SKINNY_128_256, SkinnyGF2n.SKINNY_64_128

    Returns a list of round keys
    '''
    if variant == SkinnyGF2n.SKINNY_128_256:
        assert len(key) + len(tweak) == 256
    elif variant == SkinnyGF2n.SKINNY_64_128:
        assert len(key) + len(tweak) == 128
    else:
        raise NotImplemented
    fk = SkinnyGF2n(variant, vector_size=key[0].size)
    return fk.expand_key(key, tweak)

class SkinnyBase:
    ROUND_KEY_PERMUTATION = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    SHIFT_ROWS_PERMUTATION = [0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12]
    SHIFT_ROWS_PERMUTATION_INV = [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]
    MIX_COLUMNS_MATRIX = [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
    MIX_COLUMNS_MATRIX_INV = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    ROUND_CONSTANTS = [
        0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
        0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
        0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
        0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A
    ]
    SKINNY_128_256 = (8, 48, 2)
    SKINNY_64_128 = (4, 36, 2)
    def __init__(self, variant):
        cellsize, rounds, tk = variant
        assert cellsize in [4,8]
        self.cellsize = cellsize
        self.rounds = rounds
        self.tk = tk
    
    def expand_key(self, key, tweak):
        return self.skinny_expand_key(key, tweak, self.rounds)
    
    def skinny_enc(self, state, tweakey_schedule):
        assert len(tweakey_schedule) == self.rounds, f'{len(tweakey_schedule)} != {self.rounds}'
        assert all((len(tk) == self.tk for tk in tweakey_schedule))
        if self.cellsize == 4:
            sbox = self.s4_sbox
        elif self.cellsize == 8:
            sbox = self.s8_sbox
        else:
            raise NotImplemented
        for r in range(self.rounds):
            state = self.skinny_round_enc(state, tweakey_schedule[r], r, sbox, False)
        return state
    
    def skinny_dec(self, state, tweakey_schedule):
        assert len(tweakey_schedule) == self.rounds, f'{len(tweakey_schedule)} != {self.rounds}'
        assert all((len(tk) == self.tk for tk in tweakey_schedule))
        if self.cellsize == 4:
            sbox = self.s4_sbox_inv
        elif self.cellsize == 8:
            sbox = self.s8_sbox_inv
        else:
            raise NotImplemented
        for r in reversed(range(self.rounds)):
            state = self.skinny_round_dec(state, tweakey_schedule[r], r, sbox, False)
        return state
    
    def _xor_cell(self, a, b):
        raise NotImplemented
    
    def sub_cells(self, state, sbox):
        assert(len(state) == 16)
        for i in range(len(state)):
            state[i] = sbox(state[i])
        return state

    def add_round_constants(self, state, r, has_tweak):
        raise NotImplemented
    
    def s4_sbox(self, cell):
        raise NotImplemented
    def s8_sbox(self, cell):
        raise NotImplemented
    def s4_sbox_inv(self, cell):
        raise NotImplemented
    def s8_sbox_inv(self, cell):
        raise NotImplemented
    
    def add_round_key(self, state, tk):
        assert(len(state) == 16)
        # xor the first two rows
        for i in range(8):
            for j in range(len(tk)):
                state[i] = self._xor_cell(state[i], tk[j][i])
        return state
    
    def shift_rows(self, state, permutation):
        assert(len(state) == 16)
        new_state = [state[permutation[i]] for i in range(len(state))]
        return new_state
    
    def mix_columns(self, state, matrix):
        new_state = [None] * 16
        for i in range(4):
            for j in range(4):
                to_xor = [state[4*k+j] for k in range(4) if matrix[4*i+k] > 0]
                if len(to_xor) == 1:
                    new_state[4*i+j] = to_xor[0]
                elif len(to_xor) == 2:
                    new_state[4*i+j] = self._xor_cell(to_xor[0], to_xor[1])
                elif len(to_xor) == 3:
                    new_state[4*i+j] = self._xor_cell(to_xor[0], self._xor_cell(to_xor[1], to_xor[2]))
                else:
                    raise NotImplemented
        return new_state
        
    def update_round_key(self, tk):
        new_tk = []
        for ti in tk:
            new_tk.append([ti[self.ROUND_KEY_PERMUTATION[i]] for i in range(len(ti))])
        if len(new_tk) >= 2:
            # apply LFSR2 to the first two rows of TK2
            for i in range(8):
                if self.cellsize == 4:
                    x = new_tk[1][i]
                    new_tk[1][i] = [x[3] + x[2], x[0], x[1], x[2]]
                elif self.cellsize == 8:
                    x = new_tk[1][i]
                    new_tk[1][i] = [x[7] + x[5], x[0], x[1], x[2], x[3], x[4], x[5], x[6]]
                else:
                    raise NotImplemented
        
        if len(new_tk) >= 3:
            # apply LFSR3 to the first two rows of TK3
            for i in range(8):
                if self.cellsize == 4:
                    x = new_tk[2][i]
                    new_tk[2][i] = [x[1], x[2], x[3], x[0] + x[3]]
                elif self.cellsize == 8:
                    x = new_tk[2][i]
                    new_tk[2][i] = [x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[0] + x[6]]
                else:
                    raise NotImplemented
        return new_tk
    
    def skinny_expand_key(self, key, tweak, rounds):
        '''
        Computes and returns the key schedule of (Fork)SKINNY
        key:        list of key bits encoded in groups of cellsize bits with least significant bit first
        tweak:      list of tweak bits encoded in groups of cellsize bits with least significant bit first
        
        Returns a list of round keys
        '''
        bits = key + tweak
        tk = []
        blocksize = 16 * self.cellsize
        n_blocks = int(math.ceil(len(bits)/blocksize))
        assert(len(bits) == blocksize*n_blocks), 'The implementation currently does not support incomplete tweakeys'
        for i in range(n_blocks):
            tki = [[bits[i*blocksize + j * self.cellsize + k] for k in range(self.cellsize)] for j in range(16)]
            tk.append(tki)
        
        schedule = []
        for r in range(rounds):
            schedule.append([[self._embed_cell(cell) for cell in tki] for tki in tk])
            tk = self.update_round_key(tk)
        return schedule

    def skinny_round_enc(self, state, tweakey, r, sbox, has_tweak):
        state = self.sub_cells(state, sbox)
        state = self.add_round_constants(state, r, has_tweak)
        state = self.add_round_key(state, tweakey)
        state = self.shift_rows(state, self.SHIFT_ROWS_PERMUTATION)
        state = self.mix_columns(state, self.MIX_COLUMNS_MATRIX)
        return state
    
    def skinny_round_dec(self, state, tweakey, r, sbox, has_tweak):
        state = self.mix_columns(state, self.MIX_COLUMNS_MATRIX_INV)
        state = self.shift_rows(state, self.SHIFT_ROWS_PERMUTATION_INV)
        state = self.add_round_key(state, tweakey)
        state = self.add_round_constants(state, r, has_tweak)
        state = self.sub_cells(state, sbox)
        return state

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

#RC2_4_EMBEDDED = embed4(0x2)
RC2_4_EMBEDDED = VectorConstant(lambda n: cgf2n(cembed4(0x2), size=n))
RC2_8_EMBEDDED = VectorConstant(lambda n: cgf2n(cembed8(0x2), size=n))

#RC2_8_EMBEDDED = embed8(0x2)
RC2_8_EMBEDDED = VectorConstant(lambda n: cgf2n(0x2000400, size=n))

#ROUND_CONSTANTS8_EMBEDDED = [(embed8(rc & 0xf), embed8((rc >> 4) & 0x3), RC2_8_EMBEDDED) for rc in ROUND_CONSTANTS]
ROUND_CONSTANTS8_EMBEDDED = VectorConstant(lambda n: [(cgf2n(cembed8(rc & 0xf), size=n), cgf2n(cembed8((rc >> 4) & 0x7), size=n), RC2_8_EMBEDDED.get_type(n)) for rc in ForkSkinnyBase.ROUND_CONSTANTS])

#BC4 = [0x1, 0x2, 0x4, 0x9, 0x3, 0x6, 0xd, 0xa, 0x5, 0xb, 0x7, 0xf, 0xe, 0xc, 0x8, 0x1]
BC4 = VectorConstant(lambda n: [cgf2n(cembed4(bc), size=n) for bc in ForkSkinnyBase.BC4])
#BC8 = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x41, 0x82, 0x05, 0x0a, 0x14, 0x28, 0x51, 0xa2, 0x44, 0x88]
BC8 = VectorConstant(lambda n: [cgf2n(cembed8(bc), size=n) for bc in ForkSkinnyBase.BC8])

# CRV polynomials
Q4 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x0,0x6,0x4,0xd,0x3,0x4,0x8,0xb,0x8]])
P4_1 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x1,0xc,0xf,0x1,0x5,0x2,0xb,0xb,0x0]])
P4_2 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0xc,0x3,0x0,0x8,0xe,0xa,0x0,0x9,0x0]])

P4_inv_1 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x7,0x7,0x7,0x9,0x3,0xa,0xc,0xf,0x7]])
P4_inv_2 = VectorConstant(lambda n: [cgf2n(cembed4(x), size=n) for x in [0x3,0xd,0x4,0xf,0x5,0x7,0x0,0x6,0x0]])

Q8_1 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x54,0x48,0x55,0x13,0x34,0x7f,0x41,0xd8,0xc5,0xaf,0xc7,0x30,0x7b,0x20,0x4f,0xf9,0x40,0x3a,0xb9,0x4f,0x53,0x5a,0xcd,0x0,0xc6,0x41,0xa9,0x8e,0xc8,0xf2,0xb7,0x43,0xb,0xb6,0x9f,0xc5,0x12,0x11,0xbb,0x97,0x8]])
Q8_2 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x80,0x65,0x69,0x6a,0x5d,0xc6,0x1b,0x2f,0xf9,0xc8,0xb0,0x57,0xa0,0x97,0x3,0x85,0x60,0x8a,0xa8,0x6a,0xb7,0x7c,0xfb,0x4d,0x24,0x23,0xf2,0x16,0x82,0xbe,0xb3,0x27,0xd,0x80,0x8c,0xaf,0x44,0xeb,0x81,0x99,0x11]])
Q8_3 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x85,0x3,0xb0,0x96,0xa8,0xf3,0x4b,0x6d,0xc1,0x47,0x4f,0x10,0xf2,0x1b,0x25,0x9d,0x56,0x12,0xdf,0xd7,0x97,0x7b,0x4,0xe7,0x1d,0x28,0x63,0x65,0x99,0x38,0x9,0x70,0x3a,0x1a,0xf6,0x39,0xd6,0x28,0x71,0xb,0xb4]])
Q8_4 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xbb,0xd7,0x38,0x85,0xb7,0x5b,0x2c,0x1f,0x1f,0xc2,0xec,0x81,0x70,0xc,0x13,0x71,0xd1,0x8a,0xae,0xf8,0xc8,0x68,0xde,0x3c,0x52,0x2e,0xa3,0x76,0x49,0xd7,0x7d,0xe3,0x11,0xa9,0xc5,0x1e,0xbd,0xc5,0xcf,0xcd,0x77]])
Q8_5 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x48,0x62,0x5f,0x86,0x9,0x96,0xd3,0x72,0xea,0x6,0x75,0x68,0xdf,0x6,0x85,0x3d,0x2f,0xa1,0x63,0x2,0xcd,0x6e,0xd6,0x77,0x1d,0x42,0x53,0x2b,0xac,0x13,0xe3,0x87,0x7d,0x47,0xe6,0x37,0xb0,0x50,0xe8,0x1f,0xe5]])
Q8_6 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xe4,0x4e,0xfc,0x79,0x8b,0xd9,0xe,0xd,0xf5,0xd,0x48,0x3e,0x43,0x24,0x8e,0xca,0x73,0x3e,0x76,0x2f,0x69,0xc8,0x5c,0xd4,0x7e,0x8a,0xb5,0xb5,0x3b,0xaa,0x72,0x93,0x89,0x30,0xd3,0xf9,0x6,0x3e,0x9,0x35,0x6f]])



P8_1 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xc7,0xb5,0xc6,0x18,0xf4,0x1c,0x51,0xdc,0xe6,0xb7,0xa0,0xfe,0xb7,0x73,0xc4,0x1e,0x1,0x76,0x12,0xee,0xbe,0xb1,0xf7,0x1,0x8b,0x4d,0x8e,0xf6,0x24,0x31,0x84,0xb7,0xf0,0x25,0x45,0x2a,0xd0,0x67,0x17,0x6c,0xa7]])
P8_2 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xc5,0x25,0x51,0x82,0x2f,0x28,0x44,0xb3,0xa1,0xc7,0xc9,0x87,0xed,0x3e,0x23,0x46,0x73,0xbf,0x2d,0x66,0x3e,0x5b,0x10,0xca,0x37,0x83,0x8b,0xa5,0xe7,0xcb,0xc,0x28,0xfc,0x13,0x66,0x1c,0x35,0x5a,0x25,0xbb,0x0]])
P8_3 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x16,0x4b,0xf8,0x57,0x0,0x60,0x59,0xca,0xe6,0x90,0x54,0x53,0x5,0x8c,0x91,0x9d,0x2,0x43,0xa3,0xdc,0x5,0x32,0xad,0xe8,0x83,0xb6,0xe5,0x67,0x40,0x3f,0x44,0xe4,0x9f,0xaf,0x91,0x69,0x72,0x2a,0xf2,0x0,0x0]])
P8_4 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x85,0x5f,0x31,0xde,0xbc,0x91,0x53,0x7e,0x87,0xb6,0x72,0x50,0xc9,0xef,0x9f,0x63,0xeb,0x6a,0xae,0xc6,0x85,0xf,0x2a,0x61,0xc8,0xeb,0x7a,0xb7,0x4,0x60,0x23,0x99,0x89,0xce,0x85,0x13,0x16,0xa3,0x0,0x0,0x0]])
P8_5 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xc2,0x22,0x97,0xac,0xea,0x50,0xff,0x2d,0x81,0xc4,0x17,0x1a,0xa2,0x48,0x61,0x18,0xd9,0x47,0x68,0xf3,0x55,0xd6,0x2f,0x71,0x17,0x6f,0xf7,0xb6,0xaf,0xf0,0xdd,0x54,0x5e,0x9,0x70,0x10,0xc9,0x0,0x0,0x0,0x0]])
P8_6 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x1b,0x95,0xc2,0x9d,0x84,0x64,0x87,0xf1,0x2a,0x95,0xce,0x62,0x75,0xff,0x38,0xae,0x15,0x2b,0xce,0xfa,0x6f,0x2a,0xd,0xf6,0x66,0xf2,0x16,0x85,0xf2,0x4e,0xb8,0xd0,0xbb,0xd9,0xf,0x8c,0x0,0x0,0x0,0x0,0x0]])
P8_7 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xd0,0xcb,0xc9,0x84,0x71,0xa8,0xce,0xac,0xd1,0x2c,0x8a,0x44,0x7f,0x96,0x73,0xe8,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0]])

P8_inv_1 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0x5b,0x6f,0x33,0x6e,0xb9,0x3b,0x7c,0xc2,0x76,0x83,0x8a,0xf4,0x0,0x94,0x35,0x4e,0xf1,0xae,0x94,0xa4,0x24,0xa5,0x47,0x8,0x33,0x4c,0x62,0x8,0x95,0xe4,0xef,0x85,0x5c,0x62,0x4b,0x86,0x20,0x8a,0x29,0xf9,0x93]])
P8_inv_2 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xa8,0x97,0xf3,0xce,0x28,0xed,0x93,0xbb,0x3a,0xc2,0xd5,0x11,0xbb,0x66,0x19,0xab,0x77,0xc7,0x2b,0xaa,0x18,0xb4,0x6e,0xa9,0x78,0x20,0x7e,0xf4,0x60,0x4d,0x29,0xa8,0xc1,0xb4,0x71,0xcb,0x2c,0x7,0x25,0x5e,0x0]])
P8_inv_3 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xe5,0xef,0xb2,0xb7,0x94,0x38,0x88,0xdf,0x72,0x74,0xad,0x1c,0x63,0x51,0xea,0xe4,0xa4,0x51,0x62,0x1e,0x6f,0x36,0xe1,0x72,0x6d,0xc1,0x87,0xc5,0xa5,0x44,0x35,0xc6,0xdc,0x64,0xda,0x11,0x5,0xda,0xd4,0x0,0x0]])
P8_inv_4 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xd9,0x43,0xe9,0x1,0xb9,0xfb,0x7c,0xa1,0xc0,0x6e,0x5c,0xb9,0x25,0x1f,0x28,0x95,0x42,0x2b,0xb,0x68,0xea,0xef,0xd8,0x1b,0xc8,0x90,0xca,0x38,0x82,0xb3,0xd9,0x78,0xdb,0xbd,0xcf,0xe,0x18,0x3c,0x0,0x0,0x0]])
P8_inv_5 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xcf,0x89,0x68,0xcb,0x36,0x67,0x1d,0xe6,0x90,0x19,0x3e,0xe5,0x47,0xec,0xe2,0x52,0x6b,0x8a,0xea,0x72,0x98,0x9a,0x40,0x5d,0xef,0xd0,0x91,0x9a,0x6b,0x8f,0x50,0x79,0xd4,0x9e,0xd,0xc0,0x88,0x0,0x0,0x0,0x0]])
P8_inv_6 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xe4,0x7b,0x3b,0xeb,0xfb,0x1f,0x56,0x24,0x3e,0x46,0x99,0xe,0x36,0x96,0x48,0x6d,0x92,0xf8,0x46,0x68,0x4,0xf7,0x4c,0x93,0xea,0x5d,0x99,0x38,0x5e,0x5a,0x3b,0xfb,0x2c,0xf0,0xe0,0xa8,0x0,0x0,0x0,0x0,0x0]])
P8_inv_7 = VectorConstant(lambda n: [cgf2n(cembed8(x), size=n) for x in [0xe9,0xcd,0xae,0x80,0xee,0x37,0x73,0xeb,0xa7,0x40,0xd2,0xdd,0x6,0x7a,0xc3,0x4e,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0]])



# two-step CRV decomposition polynomials for inverse 4-bit Sbox
P00 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x1, 0x800108001, 0x0, 0x40008421, 0x840108401]])
Q00 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x40000401, 0x800100020, 0x840100420, 0x800108001, 0x1]])
P01 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x8020, 0x840108400, 0x800100020, 0x0, 0x0]])
P10 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x840100421, 0x1, 0x840100420, 0x40000401, 0x800100020]])
Q10 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x840108401, 0x8020, 0x40008420, 0x840108400, 0x800100021]])
P11 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x800108000, 0x840100420, 0x800100021, 0x0, 0x0]])

U00 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x840100421, 0x40008421, 0x40008421, 0x840108401, 0x40008421]])
V00 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x800108000, 0x800100021, 0x800100020, 0x800108001, 0x800108000]])
U01 = VectorConstant(lambda n: [cgf2n(x, size=n) for x in [0x840108400, 0x40008421, 0x40008420, 0x840108401, 0x40008421]])


SKINNY_ROUND_CONSTANTS4_EMBEDDED = VectorConstant(lambda n: [(cgf2n(cembed4(rc & 0xf), size=n), cgf2n(cembed4((rc >> 4) & 0x3), size=n), RC2_4_EMBEDDED.get_type(n)) for rc in SkinnyBase.ROUND_CONSTANTS])
SKINNY_ROUND_CONSTANTS8_EMBEDDED = VectorConstant(lambda n: [(cgf2n(cembed8(rc & 0xf), size=n), cgf2n(cembed8((rc >> 4) & 0x3), size=n), RC2_8_EMBEDDED.get_type(n)) for rc in SkinnyBase.ROUND_CONSTANTS])

def embed4(n):
    '''
    Computes embedding F_{2^4} -> F_{2^40} via Y^35 + Y^20 + Y^5 + 1
    F_{2^4} = GF(2)[X]/X^4 + X^3 + 1
    F_{2^40} = GF(2)[Y]/Y^40 + Y^20 + Y^15 + Y^10 + 1
    This embedding requires 3 additions
    '''
    assert len(n) == 4
    b0, b1, b2, b3 = n
    assert [b0.size] * 3 == [b1.size, b2.size, b3.size]
    return (b0 + b1 + b2) + sum((b * x for b,x in zip([b1 + b2, b3, b2, b1, b3, b1], EMBED_POWERS4.get_type(b0.size))))

def cembed4(n):
    assert isinstance(n, int)
    b = [(n >> i) & 0x1 for i in range(4)]
    return (b[0] ^ b[1] ^ b[2]) + sum((b * x for b,x in zip([b[1] ^ b[2], b[3], b[2], b[1], b[3], b[1]], CEMBED_POWERS4)))

def embed8(n):
    '''
    Computes embedding F_{2^8} -> F_{2^40} via Y^35 + Y^30 + Y^25 + Y^20 + Y^10 + Y^5
    F_{2^8} = GF(2)[X]/X^8 + X^7 + X^6 + X^5 + X^4 + X^2 + 1
    F_{2^40} = GF(2)[Y]/Y^40 + Y^20 + Y^15 + Y^10 + 1
    This embedding requires 15 additions
    '''
    if isinstance(n, int):
        b0, b1, b2, b3, b4, b5, b6, b7 = [cgf2n((n >> i) & 0x1) for i in range(8)]
    else:
        b0, b1, b2, b3, b4, b5, b6, b7 = n
    assert [b0.size] * 7 == [b1.size, b2.size, b3.size, b4.size, b5.size, b6.size, b7.size]
    b12 = b1 + b2
    b34 = b3 + b4
    
    y0 = b0 + b2 + b5 + b6 + b7
    y5 = b12 + b34
    y10 = b1 + b6
    y15 = b3 + b6
    y20 = b1 + b3 + b5
    y25 = b12 + b34 + b6
    y30 = b1 + b4 + b7
    y35 = b12

    return y0 + sum((b * x for b,x in zip([y5,y10,y15,y20,y25,y30,y35], EMBED_POWERS8.get_type(b0.size))))

def cembed8(n):
    assert isinstance(n, int)
    b0, b1, b2, b3, b4, b5, b6, b7 = [(n >> i) & 0x1 for i in range(8)]
    b12 = b1 ^ b2
    b34 = b3 ^ b4
    
    y0 = b0 ^ b2 ^ b5 ^ b6 ^ b7
    y5 = b12 ^ b34
    y10 = b1 ^ b6
    y15 = b3 ^ b6
    y20 = b1 ^ b3 ^ b5
    y25 = b12 ^ b34 ^ b6
    y30 = b1 ^ b4 ^ b7
    y35 = b12
    return y0 + sum((b * x for b,x in zip([y5,y10,y15,y20,y25,y30,y35], CEMBED_POWERS8)))

def bit_decompose4(x):
    '''
    Computes the inverse embedding F_{2^4} -> F_{2^40} via Y^35 + Y^20 + Y^5 + 1
    F_{2^4} = GF(2)[X]/X^4 + X^3 + 1
    F_{2^40} = GF(2)[Y]/Y^40 + Y^20 + Y^15 + Y^10 + 1
    This inverse embedding requires 2 additions
    '''
    b0, b5, b10, b15 = x.bit_decompose(bit_length=20, step=5)
    return [b0 + b5, b5 + b15, b15, b10]

def bit_decompose8(x):
    '''
    Computes the inverse embedding F_{2^8} -> F_{2^40} via Y^35 + Y^30 + Y^25 + Y^20 + Y^10 + Y^5
    F_{2^8} = GF(2)[X]/X^8 + X^6 + X^5 + X^4 + X^2 + 1
    F_{2^40} = GF(2)[Y]/Y^40 + Y^20 + Y^15 + Y^10 + 1
    This inverse embedding requires 17 additions
    '''
    b0, b5, b10, b15, b20, b25, b30, b35 = x.bit_decompose(bit_length=40, step=5)
    x0 = b0 + b5 + b10 + b20 + b30
    x1 = b5 + b10 + b25
    x2 = b5 + b10 + b25 + b35
    x3 = b5 + b15 + b25
    x4 = b15 + b25 + b35
    x5 = b10 + b15 + b20
    x6 = b5 + b25
    x7 = b5 + b10 + b15 + b30 + b35
    return [x0, x1, x2, x3, x4, x5, x6, x7]

def square4(x):
    """
    Computes the square of the decomposed field element in F_{2^4}
    F_{2^4} = GF(2)[X]/X^4 + X^3 + 1
    Squaring is linear
    """
    b0,b1,b2,b3 = x
    return [b0+b2+b3, b3, b1+b3, b2+b3]

def square8(x):
    b0,b1,b2,b3,b4,b5,b6,b7 = x
    y0 = b0 + b4            #[1 0 0 0 1 0 0 0]
    y1 = b5 + b7            #[0 0 0 0 0 1 0 1]
    y2 = b1 + b4 + b5       #[0 1 0 0 1 1 0 0]
    y3 = b5 + b6 + b7       #[0 0 0 0 0 1 1 1]
    y4 = b2 + b4 + b5 + b6  #[0 0 1 0 1 1 1 0]
    y5 = b4 + b5 + b6       #[0 0 0 0 1 1 1 0]
    y6 = b3 + b4 + b6       #[0 0 0 1 1 0 1 0]
    y7 = b4 + b6            #[0 0 0 0 1 0 1 0]
    
    return [y0,y1,y2,y3,y4,y5,y6,y7]

def compute_polynomial(coeffs, xpowers):
    assert len(coeffs) == len(xpowers) + 1, f'{len(coeffs)} == {len(xpowers) + 1}'
    return coeffs[0] + sum([c * x for c, x in zip(coeffs[1:], xpowers)])

def _s4_sbox(cell, ONE):
    b0, b1, b2, b3 = bit_decompose4(cell)
    not_b2 = b2 + ONE
    not_b1 = b1 + ONE
    b3_ = b0 + ((b3 + ONE) * not_b2)
    b2_ = b3 + (not_b2 * not_b1)
    not_b3_ = b3_ + ONE
    b1_ = b2 + (not_b1 * not_b3_)
    b0_ = b1 + (not_b3_ * (b2_ + ONE))
    return embed4([b0_, b1_, b2_, b3_])

def _s4_sbox_inv(cell, ONE):
    x0, x1, x2, x3 = cell

    a = (x3 + ONE)
    x1_ = x0 + (a * (x2+ONE))
    b = (x1_ + ONE)
    x2_ = x1 + (b * a)
    c = (x2_+ONE)
    x3_ = x2 + (c * b)
    x0_ = x3 + ((x3_+ONE) * c)

    return [x0_, x1_, x2_, x3_]

def _s8_sbox(cell, ONE):
    b0,b1,b2,b3,b4,b5,b6,b7 = bit_decompose8(cell)
    s6 = b4 + ((b7 + ONE) * (b6 + ONE))
    not_b3 = b3 + ONE
    not_b2 = b2 + ONE
    s5 = b0 + (not_b3 * not_b2)
    s2 = b6 + (not_b2 * (b1 + ONE))
    not_s5 = s5 + ONE
    not_s6 = s6 + ONE
    s3 = b1 + (not_s5 * not_b3)
    s7 = b5 + (not_s6 * not_s5)
    not_s7 = s7 + ONE
    s4 = b3 + (not_s7 * not_s6)
    s1 = b7 + (not_s7 * (s2 + ONE))
    s0 = b2 + ((s3 + ONE) * (s1 + ONE))
    return embed8([s0,s1,s2,s3,s4,s5,s6,s7])

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

def _s4_sbox_inv_crv(cell):
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    x3 = cell * x2
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    q0 = compute_polynomial(Q4.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    p0 = compute_polynomial(P4_inv_1.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    p1 = compute_polynomial(P4_inv_2.get_type(cell.size), [cell, x2, x3, x4, x6, x8, x9, x12])
    return p0*q0 + p1

def _s8_sbox_inv(cell, ONE):
    b0,b1,b2,b3,b4,b5,b6,b7 = bit_decompose8(cell)
    res = [None] * 8
    not_b5 = b5 + ONE
    not_b6 = b6 + ONE
    not_b7 = b7 + ONE
    s2 = b0 + ((b1 + ONE) * (b3 + ONE))
    s3 = b4 + (not_b6 * not_b7)
    s5 = b7 + (not_b5 * not_b6)
    s7 = b1 + ((b2 + ONE) * not_b7)
    not_s2 = s2 + ONE
    not_s3 = s3 + ONE
    s0 = b5 + (not_s2 * not_s3)
    s1 = b3 + (not_s3 * not_b5)
    s6 = b2 + ((s1 + ONE) * not_s2)
    s4 = b6 + ((s6 + ONE) * (s7 + ONE))
    return embed8([s0,s1,s2,s3,s4,s5,s6,s7])

S4_DIRECT_COEFS = VectorConstant(lambda n: [cgf2n(cembed4(c), size=n) for c in [0xc, 0x8, 0x3, 0xd, 0xf, 0x4, 0x8, 0x6, 0x1, 0x9, 0x8, 0xe, 0xc, 0xb]])
S4_INV_DIRECT_COEFS = VectorConstant(lambda n: [cgf2n(cembed4(c), size=n) for c in [0x3, 0xa, 0x6, 0x7, 0x5, 0xa, 0x7, 0x9, 0x6, 0x9, 0x4, 0xd, 0xb]])

def _s4_sbox_direct_poly(cell):
    """
    Computes the interpolation polynomial directly
    """
    x2 = cell * cell
    x3 = x2 * cell
    x4 = x2 * x2
    x5 = x2 * x3
    x6 = x3 * x3
    x7 = x3 * x4
    x8 = x4 * x4
    x9 = x4 * x5
    x10 = x5 * x5
    x12 = x6 * x6
    x13 = x6 * x7
    x14 = x7 * x7
    return compute_polynomial(S4_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x6, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_inv_direct_poly(cell):
    """
    Computes the interpolation polynomial directly
    """
    x2 = cell * cell
    x3 = x2 * cell
    x4 = x2 * x2
    x5 = x2 * x3
    x7 = x3 * x4
    x8 = x4 * x4
    x9 = x4 * x5
    x10 = x2 * x8
    x12 = x4 * x8
    x13 = x5 * x8
    x14 = x7 * x7
    return compute_polynomial(S4_INV_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_direct_poly_sq1(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = cell * x4
    x6 = x2 * x4
    x9 = cell * x8
    x10 = x2 * x8
    x12 = x4 * x8
    #round 3
    x7 = x3 * x4
    x13 = cell * x12
    x14 = x2 * x12
    return compute_polynomial(S4_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x6, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_inv_direct_poly_sq1(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = x4 * cell
    x9 = x8 * cell
    x10 = x2 * x8
    x12 = x4 * x8
    #round 3
    x7 = x3 * x4
    x13 = x5 * x8
    x14 = x4 * x10
    return compute_polynomial(S4_INV_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_direct_poly_sq2(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = cell * x4
    x10 = x2 * x8
    
    # round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    x7 = x3 * x4
    x13 = x8 * x5
    x14 = x10 * x4
    return  compute_polynomial(S4_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x6, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_inv_direct_poly_sq2(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = x4 * cell
    x10 = x2 * x8
    
    #round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    x7 = x3 * x4
    x13 = x5 * x8
    x14 = x4 * x10
    return compute_polynomial(S4_INV_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_direct_poly_sq3(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9 and 5,10
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = cell * x4
    
    # round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    
    cellbits5 = bit_decompose4(x5)
    x10 = embed4(square4(cellbits5))
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    x7 = x3 * x4
    x13 = x5 * x8
    
    # round 4
    x14 = x5 * x9
    return  compute_polynomial(S4_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x6, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_inv_direct_poly_sq3(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9 and 5,10
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = x4 * cell
    
    #round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    cellbits5 = bit_decompose4(x5)
    x10 = embed4(square4(cellbits5))
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    x7 = x3 * x4
    x13 = x5 * x8
    # round 4
    x14 = x5 * x9
    return compute_polynomial(S4_INV_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_direct_poly_sq4(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9 and 5,10 and 7,14,13,11
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = cell * x4
    
    # round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    
    cellbits5 = bit_decompose4(x5)
    x10 = embed4(square4(cellbits5))
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    x7 = x3 * x4
    # round 4
    cellbits7 = bit_decompose4(x7)
    cellbits14 = square4(cellbits7)
    cellbits13 = square4(cellbits14)
    x14 = embed4(cellbits14)
    x13 = embed4(cellbits13)
    return  compute_polynomial(S4_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x6, x7, x8, x9, x10, x12, x13, x14])

def _s4_sbox_inv_direct_poly_sq4(cell):
    """
    Computes the interpolation polynomial directly with squaring 2,4,8 and 3,6,12,9 and 5,10 and 7,14,13,11
    """
    # round 1
    cellbits = bit_decompose4(cell)
    cellbits2 = square4(cellbits)
    cellbits4 = square4(cellbits2)
    cellbits8 = square4(cellbits4)
    
    x2 = embed4(cellbits2)
    x4 = embed4(cellbits4)
    x8 = embed4(cellbits8)
    # round 2
    x3 = x2 * cell
    x5 = x4 * cell
    
    #round 3
    cellbits3 = bit_decompose4(x3)
    cellbits6 = square4(cellbits3)
    cellbits12 = square4(cellbits6)
    cellbits9 = square4(cellbits12)
    cellbits5 = bit_decompose4(x5)
    x10 = embed4(square4(cellbits5))
    x6 = embed4(cellbits6)
    x12 = embed4(cellbits12)
    x9 = embed4(cellbits9)
    
    x7 = x3 * x4
    # round 4
    cellbits7 = bit_decompose4(x7)
    cellbits14 = square4(cellbits7)
    cellbits13 = square4(cellbits14)
    x14 = embed4(cellbits14)
    x13 = embed4(cellbits13)
    return compute_polynomial(S4_INV_DIRECT_COEFS.get_type(cell.size), [cell, x2, x3, x4, x5, x7, x8, x9, x10, x12, x13, x14])

def squares8(cell, repeat):
    squares = []
    cellbits = bit_decompose8(cell)
    for i in range(repeat):
        cellbits_sq = square8(cellbits)
        squares.append(embed8(cellbits_sq))
        cellbits = cellbits_sq
    return squares

def _s8_sbox_crv(cell):
    # round 1
    x2, x4, x8, x16, x32, x64, x128 = squares8(cell, 7)
    
    # round 2
    x3 = cell * x2
    x5 = cell * x4
    
    # round 3
    x7 = x3 * x4
    x11 = x3 * x8
    x6, x12, x24, x48, x96, x192, x129 = squares8(x3, 7)
    x10, x20, x40, x80, x160, x65, x130 = squares8(x5, 7)
    
    # round 4
    x14, x28, x56, x112, x224, x193, x131 = squares8(x7, 7)
    x22, x44, x88, x176, x97, x194, x133 = squares8(x11, 7)
    
    powers = [cell, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12, x14, x16, x20, x22, x24, x28, x32, x40, x44, x48, x56, x64, x65, x80, x88, x96, x97, x112, x128, x129, x130, x131, x133, x160, x176, x192, x193, x194, x224]
    q1 = compute_polynomial(Q8_1.get_type(cell.size), powers)
    q2 = compute_polynomial(Q8_2.get_type(cell.size), powers)
    q3 = compute_polynomial(Q8_3.get_type(cell.size), powers)
    q4 = compute_polynomial(Q8_4.get_type(cell.size), powers)
    q5 = compute_polynomial(Q8_5.get_type(cell.size), powers)
    q6 = compute_polynomial(Q8_6.get_type(cell.size), powers)
    
    p1 = compute_polynomial(P8_1.get_type(cell.size), powers)
    p2 = compute_polynomial(P8_2.get_type(cell.size), powers)
    p3 = compute_polynomial(P8_3.get_type(cell.size), powers)
    p4 = compute_polynomial(P8_4.get_type(cell.size), powers)
    p5 = compute_polynomial(P8_5.get_type(cell.size), powers)
    p6 = compute_polynomial(P8_6.get_type(cell.size), powers)
    p7 = compute_polynomial(P8_7.get_type(cell.size), powers)
    
    # round 5
    return p1*q1 + p2*q2 + p3*q3 + p4*q4 + p5*q5 + p6*q6 + p7

def _s8_sbox_inv_crv(cell):
    # round 1
    x2, x4, x8, x16, x32, x64, x128 = squares8(cell, 7)
    
    # round 2
    x3 = cell * x2
    x5 = cell * x4
    
    # round 3
    x7 = x3 * x4
    x11 = x3 * x8
    x6, x12, x24, x48, x96, x192, x129 = squares8(x3, 7)
    x10, x20, x40, x80, x160, x65, x130 = squares8(x5, 7)
    
    # round 4
    x14, x28, x56, x112, x224, x193, x131 = squares8(x7, 7)
    x22, x44, x88, x176, x97, x194, x133 = squares8(x11, 7)
    
    powers = [cell, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12, x14, x16, x20, x22, x24, x28, x32, x40, x44, x48, x56, x64, x65, x80, x88, x96, x97, x112, x128, x129, x130, x131, x133, x160, x176, x192, x193, x194, x224]
    q1 = compute_polynomial(Q8_1.get_type(cell.size), powers)
    q2 = compute_polynomial(Q8_2.get_type(cell.size), powers)
    q3 = compute_polynomial(Q8_3.get_type(cell.size), powers)
    q4 = compute_polynomial(Q8_4.get_type(cell.size), powers)
    q5 = compute_polynomial(Q8_5.get_type(cell.size), powers)
    q6 = compute_polynomial(Q8_6.get_type(cell.size), powers)
    
    p1 = compute_polynomial(P8_inv_1.get_type(cell.size), powers)
    p2 = compute_polynomial(P8_inv_2.get_type(cell.size), powers)
    p3 = compute_polynomial(P8_inv_3.get_type(cell.size), powers)
    p4 = compute_polynomial(P8_inv_4.get_type(cell.size), powers)
    p5 = compute_polynomial(P8_inv_5.get_type(cell.size), powers)
    p6 = compute_polynomial(P8_inv_6.get_type(cell.size), powers)
    p7 = compute_polynomial(P8_inv_7.get_type(cell.size), powers)
    
    # round 5
    return p1*q1 + p2*q2 + p3*q3 + p4*q4 + p5*q5 + p6*q6 + p7

class SkinnyGF2n(SkinnyBase):
    def __init__(self, variant, vector_size, sbox=None):
        super().__init__(variant)
        if self.cellsize == 4:
            one = cembed4(0x1)
        elif self.cellsize == 8:
            one = cembed8(0x1)
        else:
            raise NotImplemented
        self.ONE = VectorConstant(lambda n: cgf2n(one, size=n))
        self.vector_size = vector_size
        self.sbox = sbox
        print(self.sbox)

    def _embed_cell(self, cell):
        if self.cellsize == 4:
            return embed4(cell)
        elif self.cellsize == 8:
            return embed8(cell)
        else:
            raise NotImplemented
    
    def _xor_cell(self, a, b):
        return a+b
    
    def s4_sbox(self, cell):
        if self.sbox == None:
            ONE = self.ONE.get_type(self.vector_size)
            return _s4_sbox(cell, ONE)
        elif self.sbox in ['mul', 'mul_sq1', 'mul_sq2', 'mul_sq3', 'mul_sq4', 'crv']:
            d = {'mul': _s4_sbox_direct_poly, 'mul_sq1': _s4_sbox_direct_poly_sq1, 'mul_sq2': _s4_sbox_direct_poly_sq2, 'mul_sq3': _s4_sbox_direct_poly_sq3, 'mul_sq4': _s4_sbox_direct_poly_sq4, 'crv': _s4_sbox_crv}
            return d[self.sbox](cell)
        else:
            assert False
    
    def s8_sbox(self, cell):
        assert self.sbox == 'crv'
        return _s8_sbox_crv(cell)
    
    def s4_sbox_inv(self, cell):
        if self.sbox == None:
            return _s4_sbox_inv(cell, self.vector_size)
        elif self.sbox in ['mul', 'mul_sq1', 'mul_sq2', 'mul_sq3', 'mul_sq4', 'crv']:
            d = {'mul': _s4_sbox_inv_direct_poly, 'mul_sq1': _s4_sbox_inv_direct_poly_sq1, 'mul_sq2': _s4_sbox_inv_direct_poly_sq2, 'mul_sq3': _s4_sbox_inv_direct_poly_sq3, 'mul_sq4': _s4_sbox_inv_direct_poly_sq4, 'crv': _s4_sbox_inv_crv}
            return d[self.sbox](cell)
        else:
            assert False
        
    def s8_sbox_inv(self, cell):
        assert self.sbox == 'crv'
        return _s8_sbox_inv_crv(cell)
    
    def add_round_constants(self, state, r, has_tweak):
        assert(len(state) == 16)
        if self.cellsize == 4:
            round_constants = SKINNY_ROUND_CONSTANTS4_EMBEDDED.get_type(self.vector_size)
        elif self.cellsize == 8:
            round_constants = SKINNY_ROUND_CONSTANTS8_EMBEDDED.get_type(self.vector_size)
        else:
            raise NotImplemented
        c_0, c_1, c_2 = round_constants[r]
        state[0] += c_0
        state[4] += c_1
        state[8] += c_2
        return state

class SkinnyBin(SkinnyBase):
    def __init__(self, variant, vector_size):
        super().__init__(variant)
        self.ONE = VectorConstant(lambda n: cgf2n(1, size=n))
        self.vector_size = vector_size

    def _embed_cell(self, cell):
        return cell
    
    def _xor_cell(self, a, b):
        assert len(a) == self.cellsize
        assert len(b) == self.cellsize
        return [ai + bi for ai,bi in zip(a,b)]
    
    def s4_sbox(self, cell):
        ONE = self.ONE.get_type(cell[0].size)
        b0, b1, b2, b3 = cell
        not_b2 = b2 + ONE
        not_b1 = b1 + ONE
        b3_ = b0 + ((b3 + ONE) * not_b2)
        b2_ = b3 + (not_b2 * not_b1)
        not_b3_ = b3_ + ONE
        b1_ = b2 + (not_b1 * not_b3_)
        b0_ = b1 + (not_b3_ * (b2_ + ONE))
        return [b0_, b1_, b2_, b3_]
    
    def s8_sbox(self, cell):
        ONE = self.ONE.get_type(cell[0].size)
        b0,b1,b2,b3,b4,b5,b6,b7 =cell
        s6 = b4 + ((b7 + ONE) * (b6 + ONE))
        not_b3 = b3 + ONE
        not_b2 = b2 + ONE
        s5 = b0 + (not_b3 * not_b2)
        s2 = b6 + (not_b2 * (b1 + ONE))
        not_s5 = s5 + ONE
        not_s6 = s6 + ONE
        s3 = b1 + (not_s5 * not_b3)
        s7 = b5 + (not_s6 * not_s5)
        not_s7 = s7 + ONE
        s4 = b3 + (not_s7 * not_s6)
        s1 = b7 + (not_s7 * (s2 + ONE))
        s0 = b2 + ((s3 + ONE) * (s1 + ONE))
        return [s0,s1,s2,s3,s4,s5,s6,s7]
    
    def s4_sbox_inv(self, cell):
        return _s4_sbox_inv(cell, self.ONE.get_type(cell[0].size))
        
    def s8_sbox_inv(self, cell):
        ONE = self.ONE.get_type(cell[0].size)
        b0,b1,b2,b3,b4,b5,b6,b7 = cell
        not_b5 = b5 + ONE
        not_b6 = b6 + ONE
        not_b7 = b7 + ONE
        s2 = b0 + ((b1 + ONE) * (b3 + ONE))
        s3 = b4 + (not_b6 * not_b7)
        s5 = b7 + (not_b5 * not_b6)
        s7 = b1 + ((b2 + ONE) * not_b7)
        not_s2 = s2 + ONE
        not_s3 = s3 + ONE
        s0 = b5 + (not_s2 * not_s3)
        s1 = b3 + (not_s3 * not_b5)
        s6 = b2 + ((s1 + ONE) * not_s2)
        s4 = b6 + ((s6 + ONE) * (s7 + ONE))
        return [s0,s1,s2,s3,s4,s5,s6,s7]
    
    def add_round_constants(self, state, r, has_tweak):
        assert(len(state) == 16)
        rc = SkinnyBase.ROUND_CONSTANTS[r]
        c0 = rc & 0xf
        c1 = (rc >> 4) & 0x3
        c2 = 0x2
        ONE = self.ONE.get_type(state[0][0].size)
        state[0] = [b + ONE if ((c0 >> i) & 0x1) == 1 else b for i,b in enumerate(state[0])]
        state[4] = [b + ONE if ((c1 >> i) & 0x1) == 1 else b for i,b in enumerate(state[4])]
        state[8] = [b + ONE if ((c2 >> i) & 0x1) == 1 else b for i,b in enumerate(state[8])]
        return state