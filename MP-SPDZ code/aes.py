from copy import copy
from Compiler.types import VectorArray, cgf2n, sgf2n
from Compiler.library import vectorize, print_ln

class SpdzBox(object):
    def init_matrices(self):
        self.matrix_inv = [ 
            [0,0,1,0,0,1,0,1],
            [1,0,0,1,0,0,1,0],
            [0,1,0,0,1,0,0,1],
            [1,0,1,0,0,1,0,0],
            [0,1,0,1,0,0,1,0],
            [0,0,1,0,1,0,0,1],
            [1,0,0,1,0,1,0,0],
            [0,1,0,0,1,0,1,0]
        ]
        to_add = [1,0,1,0,0,0,0,0]
        self.addition_inv = [cgf2n(_,size=self.nparallel) for _ in to_add]
        self.forward_matrix = [
            [1,0,0,0,1,1,1,1],
            [1,1,0,0,0,1,1,1],
            [1,1,1,0,0,0,1,1],
            [1,1,1,1,0,0,0,1],
            [1,1,1,1,1,0,0,0],
            [0,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,0],
            [0,0,0,1,1,1,1,1]
        ]
        forward_add = [1,1,0,0,0,1,1,0]
        self.forward_add = VectorArray(len(forward_add), cgf2n, self.nparallel)
        for i,x in enumerate(forward_add):
            self.forward_add[i] = cgf2n(x, size=self.nparallel)
        self.K01 = VectorArray(8, cgf2n, self.nparallel)
        for idx in range(8):
            self.K01[idx] = self.aes.ApplyBDEmbedding([0,1]) ** idx

    def __init__(self, aes):
        self.nparallel = aes.nparallel
        self.aes = aes
        constants = [
            0x63, 0x8F, 0xB5, 0x01, 0xF4, 0x25, 0xF9, 0x09, 0x05
        ]
        self.powers = [
            0, 127, 191, 223, 239, 247, 251, 253, 254
        ]
        self.constants = [aes.ApplyEmbedding(cgf2n(_,size=self.nparallel)) for _ in constants]
        self.init_matrices()

    def forward_bit_sbox(self, emb_byte):
        emb_byte_inverse = self.aes.inverseMod(emb_byte)
        unembedded_x = self.aes.InverseBDEmbedding(emb_byte_inverse)

        linear_transform = list()
        for row in self.forward_matrix:
            result = cgf2n(0, size=self.nparallel)
            for idx in range(len(row)):
                result = result + unembedded_x[idx] * row[idx]
            linear_transform.append(result)

        #do the sum(linear_transform + additive_layer)
        summation_bd = [0 for _ in range(8)]
        for idx in range(8):
            summation_bd[idx] = linear_transform[idx] + self.forward_add[idx]

        #Now raise this to power of 254
        result = cgf2n(0,size=self.nparallel)
        for idx in range(8):
            result += self.aes.ApplyBDEmbedding([summation_bd[idx]]) * self.K01[idx];
        return result

    def apply_sbox(self, what):
        #applying with the multiplicative chain
        return self.forward_bit_sbox(what)
    
    def backward_bit_sbox(self, emb_byte):
        unembedded_x = self.aes.InverseBDEmbedding(emb_byte)
        # invert additive layer
        unembedded_x = [x - c for x,c in zip(unembedded_x, self.forward_add)]
        # invert linear transform
        linear_transform = list()
        for row in self.matrix_inv:
            result = cgf2n(0, size=self.nparallel)
            for idx in range(len(row)):
                result = result + unembedded_x[idx] * row[idx]
            linear_transform.append(result)
        return self.aes.inverseMod(self.aes.embed_helper(linear_transform))

rcon_raw = [
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
    0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
    0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
    0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
    0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
    0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
    0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
    0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
    0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
    0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
    0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
    0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
    0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
    0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb
]

# such constants. very wow.
_embedded_powers = [
    [0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000,0x100000000,0x200000000,0x400000000,0x800000000,0x1000000000,0x2000000000,0x4000000000,0x8000000000],
    [0x1,0x4,0x10,0x40,0x100,0x400,0x1000,0x4000,0x10000,0x40000,0x100000,0x400000,0x1000000,0x4000000,0x10000000,0x40000000,0x100000000,0x400000000,0x1000000000,0x4000000000,0x108401,0x421004,0x1084010,0x4210040,0x10840100,0x42100400,0x108401000,0x421004000,0x1084010000,0x4210040000,0x840008401,0x2100021004,0x8400084010,0x1000000842,0x4000002108,0x100021,0x400084,0x1000210,0x4000840,0x10002100],
    [0x1,0x10,0x100,0x1000,0x10000,0x100000,0x1000000,0x10000000,0x100000000,0x1000000000,0x108401,0x1084010,0x10840100,0x108401000,0x1084010000,0x840008401,0x8400084010,0x4000002108,0x400084,0x4000840,0x40008400,0x400084000,0x4000840000,0x8021004,0x80210040,0x802100400,0x8021004000,0x210802008,0x2108020080,0x1080010002,0x800008421,0x8000084210,0x108,0x1080,0x10800,0x108000,0x1080000,0x10800000,0x108000000,0x1080000000],
    [0x1,0x100,0x10000,0x1000000,0x100000000,0x108401,0x10840100,0x1084010000,0x8400084010,0x400084,0x40008400,0x4000840000,0x80210040,0x8021004000,0x2108020080,0x800008421,0x108,0x10800,0x1080000,0x108000000,0x800108401,0x10002108,0x1000210800,0x20004010,0x2000401000,0x42008020,0x4200802000,0x84200842,0x8420084200,0x2000421084,0x40000420,0x4000042000,0x10040,0x1004000,0x100400000,0x40108401,0x4010840100,0x1080200040,0x8021080010,0x2100421080],
    [0x1,0x10000,0x100000000,0x10840100,0x8400084010,0x40008400,0x80210040,0x2108020080,0x108,0x1080000,0x800108401,0x1000210800,0x2000401000,0x4200802000,0x8420084200,0x40000420,0x10040,0x100400000,0x4010840100,0x8021080010,0x40108421,0x1080000040,0x100421080,0x4200040100,0x1084200,0x842108401,0x1004210042,0x2008400004,0x4210000008,0x401080210,0x840108001,0x1000000840,0x100001000,0x840100,0x8401000000,0x800000001,0x84210800,0x2100001084,0x210802100,0x8001004210],
    [0x1,0x100000000,0x8400084010,0x80210040,0x108,0x800108401,0x2000401000,0x8420084200,0x10040,0x4010840100,0x40108421,0x100421080,0x1084200,0x1004210042,0x4210000008,0x840108001,0x100001000,0x8401000000,0x84210800,0x210802100,0x800000401,0x2100420080,0x8000004000,0x4010002,0x4000800100,0x842000420,0x8421084,0x421080210,0x80010042,0x10802108,0x800000020,0x1084,0x8401084010,0x1004200040,0x4000840108,0x100020,0x2108401000,0x8400080210,0x84210802,0x10802100],
    [0x1,0x8400084010,0x108,0x2000401000,0x10040,0x40108421,0x1084200,0x4210000008,0x100001000,0x84210800,0x800000401,0x8000004000,0x4000800100,0x8421084,0x80010042,0x800000020,0x8401084010,0x4000840108,0x2108401000,0x84210802,0x20,0x8000004210,0x2100,0x8401004,0x200800,0x802108420,0x21084000,0x4200842108,0x2000020000,0x1084210000,0x100421,0x1004010,0x10840008,0x108421080,0x1000200840,0x108001,0x8020004210,0x10040108,0x2108401004,0x1084210040],
    [0x1,0x108,0x10040,0x1084200,0x100001000,0x800000401,0x4000800100,0x80010042,0x8401084010,0x2108401000,0x20,0x2100,0x200800,0x21084000,0x2000020000,0x100421,0x10840008,0x1000200840,0x8020004210,0x2108401004,0x400,0x42000,0x4010000,0x421080000,0x21004,0x2008420,0x210800100,0x4200002,0x401000210,0x2108401084,0x8000,0x840000,0x80200000,0x8421000000,0x420080,0x40108400,0x4210002000,0x84000040,0x8020004200,0x2108400084]
]

class Aes128:
    
    def __init__(self, nparallel):
        self.nparallel = nparallel
        self.rcon = VectorArray(len(rcon_raw), cgf2n, nparallel)
        for idx in range(len(rcon_raw)):
            self.rcon[idx] = cgf2n(rcon_raw[idx],size=nparallel)

        self.powers2 = VectorArray(8, cgf2n, nparallel)
        for idx in range(8):
            self.powers2[idx] = cgf2n(2,size=nparallel) ** (5 * idx)
        
        # mixColumn takes a column and does stuff

        self.Kv = VectorArray(4, cgf2n, nparallel)
        self.Kv[1] = self.ApplyEmbedding(cgf2n(1,size=nparallel))
        self.Kv[2] = self.ApplyEmbedding(cgf2n(2,size=nparallel))
        self.Kv[3] = self.ApplyEmbedding(cgf2n(3,size=nparallel))
        self.Kv[4] = self.ApplyEmbedding(cgf2n(4,size=nparallel))
        
        self.InvMixColKv = [None] * 4
        self.InvMixColKv[0] = self.ApplyEmbedding(cgf2n(0xe,size=nparallel))
        self.InvMixColKv[1] = self.ApplyEmbedding(cgf2n(0xb,size=nparallel))
        self.InvMixColKv[2] = self.ApplyEmbedding(cgf2n(0xd,size=nparallel))
        self.InvMixColKv[3] = self.ApplyEmbedding(cgf2n(0x9,size=nparallel))
    
        self.enum_squarings = VectorArray(8 * 40, cgf2n, nparallel)
        for i,_list in enumerate(_embedded_powers):
            for j,x in enumerate(_list):
                self.enum_squarings[40 * i + j] = cgf2n(x, size=nparallel)
        
        self.box = SpdzBox(self)
    
    def ApplyEmbedding(self, x):
        in_bytes = x.bit_decompose(8)

        out_bytes = [cgf2n(0, size=self.nparallel) for _ in range(8)]

        out_bytes[0] = sum(in_bytes[0:8])
        out_bytes[1] = sum(in_bytes[idx] for idx in range(1, 8, 2))
        out_bytes[2] = in_bytes[2] + in_bytes[3] + in_bytes[6] + in_bytes[7]
        out_bytes[3] = in_bytes[3] + in_bytes[7]
        out_bytes[4] = in_bytes[4] + in_bytes[5] + in_bytes[6] + in_bytes[7]
        out_bytes[5] = in_bytes[5] + in_bytes[7]
        out_bytes[6] = in_bytes[6] + in_bytes[7]
        out_bytes[7] = in_bytes[7]

        return sum(self.powers2[idx] * out_bytes[idx] for idx in range(8))


    def embed_helper(self, in_bytes):
        out_bytes = [None] * 8
        out_bytes[0] = sum(in_bytes[0:8])
        out_bytes[1] = sum(in_bytes[idx] for idx in range(1, 8, 2))
        out_bytes[2] = in_bytes[2] + in_bytes[3] + in_bytes[6] + in_bytes[7]
        out_bytes[3] = in_bytes[3] + in_bytes[7]
        out_bytes[4] = in_bytes[4] + in_bytes[5] + in_bytes[6] + in_bytes[7]
        out_bytes[5] = in_bytes[5] + in_bytes[7]
        out_bytes[6] = in_bytes[6] + in_bytes[7]
        out_bytes[7] = in_bytes[7]
        return out_bytes

    def ApplyBDEmbedding(self, x):
        entire_sequence_bits = copy(x)

        while len(entire_sequence_bits) < 8:
            entire_sequence_bits.append(0)

        in_bytes = entire_sequence_bits
        out_bytes = self.embed_helper(in_bytes)

        return sum(self.powers2[idx] * out_bytes[idx] for idx in range(8))


    def PreprocInverseEmbedding(self, x):
        in_bytes = x.bit_decompose_embedding()

        out_bytes = [cgf2n(0, size=self.nparallel) for _ in range(8)]

        out_bytes[7] = in_bytes[7]
        out_bytes[6] = in_bytes[6] + out_bytes[7]
        out_bytes[5] = in_bytes[5] + out_bytes[7]
        out_bytes[4] = in_bytes[4] + out_bytes[5] + out_bytes[6] + out_bytes[7]
        out_bytes[3] = in_bytes[3] + out_bytes[7]
        out_bytes[2] = in_bytes[2] + out_bytes[3] + out_bytes[6] + out_bytes[7]
        out_bytes[1] = in_bytes[1] +  out_bytes[3] + out_bytes[5] + out_bytes[7]
        out_bytes[0] = in_bytes[0] + sum(out_bytes[1:8])

        return out_bytes

    def InverseEmbedding(self,x):
        out_bytes = self.PreprocInverseEmbedding(x)
        ret = cgf2n(0, size=self.nparallel)
        for idx in range(7, -1, -1):
            ret = ret + (cgf2n(2, size=self.nparallel) ** idx) * out_bytes[idx]
        return ret

    def InverseBDEmbedding(self, x):
        return self.PreprocInverseEmbedding(x)

    def expandAESKey(self, cipherKey, Nr = 10, Nb = 4, Nk = 4):
        #cipherkey should be in hex
        cipherKeySize = len(cipherKey)

        round_key = [sgf2n(0,size=self.nparallel)] * 176
        temp = [cgf2n(0,size=self.nparallel)] * 4

        for i in range(Nk):
            for j in range(4):
                round_key[4 * i + j] = cipherKey[4 * i + j]

        for i in range(Nk, Nb * (Nr + 1)):
            for j in range(4):
                temp[j] = round_key[(i-1) * 4 + j]
            if i % Nk == 0:
                #rotate the 4 bytes word to the left
                k = temp[0]
                temp[0] = temp[1]
                temp[1] = temp[2]
                temp[2] = temp[3]
                temp[3] = k

                #now substitute word
                temp[0] = self.box.apply_sbox(temp[0])
                temp[1] = self.box.apply_sbox(temp[1])
                temp[2] = self.box.apply_sbox(temp[2])
                temp[3] = self.box.apply_sbox(temp[3])

                temp[0] = temp[0] + self.ApplyEmbedding(self.rcon[int(i//Nk)])

            for j in range(4):
                round_key[4 * i + j] = round_key[4 * (i - Nk) + j] + temp[j]
        return round_key

        #Nr = 10 -> The number of rounds in AES Cipher.
        #Nb = 4 -> The number of columns of the AES state
        #Nk = 4 -> The number of words of a AES key 

    def SecretArrayEmbedd(self,byte_array):
        return [self.ApplyEmbedding(_) for _ in byte_array]

    def subBytes(self,state):
        for i in range(len(state)):
            state[i] = self.box.apply_sbox(state[i])

    def invSubBytes(self,state):
        for i in range(len(state)):
            state[i] = self.box.backward_bit_sbox(state[i])

    def addRoundKey(self,roundKey):
        @vectorize
        def inner(state):
            for i in range(len(state)):
                state[i] = state[i] + roundKey[i]
        return inner

    def mixColumn(self,column):
        temp = copy(column)
        # no multiplication
        doubles = [self.Kv[2] * t for t in temp]
        column[0] = doubles[0] + (temp[1] + doubles[1]) + temp[2] + temp[3]
        column[1] = temp[0] + doubles[1] + (temp[2] + doubles[2]) + temp[3]
        column[2] = temp[0] + temp[1] + doubles[2] + (temp[3] + doubles[3])
        column[3] = (temp[0] + doubles[0]) + temp[1] + temp[2] + doubles[3]

    def mixColumns(self,state):
        for i in range(4):
            column = []
            for j in range(4):
                column.append(state[i*4+j])
            self.mixColumn(column)
            for j in range(4):
                state[i*4+j] = column[j]

    def invMixColumn(self, column):
        temp = copy(column)
        column[0] = self.InvMixColKv[0] * temp[0] + self.InvMixColKv[1] * temp[1] + self.InvMixColKv[2] * temp[2] + self.InvMixColKv[3] * temp[3]
        column[1] = self.InvMixColKv[3] * temp[0] + self.InvMixColKv[0] * temp[1] + self.InvMixColKv[1] * temp[2] + self.InvMixColKv[2] * temp[3]
        column[2] = self.InvMixColKv[2] * temp[0] + self.InvMixColKv[3] * temp[1] + self.InvMixColKv[0] * temp[2] + self.InvMixColKv[1] * temp[3]
        column[3] = self.InvMixColKv[1] * temp[0] + self.InvMixColKv[2] * temp[1] + self.InvMixColKv[3] * temp[2] + self.InvMixColKv[0] * temp[3]

    def invMixColumns(self,state):
        for i in range(4):
            column = []
            for j in range(4):
                column.append(state[i*4+j])
            self.invMixColumn(column)
            for j in range(4):
                state[i*4+j] = column[j]

    def rotate(self, word, n):
        return word[n:]+word[0:n]

    def shiftRows(self,state):
        for i in range(4):
            state[i::4] = self.rotate(state[i::4],i)

    def invShiftRows(self,state):
        for i in range(4):
            word = state[i::4]
            state[i::4] = word[4-i:] + word[0:4-i]

    def state_collapse(self,state):
        return [self.InverseEmbedding(_) for _ in state]

    def fancy_squaring(self,bd_val, exponent):
        #This is even more fancy; it performs directly on bit dec values
        #returns x ** (2 ** exp) from a bit decomposed value
        return sum(self.enum_squarings[exponent * 40 + idx] * bd_val[idx]
                for idx in range(len(bd_val)))

    def inverseMod(self,val):
        #embedded now!
        #returns x ** 254 using offline squaring
        #returns an embedded result
        
        if isinstance(val, (sgf2n,cgf2n)):
            raw_bit_dec = val.bit_decompose_embedding()
        else:
            assert isinstance(val, list)
            raw_bit_dec = val
        bd_val = [cgf2n(0,size=self.nparallel)] * 40

        for idx in range(40):
            if idx % 5 == 0:
                bd_val[idx] = raw_bit_dec[idx // 5]

        bd_squared = bd_val
        squared_index = 2

        mapper = [0] * 129
        for idx in range(1, 8):
            bd_squared = self.fancy_squaring(bd_val, idx)
            mapper[squared_index] = bd_squared
            squared_index *= 2

        enum_powers = [
            2, 4, 8, 16, 32, 64, 128
        ]

        inverted_product = \
            ((mapper[2] * mapper[4]) * (mapper[8] * mapper[16])) * ((mapper[32] * mapper[64]) * mapper[128])
        return inverted_product

    def aesRound(self,roundKey):
        @vectorize
        def inner(state):
            self.subBytes(state)
            self.shiftRows(state)
            self.mixColumns(state)
            self.addRoundKey(roundKey)(state)
        return inner

    def invAesRound(self,roundKey):
        @vectorize
        def inner(state):
            self.addRoundKey(roundKey)(state)
            self.invMixColumns(state)
            self.invShiftRows(state)
            self.invSubBytes(state) 
        return inner

    # returns a 16-byte round key based on an expanded key and round number
    def createRoundKey(self, expandedKey, n):
        return expandedKey[(n*16):(n*16+16)]

    # wrapper function for 10 rounds of AES since we're using a 128-bit key
    def aesMain(self, expandedKey, numRounds=10):
        @vectorize
        def inner(state):
            roundKey = self.createRoundKey(expandedKey, 0)
            self.addRoundKey(roundKey)(state)
            for i in range(1, numRounds):

                roundKey = self.createRoundKey(expandedKey, i)
                self.aesRound(roundKey)(state)

            roundKey = self.createRoundKey(expandedKey, numRounds)

            self.subBytes(state) 
            self.shiftRows(state)
            self.addRoundKey(roundKey)(state)
        return inner

    # wrapper function for 10 rounds of AES since we're using a 128-bit key
    def invAesMain(self, expandedKey, numRounds=10):
        @vectorize
        def inner(state):
            roundKey = self.createRoundKey(expandedKey, numRounds)
            self.addRoundKey(roundKey)(state)
            self.invShiftRows(state)
            self.invSubBytes(state)
            
            for i in list(range(1, numRounds))[::-1]:
                roundKey = self.createRoundKey(expandedKey, i)
                self.invAesRound(roundKey)(state)

            roundKey = self.createRoundKey(expandedKey, 0)
            self.addRoundKey(roundKey)(state)
        return inner

    def encrypt_without_key_schedule(self, expandedKey):
        @vectorize
        def encrypt(plaintext):
            plaintext = self.SecretArrayEmbedd(plaintext)
            self.aesMain(expandedKey)(plaintext)
            return self.state_collapse(plaintext)
        return encrypt

    def decrypt_without_key_schedule(self, expandedKey):
        @vectorize
        def decrypt(ciphertext):
            ciphertext = self.SecretArrayEmbedd(ciphertext)
            self.invAesMain(expandedKey)(ciphertext)
            return self.state_collapse(ciphertext)
        return decrypt

"""
Test Vectors:

plaintext:
6bc1bee22e409f96e93d7e117393172a

key: 
2b7e151628aed2a6abf7158809cf4f3c

resulting cipher
3ad77bb40d7a3660a89ecaf32466ef97 

"""

test_message = "6bc1bee22e409f96e93d7e117393172a"
test_key = "2b7e151628aed2a6abf7158809cf4f3c"

def conv(x):
    return [int(x[i : i + 2], 16) for i in range(0, len(x), 2)]

def single_encryption(nparallel = 1):
    key = [sgf2n(x, size=nparallel) for x in conv(test_key)]
    message = [sgf2n(x, size=nparallel) for x in conv(test_message)]

    cipher = Aes128(nparallel)

    key = [cipher.ApplyEmbedding(_) for _ in key]
    expanded_key = cipher.expandAESKey(key)

    AES = cipher.encrypt_without_key_schedule(expanded_key)

    ciphertext = AES(message)

    for block in ciphertext:
        print_ln('%s', block.reveal())

    invAES = cipher.decrypt_without_key_schedule(expanded_key)
    decrypted_message = invAES(ciphertext)
    for block in (decrypted_message):
        print_ln('%s', block.reveal())

#single_encryption(10)
