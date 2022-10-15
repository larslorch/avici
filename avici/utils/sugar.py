import haiku as hk
import numpy as onp

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np
# import torch

"""
Haiku syntactic sugar
"""

# initializers
zero_init = hk.initializers.Constant(0.0)

glorot_uniform_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
glorot_normal_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal") # truncated normal, with stddev 1 / sqrt(fan_in)

kaiming_uniform_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
kaiming_uniform_init_scaled = lambda c: hk.initializers.VarianceScaling(2.0 * c, "fan_in", "uniform")
kaiming_normal_init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
kaiming_normal_init_scaled = lambda c: hk.initializers.VarianceScaling(2.0 * c, "fan_in", "truncated_normal")

# visualization
def print_model_params(f, x):
    """Prints haiku model parameters"""
    print(hk.experimental.tabulate(f)(x))
    return


def print_model_exec(f, x):
    """Prints haiku model execution"""
    for i in hk.experimental.eval_summary(f)(x):
        print("mod := {:30} | in := {} out := {}".format(
            i.module_details.module.module_name, i.args_spec[0], i.output_spec))
    return



"""
JAX syntactic sugar
"""

def index_at_axis(arr, indx, *, axis):
    """
    Returns arr[..., indx, ...] where the position of indx is determined by `axis`
    
    """
    slc = [slice(None)] * arr.ndim
    slc[axis] = indx
    return arr[tuple(slc)]


def matmul_at(a, b, axis):
    """Multiplies matrix `b` at a given axis of `a`
    When applying a permutation matrix, for numerical stability reasons
    use the functions `random_permutation_inv` or `inv_permutation` instead
    
    Arguments:
        a:  [w, x, y, z]
        b:  [d, d]
        axis: int s.t. dimension of a[axis] == d
    
    Returns:
        a @ b at dimension `axis`
    """
    base = "abcdefghijklmn"[:a.ndim]
    return jnp.einsum(f"{base},{base[axis]}z->{base.replace(base[axis], 'z')}", a, b)


def inv_permutation(p):
    """
    Returns inverse permutation
    For any given array `v`, we have v[p][inv_permutation(p)] = p

    Arguments:
        p: [N, ] some permutation with integer elements 0, 1, ..., N-1

    Returns:
        array `pT` where pT[i] = index of `i` in `p`
    """
    pT = jnp.empty_like(p)
    pT = pT.at[p].set(jnp.arange(p.size))
    return pT


def random_permutation_inv(key, n):
    """Samples random permutation of length `n`.
    Returns its inverse too.
    """
    p = random.permutation(key, n)
    return p, inv_permutation(p)


def float2ieee(val):
    b = bin(np.float16(val).view('H'))[2:].zfill(16)
    arr = np.fromstring(",".join(b), sep=',').astype(int)
    return jnp.array(arr)

def vec2ieee(vec):
    assert vec.ndim == 1
    return jnp.array(list(map(float2ieee, vec)))

def mat2ieee(mat):
    assert mat.ndim == 2
    return jnp.array(list(map(vec2ieee, mat)))

def bit2id(b):
    N, d, _ = b.shape
    b_flat = b.reshape(N, d * d)
    return onp.packbits(b_flat, axis=1, bitorder='little')


def id2bit(id, d):
    N, _ = id.shape
    b_flat = onp.unpackbits(id, axis=1, bitorder='little')
    b_flat = b_flat[:, :d * d]
    return b_flat.reshape(N, d, d)


def to_unique(b):
    assert b.ndim == 3
    return id2bit(onp.unique(bit2id(b), axis=0), b.shape[-1])

# def float2bit(f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
#     ## SIGN BIT
#     s = (torch.sign(f + 0.001) * -1 + 1) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
#     s = s.unsqueeze(-1)
#     f1 = torch.abs(f)
#     ## EXPONENT BIT
#     e_scientific = torch.floor(torch.log2(f1))
#     e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
#     e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
#     e = integer2bit(e_decimal, num_bits=num_e_bits)
#     print(s, e)
#     ## MANTISSA
#     f2 = f1 / 2 ** (e_scientific)
#     m2 = remainder2bit(f2 % 1, num_bits=bias)
#     # fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
#     # fin_m = m2[..., :num_m_bits]  # [:,:,:,8:num_m_bits+8]
#     fin_m = m2[..., :num_m_bits]
#     return torch.cat([s, e, fin_m], dim=-1).type(dtype)
#
# def remainder2bit(remainder, num_bits=127):
#     dtype = remainder.type()
#     exponent_bits = torch.arange(num_bits).type(dtype)
#     exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
#     out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
#     return torch.floor(2 * out)
#
# def integer2bit(integer, num_bits=8):
#     dtype = integer.type()
#     exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
#     exponent_bits = exponent_bits.repeat(integer.shape + (1,))
#     out = integer.unsqueeze(-1) / 2 ** exponent_bits
#     return (out - (out % 1)) % 2
# #
#
# # TODO compare big implementations

if __name__ == "__main__":



    print(float2ieee(9.81))
    print(float2ieee(98.1))

    # v = jnp.array([9.81, 98.1])
    # m = jnp.array([[9.81, 98.1],
    #                  [98.1, 9.81]])
    #
    # print(vec2ieee(v))
    # print(mat2ieee(m))

    # print(float2bit(torch.tensor(9.81), num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.int32))



    exit()




    key = random.PRNGKey(0)
    n = 1000
    
    for _ in range(30):
        
        # # np sum accuracy check
        # x = np.random.normal(size=(n,))
        # p = np.random.permutation(n)
        # print(np.abs(x.sum() - x[p].sum()))

        # dynamic slicing
        key, subk = random.split(key)
        x = random.normal(key, shape=(64, n, 32))
        key, subk = random.split(key)
        p = random.permutation(subk, n)

        print(jnp.allclose(x[:, p, :], index_at_axis(x, p, axis=1)))
        assert jnp.allclose(x[:, p, :], index_at_axis(x, p, axis=1))

        # inverse permutations
        key, subk = random.split(key)
        x = random.normal(key, shape=(n,))

        key, subk = random.split(key)
        p, pT = random_permutation_inv(key, n)

        print(jnp.allclose(x, x[p][pT]), jnp.abs(x.sum() - x[p][pT].sum()), jnp.abs(x.sum() - x[p].sum()))
        print(jnp.allclose(x, x[pT][p]), jnp.abs(x.sum() - x[pT][p].sum()))
        assert jnp.allclose(x, x[p][pT])
        assert jnp.allclose(x, x[pT][p])

    print('Passed.')
