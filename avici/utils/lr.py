import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt


def const_schedule(lr):
    return lambda _: jnp.array(lr)


# def noam_schedule(model_size, factor=1.0, warmup=4000):
#     """Warmup + Square root decay as in vanilla transformer"""
#     def noam(t):
#         return factor * ((model_size ** (-0.5)) * jnp.minimum((t+1) ** (-0.5), (t+1) * (warmup ** (-1.5))))
#     return noam

def noam_schedule(max_lr=1.0, warmup=4000):
    """Warmup + Square root decay as in vanilla transformer, but different parameterization"""
    def noam(t):
        return jnp.minimum(max_lr * (warmup ** 0.5) * (t+1) ** (-0.5), (t+1) * max_lr / warmup)
    return noam



if __name__ == "__main__":

    m = 200000
    step = 100
    opts = [
        # noam_schedule(512, factor=0.15, warmup=16000),
        # noam_schedule(512, factor=0.07, warmup=16000),
        # noam_schedule(max_lr=0.1, warmup=16000),
        # noam_schedule(max_lr=0.05, warmup=16000),
        # noam_schedule(max_lr=0.05, warmup=32000),
        optax._src.schedule.piecewise_constant_schedule(0.002, {50000: 0.1, 100000: 0.1, 150000: 0.1}),
        optax._src.schedule.piecewise_interpolate_schedule("linear", 0.002, {25000: 1.0, 75000: 0.1, 125000: 0.1, 175000: 0.1}),
    ]

    arr = [[opt(i) for opt in opts] for i in jnp.arange(1, m, step)]
    arrnp = jnp.array(arr) # steps x opt
    for i in range(arrnp.shape[1]):
        e = arrnp[-1, i]
        ma = jnp.max(arrnp[:, i])
        print(f"opt {i}    max: {ma}   end {e}")

    plt.plot(jnp.arange(1, m, step), arr)
    # plt.semilogy(jnp.arange(1, m, step), arr)
    plt.legend(["256:0.15:16000", "256:0.07:16000"])
    plt.show()
