import avici
from avici import simulate_data
from avici.metrics import shd

if __name__ == "__main__":

    # g: [d, d] causal graph of `d` variables
    # x: [n, d] data matrix containing `n` observations of the `d` variables
    # interv: [n, d] binary matrix indicating which nodes were intervened upon
    g, x, interv = simulate_data(n=100, n_interv=20, d=20, domain="linear-gaussian-scm")

    # load pretrained model
    model = avici.load_pretrained(download="linear")

    # g: [d, d] predicted edge probabilities of the causal graph
    g_prob = model(x=x, interv=interv)

    g_pred = (g_prob > 0.5).astype(int)

    print(f"g:\n{g}")
    print(f"g_pred:\n{g_pred}")
    print(f"SHD:\n{shd(g, g_pred)}")
