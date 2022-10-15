

def update_ave(ave_d, d, n=1, is_mean=False):
    for k, v in d.items():
        ave_d[("__ctr__", k)] += n
        if is_mean:
            ave_d[k] += v * n
        else:
            ave_d[k] += v
    return ave_d


def retrieve_ave(ave_d):
    out = {}
    for k, v in ave_d.items():
        # check if `k` is a ctr element
        if isinstance(k, tuple) and k[0] == "__ctr__":
            continue
        # process value `v`
        try:
            v_val = v.item()
        # distributed training case with `pmap`: retrieve 1 device replicate
        except TypeError:
            v_val = v[0].item()
        # not an array
        except AttributeError:
            v_val = v
        assert ("__ctr__", k) in ave_d.keys()
        out[k] = v_val / ave_d[("__ctr__", k)]
    return out


def logger(d, init=False, l=20):
    """
    Args:
        d: list of tuples of key, value pairs to be printed
        init: whether or not to print header
        l: min length of each elt
    """
    assert type(l) is int

    formats = {
        float: "{:>" + str(l) + ".4f}",
    }
    format_default = "{:>" + str(l) + "}"

    s_header = "\t".join(len(d) * [format_default])
    s = "\t".join([formats.get(type(v), format_default) for _, v in d])

    if init:
        print(s_header.format(*[k for k, _ in d]))
    print(s.format(*[v for _, v in d]))

