import csv
import numpy as onp

import json
from avici.experiment.utils import save_csv

from pprint import pprint

from avici.definitions import PROJECT_DIR, REAL_DATA_SUBDIR

from avici.definitions import FILE_DATA_META, FILE_DATA_G, \
    FILE_DATA_X_OBS, FILE_DATA_X_INT, FILE_DATA_X_INT_INFO


if __name__ == "__main__":

    # load
    path = PROJECT_DIR / REAL_DATA_SUBDIR / "sachs_dcdi"

    g = onp.load(path / "DAG1.npy")
    data = onp.load(path / "data_interv1.npy")
    n, d = data.shape
    assert g.shape[0] == g.shape[1] == d

    with open(path / "intervention1.csv", newline='') as csvfile:
        intervention = [[int(r) for r in row] for row in csv.reader(csvfile, delimiter=' ')]

    print("g", g.shape)
    print("data", data.shape)
    print("intervention", len(intervention))

    assert len(intervention) == n

    # convert to mask format
    interv_mask = []
    is_observ = onp.array([len(r) == 0 for r in intervention]).astype(bool)
    for i in range(len(intervention)):
        t = intervention[i]
        assert len(t) in [0, 1]
        if t:
            interv_mask.append(onp.eye(d)[t[0]])
        else:
            interv_mask.append(onp.zeros(d))

    interv_mask = onp.array(interv_mask)
    assert data.shape == interv_mask.shape

    unique = {tuple(interv_mask[i].tolist()) for i in range(len(interv_mask))}

    # save in our format
    new_path =  PROJECT_DIR / REAL_DATA_SUBDIR / "sachs"
    new_path.mkdir(exist_ok=True)

    data_observ = data[is_observ]
    data_interv, data_interv_info = data[~is_observ], interv_mask[~is_observ]
    assert onp.allclose(interv_mask[is_observ].sum(), 0.0)
    assert data_interv.shape == data_interv_info.shape
    assert onp.all(onp.allclose(interv_mask[~is_observ].sum(-1), 1))

    save_csv(g.astype(onp.int32), new_path / FILE_DATA_G)
    save_csv(data_observ.astype(onp.float32), new_path / FILE_DATA_X_OBS)
    save_csv(data_interv.astype(onp.float32), new_path / FILE_DATA_X_INT)
    save_csv(data_interv_info.astype(onp.int32),  new_path / FILE_DATA_X_INT_INFO)

    meta_info_path = new_path / FILE_DATA_META
    meta_info_path.parent.mkdir(exist_ok=True, parents=True)
    with open(meta_info_path, "w") as file:
        meta_info = {
            "is_count_data": False,
            "n_vars": d,
            "n_data_observational": data_observ.shape[0],
            "n_data_interventional": data_interv.shape[0],
            "n_heldout_data_observational": 0,
            "n_heldout_data_interventional": 0,
            "model": "real",
        }
        json.dump(meta_info, file, indent=4, sort_keys=True)
        pprint(meta_info)

    print(f"Saved sachs data at: {new_path}")


