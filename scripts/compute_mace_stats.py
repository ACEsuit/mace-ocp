from ocpmodels.datasets import OC22LmdbDataset, data_list_collater
from ocpmodels.models import BaseModel
from torch.utils.data import DataLoader
from torch_scatter import scatter
from tqdm import tqdm
import numpy as np
import torch


dset_path = "/checkpoint/bmwood/oc20_small/oc20_xxs_dataset/lmdb/train"
oc20_ref = "/checkpoint/janlan/ocp/other_data/final_ref_energies_02_07_2021.pkl"
cutoff = 6.0
max_neighbors = 500

# adapted from https://github.com/ACEsuit/mace/blob/f304c07bbafd651f5a52a447c954f1ef561d42e2/mace/data/utils.py#L170
def compute_average_E0s(dl: DataLoader):
    len_dset = len(dl.dataset)
    len_zs = 83

    A = np.zeros((len_dset, len_zs))
    B = np.zeros(len_dset)

    for i, batch in tqdm(enumerate(dl)):
        breakpoint()
        if i == len(dl) - 1: sz = batch.y.shape[0]
        else: sz = dl.batch_size

        B[i * dl.batch_size : i * dl.batch_size + sz] = batch.y
        for j, do in enumerate(batch.to_data_list()):
            for z in range(len_zs):
                A[i * dl.batch_size + j, z] = np.count_nonzero(do.atomic_numbers - 1 == z)  # -1 because of 0-indexing


    E0s = np.linalg.lstsq(A, B, rcond=None)[0]
    return E0s


if __name__ == "__main__":
    cfg = {
        "src": dset_path,
        "oc20_ref": oc20_ref,
        "total_energy": True,
        "otf_graph": True,
    }
    dset = OC22LmdbDataset(cfg)

    dl = DataLoader(
        dset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_list_collater,
        num_workers=32,
        pin_memory=True,
    )

    E0s = compute_average_E0s(dl)
    print("E0s", E0s.tolist())

    E0s = torch.tensor(E0s)

    model = BaseModel()

    e, f, n, nbrs = [], [], [], []

    for i, batch in tqdm(enumerate(dl)):

        n.append(batch.natoms)
        f.append(batch.force)

        graph = model.generate_graph(
            batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_pbc=True,
            otf_graph=True,
        )

        nbrs.append(torch.div(graph[-1], batch.natoms))

        # compute interaction energies
        atom_e0 = E0s[batch.atomic_numbers.long() - 1]
        graph_e0 = scatter(src=atom_e0, index=batch.batch, dim=-1)

        e.append(batch.y - graph_e0)

    e = torch.cat(e, dim=0)
    f = torch.cat(f, dim=0)
    n = torch.cat(n, dim=0)
    nbrs = torch.cat(nbrs, dim=0)

    print("interaction energy", e.mean().item())
    print("interaction energy per atom", torch.div(e, n).mean().item())
    print("forces rms", torch.sqrt(torch.mean(torch.square(f))).item())
    print("neighbors", nbrs.mean().item())
