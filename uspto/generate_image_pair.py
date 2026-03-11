import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Geometry import Point3D
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings


def get_center_x(mol):
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    return sum(xs) / len(xs) if xs else 0


def get_x_range(mol):
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    return min(xs), max(xs)


def shift_mol_horizontally(mol, offset_x):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + offset_x, pos.y, pos.z))


def align_sort_and_separate(product, reactant_list, padding=2.0):
    # 1. initial alignment
    temp_mols = []
    for mol in reactant_list:
        mcs = rdFMCS.FindMCS([product, mol], timeout=60)
        if mcs.numAtoms > 0:
            mcs_template = Chem.MolFromSmarts(mcs.smartsString)
            try:
                AllChem.GenerateDepictionMatching2DStructure(mol, product, refPatt=mcs_template)
            except:
                AllChem.Compute2DCoords(mol)
        else:
            AllChem.Compute2DCoords(mol)
        temp_mols.append(mol)

    # 2. sort by center x
    temp_mols.sort(key=lambda m: get_center_x(m))

    # 3. separate to avoid overlap
    aligned_mols = []
    last_max_x = None

    for i, mol in enumerate(temp_mols):
        if i == 0:
            _, max_x = get_x_range(mol)
            last_max_x = max_x
        else:
            curr_min_x, curr_max_x = get_x_range(mol)
            # calculate shift distance
            shift_dist = (last_max_x + padding) - curr_min_x
            shift_mol_horizontally(mol, shift_dist)

            # update last_max_x
            _, new_max_x = get_x_range(mol)
            last_max_x = new_max_x

        aligned_mols.append(mol)

    return aligned_mols


def draw_aligned_molecules(reactant_smiles_raw: str, product_smiles: str) -> tuple[Image.Image, Image.Image] | tuple[None, None]:
    if "." in product_smiles:
        return None, None  # Skip if product is not a single molecule

    product = Chem.MolFromSmiles(product_smiles)
    if product is None:
        return None, None
    AllChem.Compute2DCoords(product)

    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_raw.split('.')]
    if not reactant_mols or any(m is None for m in reactant_mols):
        return None, None

    aligned_reactants = align_sort_and_separate(product, reactant_mols, padding=2.0)

    combined_reactant = aligned_reactants[0]
    for i in range(1, len(aligned_reactants)):
        combined_reactant = Chem.CombineMols(combined_reactant, aligned_reactants[i])

    img = Draw.MolsToGridImage(
        [combined_reactant, product],
        molsPerRow=2,
        subImgSize=(args.size, args.size),
    )

    reactant_img = img.crop((0, 0, args.size, args.size))
    product_img = img.crop((args.size, 0, args.size * 2, args.size))
    return reactant_img, product_img


def draw_molecule(smiles: str, size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    img = Draw.MolToImage(mol, size=size)
    return img


def draw_unaligned_molecules(reactant_smiles_raw: str, product_smiles: str) -> tuple[Image.Image, Image.Image] | tuple[None, None]:
    product_img = draw_molecule(product_smiles, size=(args.size, args.size))
    reactant_img = draw_molecule(reactant_smiles_raw, size=(args.size, args.size))
    return reactant_img, product_img


def gen_one(item: dict, image_dir: str) -> dict | None:
    reactant_smiles_raw = item["reactants"]
    product_smiles = item["product"]

    try:
        reactant_img, product_img = draw_aligned_molecules(reactant_smiles_raw, product_smiles)
        if reactant_img is None or product_img is None:
            return None
    except:
        return None

    reactant_image_filename = f"{item['idx']:08d}_reactant.png"
    product_image_filename = f"{item['idx']:08d}_product.png"

    reactant_img.save(os.path.join(image_dir, reactant_image_filename))
    product_img.save(os.path.join(image_dir, product_image_filename))

    return {
        "idx": item["idx"],
        "reactant_image": reactant_image_filename,
        "product_image": product_image_filename,
        "reactant": reactant_smiles_raw,
        "product": product_smiles,
    }


def gen_data(data_list: list, out_path: str):
    results = []

    image_dir = os.path.join(out_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(gen_one, item, image_dir): item for item in data_list}
        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, desc=os.path.basename(out_path)):
            result = future.result()
            if result is None:
                continue
            results.append(result)

    if not results:
        return

    output_file = os.path.join(out_path, f"data.csv")
    with open(output_file, "w", newline="") as f:
        fieldnames = ["idx", "reactant_image", "product_image", "reactant", "product"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)


def clean_atom_mapping(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def prepare_raw_data(file_path: str) -> list:
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        data_list = [row for row in reader]

    data = [item["reactants>reagents>production"] for item in data_list]
    data = list(set(data))  # Remove duplicates

    results = []

    for idx, item in enumerate(tqdm(data, dynamic_ncols=True, desc=os.path.basename(file_path))):
        reactants, _, product = item.split(">")
        reactants = clean_atom_mapping(reactants)
        product = clean_atom_mapping(product)

        results.append({
            "idx": idx,
            "product": product,
            "reactants": reactants,
        })

    return results


def main():
    train_data = prepare_raw_data("./raw/uspto50k_train.csv")
    test_data = prepare_raw_data("./raw/uspto50k_test.csv")
    val_data = prepare_raw_data("./raw/uspto50k_val.csv")

    os.makedirs("./processed", exist_ok=True)
    gen_data(train_data + val_data, "./processed/train")
    gen_data(test_data, "./processed/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    main()
