# Implicit Geometry

_Note: this is simply a for fun mini-project_

This approach showcases that it’s possible to get 'some estimate' of the 3D geometry of a molecule based solely on its connectivity (and specially for some special molecules), without requiring any physics other than knowing standard bond lengths between atom pairs. These bond lengths can be retrieved from the NIST database https://cccbdb.nist.gov/diatomicexpbondx.asp.

When predicting the 3D coordinates for a system of NN atoms, the system has 3N degrees of freedom (corresponding to the x,y,z coordinates of each atom). To eliminate the global translation, we fix one atom’s position at (x1,y1,z1)=(0,0,0). To remove rotational degrees of freedom, we place a second atom along the x-axis at (x2,y2,z2)=(d12,0,0), where d12​ is the bond length between atoms 1 and 2. Additionally, we remove reflectional degrees of freedom by setting the third atom in the xy-plane at (x3,y3,z3)=(d13,d23,0). After applying these constraints, the system retains 3N−6 degrees of freedom.

* The existence of any bond between two atoms introduces a constraint, as we know that their distance is approximately equal to the bond length. Each non-zero element in the adjacency matrix corresponds to a bond and thus imposes a distance constraint. With a sufficient number of such constraints, the system is fully determined.

* Even if the system is not fully constrained by all non-zero elements in the adjacency matrix, the zeros in the matrix still provide useful information. These "non-bonds" imply a loose constraint: if two atoms are not directly connected, their distance must exceed the bond length.

Given a tuple of atomic symbols ($S_1$,$S_2$,...,$S_n$) and a NxN adjacency matrix $A_{ij}$ with elements $a_{ij}$ solve:

$\{(x_{i}^{*},x_{i}^{*},z_{i}^{*})\}^{N} = argmin_{\{(x_{i},x_{i},z_{i})\}^{N}} \sum_{i=1}^{N} \sum_{j=1,j\neq i}^{N} \left[\sigma \left(b_{ij}-d_{ij}\right)-a_{ij}\right]$

![](figures/test_fullerene.gif)


## Installation

_Note: Developed with Python 3.12.3_

1. Create virtual environment
```bash
python3 -m venv ~/python-envs/impgeom
```
2. Activate virtual environment
```bash
source ~/python-envs/impgeom/bin/activate
```
3. Install requirements
```bash
python3 -m pip install -r requirements.txt
```
## Usage

Run the script with the following command:  

```bash
python main.py --smiles 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```

### Options
| **Option**       | **Type**         | **Description**                                                                                          | **Default**           |
|-------------------|------------------|----------------------------------------------------------------------------------------------------------|-----------------------|
| `--smiles`       | *string*         | The SMILES string representing the molecule.                                                            | SMILES for caffeine   |
| `--addHs`        | *flag*           | If provided, the process will include explicit Hydrogen atoms.                                           | Not included          |
| `--max_iters`    | *int*            | The maximum number of optimization iterations.                                                          | 1000                  |
| `--plot_freq`    | *int*            | Plot a frame of the process every `x` iterations.                                                       | 500    |

### Examples

1. **Run with a custom SMILES string and default options:**
   ```bash
   python main.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
   ```

2. **Run with explicit Hydrogens and a custom iteration limit:**
   ```bash
   python main.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --addHs --max_iters 500
   ```

3. **Run with plotting every 50 iterations:**
   ```bash
   python main.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --plot_freq 50
   ```

## Gallery
![](figures/test_large.gif)