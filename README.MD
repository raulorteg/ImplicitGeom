# Implicit Geometry

The proposition is that one can get some guess of the 3D geometry of a molecule given its connectivity without any physical knowledge of the system other than the standard bond-lengths of pairs of atoms (Taken from https://cccbdb.nist.gov/diatomicexpbondx.asp).

## About 

Given a tuple of atomic symbols ($S_1$,$S_2$,...,$S_n$) and a NxN adjacency matrix $A_{ij}$ with elements $a_{ij}$ solve:

$\{(x_{i}^{*},x_{i}^{*},z_{i}^{*})\}^{N} = argmin_{\{(x_{i},x_{i},z_{i})\}^{N}} \sum_{i=1}^{N} \sum_{j=1,j\neq i}^{N} \left[\sigma \left(b_{ij}-d_{ij}\right)-a_{ij}\right]$

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
![](figures/test_c70.gif)
