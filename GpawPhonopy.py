import os
import shutil
import subprocess
from typing import Dict, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


class GPAWPhonopy:
    def __init__(self, atoms: Atoms, repeat= [2,2,2], calc: str = "gpaw") -> None:
        self.atoms = atoms
        self.dist_structures = {}
        self.phonon = None
        self.force_constants = None
        self.gamma_frequencies = None
        self.calc = calc
        self.repeat = repeat
        

    def ase_to_phonopy(self, atoms: Atoms, wrap: bool = False) -> PhonopyAtoms:
        phonopy_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            masses=atoms.get_masses(),
            positions=atoms.get_positions(wrap=wrap),
        )
        return phonopy_atoms

    def phonopy_to_ase(
        self, structure: PhonopyAtoms, info: dict = None, pbc: bool = True, db: bool = False
    ) -> Atoms:
        if info is None:
            info = {}

        if structure is None:
            return None

        atoms_dict = {
            "symbols": structure.get_chemical_symbols(),
            "cell": structure.get_cell(),
            "masses": structure.get_masses(),
            "positions": structure.get_positions(),
            "pbc": pbc,
            "info": info,
        }

        if db:
            del atoms_dict["masses"]
            db_dict = {
                "cell": np.round(structure.get_cell()),
                "positions": np.round(structure.get_positions()),
            }
            atoms_dict.update(db_dict)

        atoms = Atoms(**atoms_dict)

        return atoms

    def preprocess(self, atoms: Atoms, distance: float = 0.01) -> Tuple[Dict[str, Atoms], Phonopy]:
        phonopy_atoms = self.ase_to_phonopy(atoms)
        phonon = Phonopy(
            phonopy_atoms, supercell_matrix=np.diag(self.repeat), primitive_matrix="auto"
        )

        phonon.generate_displacements(distance=distance)
        supercells = phonon.get_supercells_with_displacements()

        for i, enumerated_supercell in enumerate(supercells, 1):
            filename = f"POSCAR-00{i}"
            dist_atoms = self.phonopy_to_ase(enumerated_supercell)
            self.dist_structures[filename] = dist_atoms
            self.phonon = phonon

        return self.dist_structures, self.phonon

    def run_phonons(self, run: str = "run.py", submit: str = "submit_ph.sh") -> None:
        if not (os.path.isfile(run) and os.path.isfile(submit)):
            raise FileNotFoundError("run.py or submit_ph.sh not found")

        for filename, atoms in self.dist_structures.items():
            dir_ = f"phonons/{filename}"
            os.makedirs(dir_, exist_ok=True)
            write(f"{dir_}/POSCAR", atoms, format="vasp")
            shutil.copyfile(run, f"{dir_}/{run}")
            shutil.copyfile(submit, f"{dir_}/{submit}")
            result = subprocess.run(
                ["sbatch", submit], cwd=dir_, capture_output=True, text=True
            )
            if result.returncode == 0:
                print("Command executed successfully.")
                print("Output:")
                print(result.stdout)
            else:
                print("Command execution failed.")
                print("Error:")
                print(result.stderr)

    def postprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        forces = []

        if self.calc == "gpaw":
            for filename in self.dist_structures.keys():
                dir_ = os.path.join("phonons", filename)
                if os.path.isfile(os.path.join(dir_, "gs.txt")):
                    gpaw_atoms = read(os.path.join(dir_, "gs.txt"), format="gpaw-out")
                    forces.append(gpaw_atoms.get_forces())
                else:
                    print(f"gs.txt not found in {dir_}")

            forces = np.array(forces)

        elif self.calc == "vasp":
            for filename in self.dist_structures.keys():
                dir_ = os.path.join("phonons", filename)
                if os.path.isfile(os.path.join(dir_, "OUTCAR")):
                    vasp_atoms = read(os.path.join(dir_, "OUTCAR"), format="vasp-out")
                    forces.append(vasp_atoms.get_forces())
                else:
                    print(f"OUTCAR not found in {dir_}")

            forces = np.array(forces)
        
        else:
            raise ValueError("calc must be either gpaw or vasp")
        self.phonon.set_forces(forces)
        self.phonon.produce_force_constants(calculate_full_force_constants=False)
        self.force_constants = self.phonon.get_force_constants()
        self.gamma_frequencies = self.phonon.get_frequencies(q=[0,0,0])

        return self.force_constants, self.gamma_frequencies

    def bandstructure(self) -> None:
        pass
        # TODO: add bandstructure function
