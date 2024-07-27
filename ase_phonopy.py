import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp  # Example calculator, replace with the actual calculator you use
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.io import read
from phonopy.units import VaspToTHz
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from ase.dft.kpoints import  get_special_points
import os
import matplotlib.pyplot as plt
from ase.io.lammpsdata import write_lammps_data


class PhononCalculation:
    def __init__(self, atoms: Atoms, supercell_matrix: list, displacement_distance: float, calc_type= 'vasp'):
        self.atoms = atoms
        self.supercell_matrix = supercell_matrix
        self.displacement_distance = displacement_distance
        self.phonon = None
        self.supercells = None
        self.fc = False
        self.mesh = False
        self.calc_type = calc_type

    def ase_to_phonopy(self):
        phonopy_atoms = PhonopyAtoms(symbols=self.atoms.get_chemical_symbols(),
                                     scaled_positions=self.atoms.get_scaled_positions(),
                                     cell=self.atoms.get_cell())
        return phonopy_atoms

    def phonopy_to_ase(self, phonopy_atoms):
        ase_atoms = Atoms(symbols=phonopy_atoms.get_chemical_symbols(),
                          positions=phonopy_atoms.get_positions(),
                          cell=phonopy_atoms.get_cell(),
                          pbc=True)
        return ase_atoms
    
    def setup_phonon(self, calculator=None):
        if self.calc_type == 'vasp':
            self.setup_phonon_vasp(calculator)
        elif self.calc_type == 'lammps':
            self.setup_phonon_lammps()
        else:
            raise ValueError('Unsupported calculator type. Supported types: "vasp", "lammps".')

    def setup_phonon_vasp(self, calculator):
        phonopy_atoms = self.ase_to_phonopy()
        self.phonon = Phonopy(phonopy_atoms, supercell_matrix=self.supercell_matrix, factor=VaspToTHz)
        self.phonon.generate_displacements(distance=self.displacement_distance)
        self.supercells = self.phonon.supercells_with_displacements

        for i, su in enumerate(self.supercells):
            ase_atoms = self.phonopy_to_ase(su)
            if calculator is None:
                raise ValueError('Calculator not provided')
            ase_atoms.set_calculator(calculator)
            directory = f'./displacement-{i}'
            os.makedirs(directory, exist_ok=True)
            calculator.set(directory=directory)
            calculator.write_input(ase_atoms)
        
        self.phonon.save("phonopy_disp.yaml")

    def setup_phonon_lammps(self):
        phonopy_atoms = self.ase_to_phonopy()
        self.phonon = Phonopy(phonopy_atoms, supercell_matrix=self.supercell_matrix, factor=VaspToTHz)
        self.phonon.generate_displacements(distance=self.displacement_distance)
        self.supercells = self.phonon.supercells_with_displacements
        self.phonon.save("phonopy_disp.yaml")

        for i, su in enumerate(self.supercells):
            ase_atoms = self.phonopy_to_ase(su)
            directory = f'./displacement-{i}'
            os.makedirs(directory, exist_ok=True)
            write_lammps_data(f'{directory}/data.lammps', ase_atoms, specorder=None, atom_style='atomic', masses=True)

                
    def calculate_fc(self):
        if self.calc_type == 'vasp':
            self.calculate_fc_vasp()
        elif self.calc_type == 'lammps':
            self.calculate_fc_lammps()
        else:
            raise ValueError('Unsupported calculator type. Supported types: "vasp", "lammps".')

    def calculate_fc_vasp(self):
        sets_of_forces = []
        for i, su in enumerate(self.supercells):
            directory = f'./displacement-{i}'
            if os.path.exists(f'{directory}/OUTCAR'):
                atoms = read(f'{directory}/OUTCAR', format='vasp-out')
                forces = atoms.get_forces()
                sets_of_forces.append(forces)
            else:
                raise FileNotFoundError(f'OUTCAR file does not exist in {directory}')

        self.phonon.forces = sets_of_forces
        self.phonon.produce_force_constants()
        self.phonon.save(settings={'force_constants': True})
        self.force_constants_calculated = True
            
    def calculate_fc_lammps(self):
        sets_of_forces = []
        for i, su in enumerate(self.supercells):
            directory = f'./displacement-{i}'
            if os.path.exists(f'{directory}/lmp.dump'):
                atoms = read(f'{directory}/lmp.dump', format='lammps-dump-text', index=-1)
                forces = atoms.get_forces()
                sets_of_forces.append(forces)
            else:
                raise FileNotFoundError(f'lmp.dump file does not exist in {directory}')

        self.phonon.forces = sets_of_forces
        self.phonon.produce_force_constants()
        self.phonon.save(settings={'force_constants': True})
        self.force_constants_calculated = True

        
    def _run_mesh(self, mesh=[20,20,20]):
        self.phonon.run_mesh( with_eigenvectors=True, is_mesh_symmetry=False)
        mesh_dict = self.phonon.get_mesh_dict()
        qpoints = mesh_dict['qpoints']
        weights = mesh_dict['weights']
        frequencies = mesh_dict['frequencies']
        eigenvectors = mesh_dict['eigenvectors']
        # best way to save it in single file
        # check if a folder exists with name mesh_data if not create one
        if not os.path.exists('mesh_data'):
            os.makedirs('mesh_data')
        np.save('mesh_data/qpoints.npy', qpoints)
        np.save('mesh_data/weights.npy', weights)
        np.save('mesh_data/frequencies.npy', frequencies)
        np.save('mesh_data/eigenvectors.npy', eigenvectors)
        self.mesh = True
        
    def phonon_dos(self):
        if not self.fc:
            self.calculte_fc()
        if not self.mesh:
            self._run_mesh()
        self.phonon.run_total_dos()
        self.phonon.run_projected_dos()
        total_dos_dict = self.phonon.get_total_dos_dict()
        projected_dos_dict = self.phonon.get_projected_dos_dict()
        frequencies_tot, total_dos = total_dos_dict['frequencies'], total_dos_dict['total_dos']
        frequencies_proj, projected_dos = projected_dos_dict['frequencies'], projected_dos_dict['projected_dos']

        if not os.path.exists('dos_data'):
            os.makedirs('dos_data')
        np.save('dos_data/frequencies_tot.npy', frequencies_tot)
        np.save('dos_data/total_dos.npy', total_dos)
        np.save('dos_data/frequencies_proj.npy', frequencies_proj)
        np.save('dos_data/projected_dos.npy', projected_dos)
    
    def phonon_band_structure(self, path_string=None):
        if not self.fc:
            self.calculate_fc()
        if not self.mesh:
            self._run_mesh()
            
        special_points = get_special_points(cell=atoms.get_cell())

        split_path = path_string.split(',')
        labels = []
        
        path = []

        # Loop through each segment in the path string
        for p in split_path:  
            tmp_list = []
            for point in p:
                labels.append(point)
                if point in special_points:
                    tmp_list.append(special_points[point].tolist())
            path.append(tmp_list)
        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
        self.phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)   
        self.phonon.write_yaml_band_structure(filename='band.yaml')     
        frequencies = self.phonon.get_band_structure_dict()['frequencies']
        qpoints = self.phonon.get_band_structure_dict()['qpoints']
        distances = self.phonon.get_band_structure_dict()['distances']
        eigenvectors = self.phonon.get_band_structure_dict()['eigenvectors']
        # make a phonon_data file and save everything there
        if not os.path.exists('phonon_data'):
            os.makedirs('phonon_data')
        np.save('phonon_data/frequencies.npy', frequencies)
        np.save('phonon_data/qpoints.npy', qpoints)
        np.save('phonon_data/distances.npy', distances)
        np.save('phonon_data/eigenvectors.npy', eigenvectors)
        np.save('phonon_data/labels.npy', labels)
        # self.phonon.auto_band_structure(plot=True).show()

