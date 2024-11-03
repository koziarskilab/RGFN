# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License

import abc
import os
from multiprocessing import Pool
from typing import List, Optional

import gin
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom


class BasePreparator(abc.ABC):
    """
    Base Preparator class.

    Args:
        conformer_attempts: Should an embedding attempt fail, how many times to retry conformer generation.
        n_conformers: How many conformers to generate per SMILES string.
    """

    def __init__(
        self,
        conformer_attempts: int,
        n_conformers: int,
    ):
        self.conformer_attempts = conformer_attempts
        self.n_conformers = n_conformers
        self._check_install()

    @abc.abstractmethod
    def _check_install(self) -> None:
        """
        Checks if necessary dependencies are installed.
        """

    def __call__(self, smis: List[str], ligand_paths_by_smiles: List[List[str]]) -> List[bool]:
        """
        Prepares a SMILES string or list of SMILES strings.

        Args:
            smi: SMILES string or list of SMILES strings.
            ligand_path: Path to which ligand conformers to be docked are saved. Not user-specified.

        Returns:
            A list of booleans indicating whether preparation was successful.
        """
        ...

    @abc.abstractmethod
    def _prepare_ligand(self, smi: str, ligand_path: str, **kwargs) -> bool:
        """
        Prepares a single SMILES string.

        Args:
            smi: SMILES string.
            ligand_path: Path to which ligand conformers to be docked are saved. Not user-specified.

        Returns:
            A boolean indicating whether preparation was successful.
        """


@gin.configurable()
class MeekoLigandPreparator(BasePreparator):
    """
    Using Meeko to prepare ligand.
    """

    def __init__(
        self,
        conformer_attempts: int = 20,
        n_conformers: int = 1,
        num_cpus: Optional[int] = None,
        pH: float = 7.4,
    ):
        super().__init__(conformer_attempts, n_conformers)

        if num_cpus is None:
            self.num_cpus = len(os.sched_getaffinity(0))
        else:
            self.num_cpus = num_cpus
        self.pH = pH

    def _check_install(self) -> None:
        try:
            pass
        except ImportError:
            raise Exception("Meeko package isn't installed.")

    def __call__(self, smis: List[str], ligand_paths_by_smiles: List[List[str]]) -> List[bool]:
        """
        Returns:
            A list of booleans indicating whether new ligand file was created (True) or already exists (False).
        """

        # Create a multiprocessing pool
        with Pool(self.num_cpus) as pool:
            results = pool.starmap(
                self._prepare_ligand,
                [(smis[i], ligand_paths_by_smiles[i]) for i in range(len(smis))],
            )

        return results

    def _prepare_ligand(self, smi: str, ligand_path: List[str]) -> bool:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        for j in range(0, len(ligand_path), self.n_conformers):
            if any(os.path.exists(path) for path in ligand_path[j : j + self.n_conformers]):
                return False

        attempt = 0
        while attempt < self.conformer_attempts:
            attempt += 1

            try:
                # get the correct protomer given the pH
                obc = ob.OBConversion()
                obc.SetInAndOutFormats("smi", "smi")
                obmol = ob.OBMol()
                obc.ReadString(obmol, smi)
                obmol.CorrectForPH(self.pH)
                smi = obc.WriteString(obmol)

                # protonate
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)

                if self.n_conformers == 1:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(mol)
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(mol)
                    setup = mol_setups[0]
                    pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
                    with open(ligand_path[0], "w") as file:
                        file.write(pdbqt_string)

                else:
                    AllChem.EmbedMultipleConfs(mol, self.n_conformers, rdDistGeom.ETKDGv3())

                    if mol.GetNumConformers() == 0:
                        raise Exception(
                            "AllChem.EmbedMultipleConfs() failed to yield any valid conformers."
                        )

                    preparator = MoleculePreparation()
                    for i in range(mol.GetNumConformers()):
                        AllChem.UFFOptimizeMolecule(mol, confId=i)
                        mol_setups = preparator.prepare(mol, conformer_id=i)
                        setup = mol_setups[0]
                        pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
                        with open(ligand_path[i], "w") as file:
                            file.write(pdbqt_string)
                return True

            except Exception as e:
                print(f"Failed embedding attempt #{attempt} with error: '{e}'.")

        return False
