#!/usr/bin/python2

import sys
from Bio.PDB import *
from numpy import *
import re
import os

if len(sys.argv)<4:
	print "Usage: xyz2fullpdb input.xyz output.pdb structure.pdb [firstframe lastframe]"
	exit(1)

trajname=sys.argv[1]
outputname=sys.argv[2]
PDBname=sys.argv[3]
if len(sys.argv)>4:
	firstframe=int(sys.argv[4])
	lastframe=int(sys.argv[5])
else:
	firstframe=0
	lastframe=100

print "PDB ",PDBname
print "XYZ Trajectory", trajname
print "PDB Trajectory", outputname
print "Fisrt frame",firstframe
print "Last frame",lastframe


def readframexyz(f,atoms):
	f.readline();f.readline();	
	k=[double(array(re.findall(r'[-+]?[0-9]*\.?[0-9]+e?[+-]?[0-9]{0,3}',f.readline()))) for i in range(0,atoms)]
	a=array(k)
	return a;
	
def writeframexyz(f,atoms):
	f.write("%d\n" % len(atoms))
	f.write("Atoms\n")
	for a in atoms:
		f.write("%s %f %f %f\n" % ('C', a[0], a[1], a[2]))

#Protein residue names
resnames=["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "SER", "THR", "ASN", "GLN", "TYR", "TRP", "ASP", "GLU", "HSE", "HSD", "HIS", "LYS", "ARG", "CYS"]

#Backbone atoms
BBA=["CA","N","C","O","HA","HN","1H","2H","3H","H","2HA","HA3","HT1","HT2","HT3","OT1","OT2","OXT"]

#RNA residue names
rnanames=["A","U","G","C"]
#PHA=["OP1","OP2","P","OP3","O5\'"]
SUA=["C1\'","C2\'","C3\'","C4\'","C5\'","O4\'","H1\'","H2\'","H3\'","H4\'","H5\'","H5\'\'","O2\'","HO2\'"]
BAA=["N1","C2","N3","C4","C5","C6","N2","N6","N7","C8","N9","N4","O4","O2","O6","H1","H2","H21","H22","H3","H41","H42","H5","H6","H61","H62","H8"]

parser=PDBParser()
structure=parser.get_structure('rnap',PDBname)

#Collect positions of CA and sidechain center-of-mass.
cas=[]
casv=[]
cbs=[]
cbsv=[]
seq=[]
phs=[];phsv=[];sus=[];susv=[];bas=[];basv=[];
# for model in structure:
for chain in structure[0]:
	if chain.get_list()[0].get_resname().strip() in resnames:
		for residue in chain:
			if not residue.get_resname().strip() in resnames:
				print "Warning: non-AA ",residue.get_resname().strip()," in chain ",chain
				break;
			seq.append(residue.get_resname())
			ca=residue['CA']
			cas.append(list(ca.get_vector()))
			casv.append(ca.get_vector())
			if residue.get_resname().strip()!="GLY":
				cm=Vector(0,0,0);m=0;
				for atom in residue:
					if not atom.get_name() in BBA:
						cm+=atom.get_vector().left_multiply(atom.mass)
						m+=atom.mass
				cm/=m
				cbg=cm	
			else:
				cbg=ca.get_vector()+Vector(0.5,0,0);
			cbs.append(list(cbg))
			cbsv.append(cbg)

	if chain.get_list()[0].get_resname().strip() in rnanames:
		for residue in chain:
			if not residue.get_resname().strip() in rnanames:
				print "Warning: non-nucleotide ",residue.get_resname().strip()," in chain ", chain
				break;
			p=residue['P']
			phs.append(p.get_coord());phsv.append(p.get_vector())
			sm=0;bm=0;
			scm=zeros(3);bcm=zeros(3)
			for atom in residue:
				if atom.get_name() in SUA:
					scm+=atom.get_coord()*atom.mass
					sm+=atom.mass
				if atom.get_name() in BAA:
					bcm+=atom.get_coord()*atom.mass
					bm+=atom.mass
			bcm/=bm;scm/=sm;
			sus.append(scm);susv.append(Vector(scm))
			bas.append(bcm);basv.append(Vector(bcm))

#structurePDB=structure.copy()
	
Naa=len(cas); #Number of residues
Nnuc=len(phs)

f=open(trajname)
io=PDBIO()
os.system('rm '+outputname)
#skip to first frame
for i in range((2*Naa+3*Nnuc+2)*firstframe):
	f.readline()
Nframes=lastframe-firstframe;
for i in range(Nframes):
	parser=PDBParser()
	structure=parser.get_structure('rnap',PDBname)
	io.set_structure(structure)
	print "Frame %d/%d\r" % (i,Nframes)
	r=readframexyz(f,2*Naa+3*Nnuc)
	i=0;j=0
	for chain in structure[0]:
		if chain.get_list()[0].get_resname().strip() in resnames:
			for residue in chain:
				if not residue.get_resname().strip() in resnames:
					#print "Warning: non-AA ",residue.get_resname().strip()," in chain ",chain
					break;
				for atom in residue:
					atom.set_coord(atom.get_coord()-cas[i])
				rotation=rotmat(cbsv[i]-casv[i],Vector(r[Naa+i]-r[i]))
				for atom in residue:
					if not atom.get_name() in BBA:
						atom.set_coord(rotation.dot(atom.get_coord()))
				for atom in residue:
					atom.set_coord(atom.get_coord()+r[i])
				i+=1
		if chain.get_list()[0].get_resname().strip() in rnanames:
			for residue in chain:
				if not residue.get_resname().strip() in rnanames:
					#print "Warning: non-nucleotide ",residue.get_resname().strip()," in chain ", chain
					break;
				for atom in residue:
					atom.set_coord(atom.get_coord()-phs[j])
				rotation=rotmat(susv[j]-phsv[j],Vector(r[2*Naa+Nnuc+j]-r[2*Naa+j]))
				for atom in residue:
					if atom.get_name() in SUA or atom.get_name()=="O3\'":
						atom.set_coord(rotation.dot(atom.get_coord()))
				rotation=rotmat(basv[j]-phsv[j],Vector(r[2*Naa+2*Nnuc+j]-r[2*Naa+j]))
				for atom in residue:
					if atom.get_name() in BAA:
						atom.set_coord(rotation.dot(atom.get_coord()))
				for atom in residue:
					atom.set_coord(atom.get_coord()+r[2*Naa+j])
				j+=1
					
	#io.set_structure(structure)
	io.save('1frame.pdb')
	os.system('cat 1frame.pdb>>'+outputname)
	#structure=structurePDB.copy()
	

