#!/usr/bin/python2

from Bio.PDB import *
from numpy import *
import re
import sys
import math

#temp_kT = 0.59
temp_kT = 0.64216500491
#temp_kT = 0.59

kB = 0.0019872041
temp_T = temp_kT / kB

bond_k  = 20.0
bond_R0 =  2.0
DTRNA_bond_k_PS = 23.0
DTRNA_bond_k_SB = 10.0
DTRNA_bond_k_SP = 64.0
DTRNA_angle_k_PSB =  5.0
DTRNA_angle_k_PSP = 20.0
DTRNA_angle_k_BSP =  5.0
DTRNA_angle_k_SPS = 20.0
DTRNA_st_k_l = 1.4
DTRNA_st_k_phi = 4.0

DTRNA_st_param = { #       h       s         Tm
                 "AA":  (4.348,  -0.319,   298.9),
                 "AC":  (4.311,  -0.319,   298.9),
                 "AG":  (5.116,   5.301,   341.2),
                 "AU":  (4.311,  -0.319,   298.9),
                 "CA":  (4.287,  -0.319,   298.9),
                 "CC":  (4.015,  -1.567,   285.8),
                 "CG":  (4.602,   0.774,   315.5),
                 "CU":  (3.995,  -1.567,   285.8),
                 "GA":  (5.079,   5.301,   341.2),
                 "GC":  (5.075,   4.370,   343.2),
                 "GG":  (5.555,   7.346,   366.3),
                 "GU":  (4.977,   2.924,   338.2),
                 "UA":  (4.287,  -0.319,   298.9),
                 "UC":  (3.992,  -1.567,   285.8),
                 "UG":  (5.032,   2.924,   338.2),
                 "UU":  (3.370,  -3.563,   251.6)  }

DTRNA_st_dist = {"AA":  4.1806530 , "AC":  3.8260185 , "AG":  4.4255305 , "AU":  3.8260185 ,
                 "CA":  4.7010580 , "CC":  4.2500910 , "CG":  4.9790760 , "CU":  4.2273615 ,
                 "GA":  4.0128560 , "GC":  3.6784360 , "GG":  4.2427250 , "GU":  3.6616930 ,
                 "UA":  4.7010580 , "UC":  4.2679180 , "UG":  4.9977560 , "UU":  4.2453650 }

DTRNA_st_dih_PSPS = -148.215 / 180.0 * math.pi
DTRNA_st_dih_SPSP = 175.975 / 180.0 * math.pi 

NN = ["AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU",
      "GA", "GC", "GG", "GU", "UA", "UC", "UG", "UU"]

DTRNA_st_U0 = {}
for n in NN:
    h  = DTRNA_st_param[n][0]
    s  = DTRNA_st_param[n][1]
    Tm = DTRNA_st_param[n][2]
    DTRNA_st_U0[n] = -h + kB * (temp_T - Tm) * s

# if len(sys.argv)<4:
#   print "Usage: initialstructure.pdb finalstructure.pdb inputfile.sopscgpu"
#   exit(1)
# outputname=sys.argv[3]
# PDBname2=sys.argv[2]
# PDBname=sys.argv[1]

if len(sys.argv)<2:
    print "Usage: structure.pdb inputfile.sopscgpu"
    exit(1)
outputname=sys.argv[2]
PDBname=sys.argv[1]

print "PDB 1 ",PDBname
#print "PDB 2 ",PDBname2
print "SOP-SC GPU input file ", outputname

#Protein residue names
resnames=["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "SER", "THR", "ASN", "GLN", "TYR", "TRP", "ASP", "GLU", "HSE", "HSD", "HIS", "LYS", "ARG", "CYS"]

#Backbone atoms
BBA=["CA","N","C","O","HA","HN","1H","2H","3H","H","2HA","HA3","HT1","HT2","HT3","OT1","OT2","OXT"]


#RNA residue names
rnanames=["A","U","G","C"]
#PHA=["OP1","OP2","P","OP3","O5\'"]
SUA=["C1\'","C2\'","C3\'","C4\'","C5\'","O4\'","H1\'","H2\'","H3\'","H4\'","H5\'","H5\'\'","O2\'","HO2\'"]
BAA=["N1","C2","N3","C4","C5","C6","N2","N6","N7","C8","N9","N4","O4","O2","H1","H2","H21","H22","H3","H41","H42","H5","H6","H61","H62","H8"]



chargedres=["GLU","ASP","ARG","LYS","HIS"]
q=dict()
qlist=[-1,-1,-1,1,1] #respective charges in list charged
for i,res in enumerate(chargedres): 
    q[res]=qlist[i]

#dielectricepsilon=10.
#elstatprefactor=(4.8*4.8*6.02/4184.*1e+2)/dielectricepsilon #kcal/mol
 
#SOP-SC parameters. el is actually defined in the simulation code right now
ebb=0.55
ess=0.3
ebs=0.4

erna=0.7
eCG=2.5
eAU=2.
#el=1.
GoCut=8.
GoCutsq=GoCut**2


#Get CA and sidechain-center-of-mass positions (to lists cas and cbs) from the PDB structure. Get native contant and salt bridges lists
def pdb2rnp(structure,cas,casv,cbs,cbsv,phs,sus,bas,phsv,susv,basv,terres,rterres,seq,rseq,ncs,sbs,charges):
    # for model in structure:
    for chain in structure[0]:
        rnum=chain.get_list()[0].get_id()[1]
        if chain.get_list()[0].get_resname().strip() in resnames:
            for residue in chain:
                if not residue.get_resname().strip() in resnames:
                    print "Warning: non-AA ",residue.get_resname().strip()," in chain ",chain
                    break;
                if residue.get_id()[1]-rnum>1:
                    print "Warning: break in ", chain, " at ", residue, residue.get_id()[1]-rnum-1," residues missing"
                    terres.append(len(cas)-1)
                rnum=residue.get_id()[1]
                seq.append(residue.get_resname())
                ca=residue['CA']
                #cas.append(list(ca.get_vector()))
                cas.append(ca.get_coord())
                casv.append(ca.get_vector())
                #cm=Vector(0,0,0)
                m=0
                cm=zeros(3)
                for atom in residue:
                    if not atom.get_name() in BBA:
                        cm+=atom.get_coord()*atom.mass
                        #cm+=atom.get_vector().left_multiply(atom.mass)
                        m+=atom.mass
                    if (atom.get_name()=='CB') or (atom.get_name()=='HA1'):
                        cb=atom.get_coord()
                cm/=m
                #cbg=cm
                cbg=cm
                #print cb,cm
                #cbs.append(list(cbg))
                cbs.append(cbg)
                cbsv.append(Vector(cbg))
            terres.append(len(cas)-1);  #Terminal residues
        
        if chain.get_list()[0].get_resname().strip() in rnanames:
            for residue in chain:
                if not residue.get_resname().strip() in rnanames:
                    print "Warning: non-nucleotide ",residue.get_resname().strip()," in chain ", chain
                    break;
                if residue.get_id()[1]-rnum>1:
                    print "Warning: break in ", chain, " at ", residue, residue.get_id()[1]-rnum-1," residues missing"
                    rterres.append(len(phs)-1)
                rnum=residue.get_id()[1]
                rseq.append(residue.get_resname().strip())
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
            rterres.append(len(phs)-1);  #Terminal residues
        
        Naa=len(cas) #Number of protein residues
        Nnuc=len(phs) #Number of rna residues
        
#Native contacts and salt-bridges
    for i in range(Naa):
        print "Amino acid %d/%d\r" % (i,Naa)
        for j in range(i,Naa):
            if (j-i)>2:
                if ((casv[i]-casv[j]).normsq()<GoCutsq):
                    ncs.append([i,j,(casv[i]-casv[j]).norm(),ebb])
                if ((cbsv[i]-cbsv[j]).normsq()<GoCutsq):
                    ncs.append([i+Naa,j+Naa,(cbsv[i]-cbsv[j]).norm(),ess*fabs(BT[seq[i]][seq[j]]-.7)])
                if ((casv[i]-cbsv[j]).normsq()<GoCutsq):
                    ncs.append([i,j+Naa,(casv[i]-cbsv[j]).norm(),ebs])
                if ((cbsv[i]-casv[j]).normsq()<GoCutsq):
                    ncs.append([i+Naa,j,(cbsv[i]-casv[j]).norm(),ebs])
                if q.has_key(seq[i]) and q.has_key(seq[j]):
                    sbs.append([i+Naa,j+Naa,q[seq[i]]*q[seq[j]]])
                    
#RNA native contacts
    for i in range(Nnuc):

        print "Nucleotide %d/%d\r" % (i,Nnuc)

        charges.append([i,-1.0])
        
        for j in range(i,Nnuc):

            if (j-i)>2:
                if ((phsv[i]-phsv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i,2*Naa+j,(phsv[i]-phsv[j]).norm(),erna])
                if ((susv[i]-susv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i+Nnuc,2*Naa+j+Nnuc,(susv[i]-susv[j]).norm(),erna])
                if ((basv[i]-basv[j]).normsq()<GoCutsq):
                    if (rseq[i]=="A" and rseq[j]=="U") or (rseq[i]=="U" and rseq[j]=="A"):
                        ncs.append([2*Naa+i+2*Nnuc,2*Naa+j+2*Nnuc,(basv[i]-basv[j]).norm(),eAU])
                    elif (rseq[i]=="C" and rseq[j]=="G") or (rseq[i]=="G" and rseq[j]=="C"):
                        ncs.append([2*Naa+i+2*Nnuc,2*Naa+j+2*Nnuc,(basv[i]-basv[j]).norm(),eCG])
                    else:
                        ncs.append([2*Naa+i+2*Nnuc,2*Naa+j+2*Nnuc,(basv[i]-basv[j]).norm(),erna])
                    
                if ((phsv[i]-susv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i,2*Naa+j+Nnuc,(phsv[i]-susv[j]).norm(),erna])
                if ((susv[i]-phsv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i+Nnuc,2*Naa+j,(susv[i]-phsv[j]).norm(),erna])
                    
                if ((phsv[i]-basv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i,2*Naa+j+2*Nnuc,(phsv[i]-basv[j]).norm(),erna])
                if ((basv[i]-phsv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i+2*Nnuc,2*Naa+j,(basv[i]-phsv[j]).norm(),erna])
                    
                if ((basv[i]-susv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i+2*Nnuc,2*Naa+j+Nnuc,(basv[i]-susv[j]).norm(),erna])
                if ((susv[i]-basv[j]).normsq()<GoCutsq):
                    ncs.append([2*Naa+i+Nnuc,2*Naa+j+2*Nnuc,(susv[i]-basv[j]).norm(),erna])
                    
#RNA to protein native
        for j in range(Naa):
            if ((phsv[i]-casv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i,j,(phsv[i]-casv[j]).norm(),erna])
            if ((susv[i]-casv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i+Nnuc,j,(susv[i]-casv[j]).norm(),erna])
            if ((basv[i]-casv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i+2*Nnuc,j,(basv[i]-casv[j]).norm(),erna])
            if ((phsv[i]-cbsv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i,j+Naa,(phsv[i]-cbsv[j]).norm(),erna])
            if ((susv[i]-cbsv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i+Nnuc,j+Naa,(susv[i]-cbsv[j]).norm(),erna])
            if ((basv[i]-cbsv[j]).normsq()<GoCutsq):
                ncs.append([2*Naa+i+2*Nnuc,j+Naa,(basv[i]-cbsv[j]).norm(),erna])
            
            
    return 1




# Read the Betancourt-Thirumalai matrix (see Betancourt M. R. & Thirumalai, D. (1999). Protein Sci. 8(2),361-369. doi:10.1110/ps.8.2.361)
BT=dict()
f=open('tb.dat')
aas=re.split(' ',f.readline().strip())[1:]
for i in range(len(aas)):
    if not BT.has_key(aas[i]):
        BT[aas[i]]=dict();
    l=re.split(' ',f.readline().strip())[1:]
    for j in range(i,len(aas)):
        BT[aas[i]][aas[j]]=double(l[j])
        if not BT.has_key(aas[j]):
            BT[aas[j]]=dict()
        BT[aas[j]][aas[i]]=double(l[j])
f.close

#Read the van der Waals diameters of the side chains
sbb=3.8
sss=dict()
f=open('aavdw.dat')
for l in f:
    s=re.split(' ',l)
    sss[s[0]]=2.*double(s[1])
print sss

sphs=4.2
ssug=4.4
sbas=3.8
    

parser=PDBParser()

#Get CAs, CBs, native contacts and salt bridges for initial structure
structure=parser.get_structure('Starting',PDBname)
cas=[];casv=[];cbs=[];cbsv=[];phs=[];phsv=[];sus=[];susv=[];bas=[];basv=[];terres=[];rterres=[];seq=[];rseq=[];ncs=[];sbs=[];charges=[]
pdb2rnp(structure,cas,casv,cbs,cbsv,phs,sus,bas,phsv,susv,basv,terres,rterres,seq,rseq,ncs,sbs,charges);

# #Get CAs, CBs, native contacts and salt bridges for final structure
# structure=parser.get_structure('Final',PDBname2)
# cas2=[];casv2=[];cbs2=[];cbsv2=[];terres2=[];seq2=[];ncs2=[];sbs2=[]
# pdb2sop(structure,cas2,casv2,cbs2,cbsv2,terres2,seq2,ncs2,sbs2)   

print "Native Contacts in starting structure: ", len(ncs)
#print "Native Contacts in final structure: ", len(ncs2)
print "Salt bridges: ", len(sbs)

Naa=len(cas); #Number of aa residues
Nnuc=len(phs); #Number of nucleic acid residues
Nch=len(terres); #Number of protein chains
Nchr=len(rterres); #Number of RNA chains
Nb=2*Naa-Nch; #Number of bonds in SOP-SC. Each residue has two bonds, except for Nch terminal residues
#Nb=Naa-Nch; #Number of bonds in SOP. Each residue has a bond, except for Nch terminal residues 
Nhb=3*Nnuc-Nchr #Number of hamonic bonds
Nang=4*Nnuc-3*Nchr #Number of bond angles (RNA)
Nst = Nnuc -3*Nchr


f=open('start.xyz','w')
f.write("%d\nAtoms\n" % (2*Naa+3*Nnuc))
for i in range(Nnuc):
    f.write("P %f %f %f\n" % (phs[i][0],phs[i][1],phs[i][2]))
    f.write("S %f %f %f\n" % (sus[i][0],sus[i][1],sus[i][2]))
    f.write("B %f %f %f\n" % (bas[i][0],bas[i][1],bas[i][2]))
for i in range(Naa):
    f.write("CA %f %f %f\n" % (cas[i][0],cas[i][1],cas[i][2]))
    f.write("CB %f %f %f\n" % (cbs[i][0],cbs[i][1],cbs[i][2]))
f.close

#Output everything to sopsc-gpu input file
f=open(outputname,'w')
f.write("NumSteps 2e+5\n")
f.write("Timestep(h) 0.05\n")
f.write("Friction(zeta) 50.\n")
f.write("Temperature %f\n" % temp_kT)
f.write("NeighborListUpdateFrequency 10\n")
f.write("OutputFrequency 1000\n")
f.write("TrajectoryWriteFrequency 10000\n")
f.write("Trajectories 1\n")
f.write("RandomSeed 1234\n")
f.write("KernelBlockSize 512\n")

f.write("ProteinResidues\n")
f.write("%d\n" % Naa) #Number of amino acid residues
f.write("ProteinChains\n")
f.write("%d\n" % Nch)  #Number of protein chains
f.write("RNAResidues\n")
f.write("%d\n" % Nnuc) #Number of nucleic acid residues
f.write("RNAChains\n")
f.write("%d\n" % Nchr)  #Number of RNA chains
f.write("ChainsStart@\n")
#Chain starts
for ter in terres[:-1]:
    f.write("%d\n" % (ter+1))
for ter in rterres[:-1]:
    f.write("%d\n" % (ter+1))

f.write("Bonds\n")
f.write("%d\n" % Nb)  #Number of bonds

#Bonds
for i in range(Naa):
    f.write("%d %d %f %f %f\n" % (i,i+Naa,(casv[i]-cbsv[i]).norm(),bond_k,bond_R0))
    if not i in terres:
        f.write("%d %d %f %f %f\n" % (i,i+1,(casv[i]-casv[i+1]).norm(),bond_k,bond_R0))

f.write("HarmonicBonds\n")
f.write("%d\n" % Nhb)  #Number of harmonic bonds
        
for i in range(Nnuc):
    iP = 2*Naa+i
    iS = iP + Nnuc
    iB = iS + Nnuc
    f.write("%d %d %f %f %f\n" % (iP,iS,(phsv[i]-susv[i]).norm(), DTRNA_bond_k_PS,0.0))
    f.write("%d %d %f %f %f\n" % (iS,iB,(susv[i]-basv[i]).norm(), DTRNA_bond_k_SB, 0.0))
    if (phsv[i]-susv[i]).norm()>10:
        print i, susv[i], phsv[i]
    if (susv[i]-basv[i]).norm()>10:
        print i, susv[i], basv[i]
    if not i in rterres:
        f.write("%d %d %f %f %f\n" % (iS,iP+1,(susv[i]-phsv[i+1]).norm(), DTRNA_bond_k_SP,0.0))
        if (susv[i]-phsv[i+1]).norm()>10:
            print "Si->Pi+1",i, susv[i], phsv[i+1]

#Bond angles
f.write("HarmonicBondAngles\n")
f.write("%d\n" % Nang) #Number of bond angles
for i in range(Nnuc):
    iP = 2*Naa+i
    iS = iP + Nnuc
    iB = iS + Nnuc
    iPnext = iP + 1
    iSnext = iS + 1
    f.write("%d %d %d %f %f\n" % (iP,iS,iB, DTRNA_angle_k_PSB, calc_angle(phsv[i],susv[i],basv[i])))
    if not i in rterres:
        f.write("%d %d %d %f %f\n" % (iP,iS,iPnext, DTRNA_angle_k_PSP, calc_angle(phsv[i],susv[i],phsv[i+1])))
        f.write("%d %d %d %f %f\n" % (iB,iS,iPnext, DTRNA_angle_k_BSP, calc_angle(basv[i],susv[i],phsv[i+1])))
        f.write("%d %d %d %f %f\n" % (iS,iPnext,iSnext, DTRNA_angle_k_BSP, calc_angle(susv[i],phsv[i+1],susv[i+1])))

#Stack
f.write("Stack\n")
f.write("%d\n" % Nst)
for i in range(Nnuc):
    ## Check if stack exists between i and (i-1)
    if i==0:       # The first residue in the first chain (no stack)
        continue
    if (i-1)==0:   # The first stack in the first chain (not considered because dihedral can not be defined)
        continue
    if (i-1) in rterres:  # The first residue in other chains (no stack)
        continue
    if (i-2) in rterres:  # The first stack in other chains (not considered because dihedral can not be defined)
        continue
    if i in rterres:  # The last stack in each chain (not considered because dihedral can not be defined)
        continue
    iP1 = 2*Naa+(i-1)
    iS1 = iP1 + Nnuc
    iB1 = iS1 + Nnuc
    iP2 = 2*Naa+i
    iS2 = iP2 + Nnuc
    iB2 = iS2 + Nnuc
    iP3 = 2*Naa+(i+1)
    n = "%s%s" % (rseq[i-1], rseq[i])
    f.write("%d %d %d %d %d %d %d %f %f %f %f %f %f %f\n" % 
            (iP1, iS1, iB1, iP2, iS2, iB2, iP3, 
            DTRNA_st_U0[n], DTRNA_st_k_l, DTRNA_st_k_phi, DTRNA_st_k_phi,
            DTRNA_st_dist[n], DTRNA_st_dih_PSPS, DTRNA_st_dih_SPSP) )
            #(basv[i-1]-basv[i]).norm(), ## l0 
            #calc_dihedral(phsv[i-1], susv[i-1], phsv[i], susv[i]),   # phi01
            #calc_dihedral(phsv[i+1], susv[i], phsv[i], susv[i-1])) ) # phi02


#Native contacts of starting structure
f.write("%d\n" % len(ncs))
for nc in ncs:
    f.write("%d %d %f %f\n" % (nc[0],nc[1],nc[2],nc[3]))

# #Native contacts of final structure
# f.write("%d\n" % len(ncs2))
# for nc in ncs2:
#   f.write("%d %d %f %f\n" % (nc[0],nc[1],nc[2],nc[3]))

#Sigmas for soft sphere repulsion               
for aa in seq:
    f.write('%f\n' % sbb)
for aa in seq:
    f.write('%f\n' % sss[aa])
for i in range(Nnuc):
    f.write('%f\n' % sphs)
    f.write('%f\n' % ssug)
    f.write('%f\n' % sbas)

# #Exclusions from soft shpere interactions (additional to bonded beads: ss of neighboring residues, bs to ss of neighboring residues)
# f.write("%d\n" % (2*(Naa-0*Nch)))
# for i in range(Naa):
#   #f.write("%d %d\n" % (i,i+Naa))
#   #if not i in terres:
#       #f.write("%d %d\n" % (i,i+1))
#   f.write("%d %d %f\n" % (i,i+Naa+1,0))
#   f.write("%d %d %f\n" % (i+Naa,i+Naa+1,0))

#Salt bridges   
f.write("%d\n" % len(sbs))
for sb in sbs:
    f.write("%d %d %f\n" % (sb[0],sb[1],sb[2])) 

f.write("Charges\n")
f.write("%d\n" % len(charges))  #Number of charges
#Charges
for c in charges:
    f.write("%d %f\n" % (c[0],c[1]))

#Starting coordinates
if Naa>0:
    for ac in vstack([array(cas),array(cbs)]):
        f.write("%f %f %f\n" % (ac[0],ac[1],ac[2]))
if Nnuc>0:
    for ac in vstack([array(phs),array(sus),array(bas)]):
        f.write("%f %f %f\n" % (ac[0],ac[1],ac[2]))
    
f.close
