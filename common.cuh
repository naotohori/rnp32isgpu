//
//  common.cuh
//  
//
//  Created by Pavel Zhuravlev on 5/8/14.
//
//

#ifndef _common_cuh
#define _common_cuh

#include<stdio.h>
#include "cuda.h"

int BLOCK_SIZE=512;
int const MaxBondsPerAtom=4;
int const MaxNCPerAtom=128;
int const MaxNeighbors=1024;
int const MaxSoftSphere=8192;

float h=0.05;
float zeta=50.;
float kT=0.59;

float NumSteps=3e+4;
int const StartingStateEquilSteps=20000;//1000000;
int const SwitchingSteps=10000;//10000;
int const SwitchingStride=10;

int neighfreq=10;
int outputfreq=1000;
int trajfreq=10000;
int ntraj=1;
int seed=1234;

struct bond { //For bond map
    int i2;
    float l0;
    float k;
    float R0;
};

struct angle { //For angle map
    int iv;
    int i1;
    int i2;
    float a0;
    float k;
};

struct stack {
    int iP1;
    int iS1;
    int iB1;
    int iP2;
    int iS2;
    int iB2;
    int iP3;
    float U0;
    float kl;
    float kphi1;
    float kphi2;
    float l0;
    float phi10;
    float phi20;
};

struct hydrogenbond {
    int i1;          // Definition of particle ID in hydrogenbond
    int i3;          //
    int i5;          //  Nucleotide A    Nucleotide B
    int i2;          //    (5,3,1)          (2,4,6)
    int i4;          //                                !           
    int i6;          //      5                         !           
    float U0;        //       \                        !
    float kl;        //        3 -- 1 === 2 -- 4       !           
    float ktheta1;   //                         \      !           
    float ktheta2;   //                          6     !
    float kpsi;
    float kpsi1;
    float kpsi2;
    float l0;
    float theta10;
    float theta20;
    float psi0;
    float psi10;
    float psi20;
};

struct __align__(16) nc { //For native contact map
    int i2;
    float r02;
    float factor; //12*epsilon/r0^2
    float epsilon;
};


struct BrDynPar {
	float kT;
	//float h;
	//float zeta;
	float hoz; //timestep over friction constant
	float Gamma;
};

BrDynPar bd_h;
__device__ __constant__ BrDynPar bd_c;

struct SoftSphere {
    static float const eps=1.;
    float Minus6eps;
    static float const Rcut=22.; //For neighbor list, 22 A
    static float const RcutOuter=66.; //For outer neighbor list
    static float const CutOffFactor=3.; // Interaction cut off at 3 sigma
    float Rcut2;
    float Rcut2Outer;
    float CutOffFactor2inv;
    float CutOffFactor6inv;
    float CutOffFactor8inv;
    float MaxSigma;
};

SoftSphere ss_h;
__device__ __constant__ SoftSphere ss_c;

struct ElStatPar {
    static float const kappainv=2.4; // Angstrom, from Changbong's 2006 PNASs
    //static float const prefactor=330.4; // kcal/mol/A = 1 Coulomb^2/4*pi*eps0 = 1 statCoulomb^2
    //static float const dielectricepsilon=10.; // from Changbong's 2006 PNASs
    static float const prefactor=33.04;
   // static float const Rcut=5.;
};

ElStatPar els_h;
__device__ __constant__ ElStatPar els_c;

struct FENE {
    static float const k=20.;
    static float const R0=2.;
    float R02;  //R0^2
};

FENE fene_h;
__device__ __constant__ FENE fene_c;

int chainstarts_h[100];
__device__ __constant__ int chainstarts_c[100];


texture<float4, 1, cudaReadModeElementType> r_t;
texture<float, 1, cudaReadModeElementType> sig_t;
//texture<int, 1, cudaReadModeElementType> neibmap_t;

void checkCUDAError(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
        printf("CUDA error: %s: %s.\n", msg, cudaGetErrorString(error));
        exit(0);
	}
}


#endif
