//#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
#include <curand_kernel.h>
#include <string>
#include <iostream>

#include "common.cuh"
#include "InteractionList.cuh"
#include "kernels.cu"




void writexyz(FILE* traj, float4* r, int Naa);

void writexyz(FILE** traj, float4* r, int Naa, int Nnuc, int ntraj);

void writexyz(FILE* traj, float4* r, float3 t,int Naa);

void writeforces(FILE* traj, float4* r, int Naa);

void readcoord(FILE* ind, float4* r, int N);

void readcoord(FILE* ind, float4* r, int N, int ntraj);

void readxyz(FILE* ind, float4* r, int N);

void readxyz(FILE* ind, float4* r, int N, int ntraj);




int main(int argc, char *argv[]){
    
    if (argc<2) {
        std::string progname=argv[0];
        printf("Usage: %s inputfilename\n",progname.c_str());
        exit(1);
    }

    std::string mode="run";
    if (argc == 3) {
        mode = argv[2];
        if (mode != "check") {
            printf("unknown mode: %s\n", mode.c_str());
            exit(1);
        }
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaSetDevice(0);
    cudaDeviceReset();
    
    std::string filename=argv[1];
    
    FILE *ind;
    if((ind = fopen(filename.c_str(), "r"))==NULL) {
        printf("Cannot open file %s \n",filename.c_str()) ;
        exit(1) ;
    }

    
////////// READING INPUT FILE
    char comments[80];
    fscanf(ind,"%s %e",comments,&NumSteps);
    printf("%s %e\n",comments,NumSteps);
    fscanf(ind,"%s %f",comments,&h);
    printf("%s %f\n",comments,h);
    fscanf(ind,"%s %f",comments,&zeta);
    printf("%s %f\n",comments,zeta);
    fscanf(ind,"%s %f",comments,&kT);
    printf("%s %f\n",comments,kT);
    fscanf(ind,"%s %d",comments,&neighfreq);
    printf("%s %d\n",comments,neighfreq);
    fscanf(ind,"%s %d",comments,&outputfreq);
    printf("%s %d\n",comments,outputfreq);
    fscanf(ind,"%s %d",comments,&trajfreq);
    printf("%s %d\n",comments,trajfreq);
    fscanf(ind,"%s %d",comments,&ntraj);
    printf("%s %d\n",comments,ntraj);
    fscanf(ind,"%s %d",comments,&seed);
    printf("%s %d\n",comments,seed);
    fscanf(ind,"%s %d",comments,&BLOCK_SIZE);
    printf("%s %d\n",comments,BLOCK_SIZE);
    
    // Initialize trajectory output files
    FILE **traj;
    traj=(FILE**)malloc(ntraj*sizeof(FILE*));
    for (int itraj=0; itraj<ntraj; itraj++) {
        char itrajstr[3];
        sprintf(itrajstr, "%d", itraj);
        std::string trajfile=filename+"traj"+itrajstr+".xyz";
        if((traj[itraj] = fopen(trajfile.c_str(), "w"))==NULL) {
            printf("Cannot open file %s \n",trajfile.c_str()) ;
            exit(1) ;
        }
    }

    
    int Naa;    //Total number of amino acid residues
    fscanf(ind,"%s",comments);
    fscanf(ind,"%d",&Naa);
    printf("Number of amino acid residues: %d\n",Naa);
   
    int Nch;    //Number of protein chains
    fscanf(ind,"%s",comments);
    fscanf(ind,"%d",&Nch);
    printf("Number of protein chains: %d\n",Nch);
    
    int Nnuc;    //Total number of RNA residues
    fscanf(ind,"%s",comments);
    fscanf(ind,"%d",&Nnuc);
    printf("Number of RNA residues: %d\n",Nnuc);
    
    int Nchr;    //Number of RNA chains
    fscanf(ind,"%s",comments);
    fscanf(ind,"%d",&Nchr);
    printf("Number of RNA chains: %d\n",Nchr);
    
    fscanf(ind,"%s",comments);
    printf("%s\n",comments);
    chainstarts_h[0]=Nch+Nchr-2;
    for (int i=1; i<Nch+Nchr-1; i++) {
        int cstart;
        fscanf(ind,"%d",&cstart);
        chainstarts_h[i]=cstart;
        printf("%d\n",cstart);
    }
    cudaMemcpyToSymbol(chainstarts_c, &chainstarts_h, 100*sizeof(int), 0, cudaMemcpyHostToDevice);
    
    int N=(2*Naa+3*Nnuc)*ntraj; //Number of beads
    
// Read bonds and build map, allocate and copy to device
    int Nb; //Number of bonds
    fscanf(ind,"%s",comments);
    fscanf(ind,"%d",&Nb);
    InteractionListBond bondlist(ind,N/ntraj,MaxBondsPerAtom,Nb,"covalent bond",ntraj);

// Read native contacts and build map for initial structure, allocate and copy to device
    int Nnc;  //Number of native contacts (initial)
    fscanf(ind,"%d",&Nnc);
    InteractionListNC nclist(ind,N/ntraj,MaxNCPerAtom,Nnc,"native contact (starting)",ntraj);
    
// Read native contacts and build map for target structure, allocate and copy to device
//    int Nnc2;       //Number of native contacts (target)
//    fscanf(ind,"%d",&Nnc2);
//    InteractionListNC nclist2(ind,N/ntraj,MaxNCPerAtom,Nnc2,"native contact (target)",ntraj);
    
//Read sigmas for non-native and neighboring soft sphere repulsion
    printf("Reading sigmas\n");
    float *sig_h, *sig_d;
    sig_h=(float*)malloc(N*sizeof(float));
    cudaMalloc((void**)&sig_d,N*sizeof(float));
    ss_h.MaxSigma=0.;
    for (int i=0; i<N/ntraj; i++) {
        if (fscanf(ind,"%f", &sig_h[i])==EOF)
            printf("Premature end of file at line %d", i);
        for (int itraj=1; itraj<ntraj; itraj++)
            sig_h[itraj*N/ntraj+i]=sig_h[i];
        if (sig_h[i]>ss_h.MaxSigma)
            ss_h.MaxSigma=sig_h[i];
    }
    cudaMemcpy(sig_d, sig_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sig_t, sig_d, N*sizeof(float));
    
//// Read soft-sphere interaction exclusions (like side-chains of neigboring beads) and build map, allocate and copy to device
//    int Nexc; //Number of exclusions
//    fscanf(ind,"%d",&Nexc);
//    InteractionListBond ssel(ind,N/ntraj,MaxBondsPerAtom,Nexc,"additional soft-sphere exclusion",ntraj);

// Read salt bridges
    //Number of salt bridges
    int Nsb;
    fscanf(ind,"%d",&Nsb);
    InteractionListSB SaltBridgeList(ind,N/ntraj,MaxNeighbors,Nsb,"electrostatic interaction",ntraj);
    
//Allocate coordinates arrays on device and host
    float4 *r_h,*r_d;
    cudaMallocHost((void**)&r_h, N*sizeof(float4));
    cudaMalloc((void**)&r_d, N*sizeof(float4));
    
// Read starting coordinates
    printf("Reading initial coordinates\n");
    ////readcoord(ind, r_h, N);
    readcoord(ind, r_h, N/ntraj, ntraj);
    
////READ FROM SEPARATE FILE
//    FILE *initl;
//   std::string initlfilename="start.init";
//    if((initl = fopen(initlfilename.c_str(), "r"))==NULL) {
//        printf("Cannot open file %s \n",initlfilename.c_str()) ;
//        exit(1) ;
//    }
//    //readxyz(initl, r_h, N);
//    readxyz(initl, r_h, N/ntraj, ntraj);
//    fclose(initl);
    
    //Copy coordinates to device
    cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaBindTexture(0, r_t, r_d, N*sizeof(float4));
    
//Allocate forces arrays on device <and host>
    float4 *f_d;
    cudaMalloc((void**)&f_d, N*sizeof(float4));
    
    //float4 *f_h;
    //cudaMallocHost((void**)&f_h, N*sizeof(float4));

    fclose(ind);
//////////////END READING INPUT FILE//////


//Initialize Brownian Dynamics integrator parameters
    bd_h.kT=kT;
    bd_h.hoz=h/zeta;
    bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
    cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Brownian dynamics parameters init");
    
    
//Initialize Soft Sphere repulsion force field parameters;
    ss_h.Minus6eps=-6.0*ss_h.eps;
    ss_h.Rcut2=ss_h.Rcut*ss_h.Rcut;
    ss_h.Rcut2Outer=ss_h.RcutOuter*ss_h.RcutOuter;
    ss_h.CutOffFactor2inv=1.0f/ss_h.CutOffFactor/ss_h.CutOffFactor;
    ss_h.CutOffFactor6inv=ss_h.CutOffFactor2inv*ss_h.CutOffFactor2inv*ss_h.CutOffFactor2inv;
    ss_h.CutOffFactor8inv=ss_h.CutOffFactor6inv*ss_h.CutOffFactor2inv;
    cudaMemcpyToSymbol(ss_c, &ss_h, sizeof(SoftSphere), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Soft sphere parameters init");
    
//Initialize FENE parameters
    fene_h.R02=fene_h.R0*fene_h.R0;
    cudaMemcpyToSymbol(fene_c, &fene_h, sizeof(FENE), 0, cudaMemcpyHostToDevice);
    checkCUDAError("FENE parameters init");
    
//Initialize electrostatic parameters
    cudaMemcpyToSymbol(els_c, &els_h, sizeof(ElStatPar), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Electrostatic parameters init");
    

    
//Neighbor list allocate
    InteractionList<int> nl;
    nl.N=N;
    nl.Nmax=MaxSoftSphere;
    nl.AllocateOnDevice("neighbor list");
    nl.AllocateOnHost();
    //cudaBindTexture(0, neibmap_t, nl.map_d, nl.N*nl.Nmax*sizeof(int));
    
//    InteractionList<int> nlo;
//    nlo.N=N;
//    nlo.Nmax=MaxSoftSphere;
//    nlo.AllocateOnDevice("outer neighbor list");
//    nlo.AllocateOnHost();
    
    
    
//Simulation
    
    int THREADS=BLOCK_SIZE;
    int BLOCKS=N/THREADS+1;
    
    //Allocate and initialize random seeds
    curandStatePhilox4_32_10_t *RNGStates_d;
    cudaMalloc( (void **)&RNGStates_d, THREADS*BLOCKS*sizeof(curandStatePhilox4_32_10_t) );
    checkCUDAError("Brownian dynamics seeds allocation");
    rand_init<<<BLOCKS,THREADS>>>(seed,RNGStates_d);
    checkCUDAError("Random number initializion");


//Check consistency between force and energy (for debugging)
    if (mode == "check") {
        printf("\n");
        printf("Check consistency between force and energy (for debugging)\n");

        //Shift coordinates
        double small=0.1;
        cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);

        coord_shift<<<BLOCKS,THREADS>>>(r_d,small,N,RNGStates_d);
        checkCUDAError("coord_shift");

        cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);

        //Allocate forces arrays on host   (on device, f_d is already allocated)
        float4 *f_h;
        cudaMallocHost((void**)&f_h, N*sizeof(float4));

        //Neighbor list
        SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,N/ntraj);
        checkCUDAError("Neighbor List");

        //Force
        force_flush<<<BLOCKS,THREADS>>>(f_d,N);
        checkCUDAError("Force flush");
        
        FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
        checkCUDAError("FENE");
        
        SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
        checkCUDAError("SoftSphere");
        
        NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,sig_d);
        checkCUDAError("Native subtract Soft Sphere");
        
        NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist);
        checkCUDAError("Native");
        
        //DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
        //checkCUDAError("DebyeHuckel");
        
        cudaMemcpy(f_h, f_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
        checkCUDAError("Copy force back");

        // Store the original coordinates
        float4 *r_save;
        cudaMallocHost((void**)&r_save, N*sizeof(float4));
        cudaMemcpy(r_save, r_h, N*sizeof(float4), cudaMemcpyHostToHost);

        small=0.001;
        for (int imove=0; imove<N; imove++) {
            for (int idim=0; idim<3; idim++) {
            
                cudaMemcpy(r_h, r_save, N*sizeof(float4), cudaMemcpyHostToHost);
    
                if (idim==0) r_h[imove].x += small;
                if (idim==1) r_h[imove].y += small;
                if (idim==2) r_h[imove].z += small;
    
                //Copy coordinates to device
                cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);

                SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,N/ntraj);
                checkCUDAError("Neighbor List");
    
                // Energy (plus)
                FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Efene");
                       
                double Efene=0.0;
                for (int i=0; i<N; i++)
                    Efene+=r_h[i].w;
                        
                SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ess");
                    
                double Ess=0.0;
                for (int i=0; i<N; i++)
                    Ess+=r_h[i].w;
                        
                NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enss");
                        
                for (int i=0; i<N; i++)
                    Ess+=r_h[i].w;
                        
                NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enat");
                        
                double Enat=0.0;
                for (int i=0; i<N; i++)
                    Enat+=r_h[i].w;
                        
                //DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
                //cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                //checkCUDAError("Copy coordinates back for Eel");
                        
                double Eel=0.0;
                //for (int i=0; i<N; i++)
                //    Eel+=r_h[i].w;
    
                double Epot_plus=(Efene+Ess+Enat+Eel)/2.;
            
                if (idim==0) r_h[imove].x -= 2.0 * small;
                if (idim==1) r_h[imove].y -= 2.0 * small;
                if (idim==2) r_h[imove].z -= 2.0 * small;
    
                //Copy coordinates to device
                cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);

                SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,N/ntraj);
                checkCUDAError("Neighbor List");
    
                // Energy (minus)
                FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Efene");
                       
                Efene=0.0;
                for (int i=0; i<N; i++)
                    Efene+=r_h[i].w;
                
                SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ess");
                    
                Ess=0.0;
                for (int i=0; i<N; i++)
                    Ess+=r_h[i].w;
                        
                NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enss");
                        
                for (int i=0; i<N; i++)
                    Ess+=r_h[i].w;
                        
                        
                NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enat");
                        
                Enat=0.0;
                for (int i=0; i<N; i++)
                    Enat+=r_h[i].w;
                        
                //DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
                //cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                //checkCUDAError("Copy coordinates back for Eel");
                        
                Eel=0.0;
                //for (int i=0; i<N; i++)
                //    Eel+=r_h[i].w;
    
                double Epot_minus=(Efene+Ess+Enat+Eel)/2.;
    
                // Numerical derivative
                double ff = - (Epot_plus - Epot_minus) * 0.5 / small;
                
                // Compare to force
                printf("imove=%d\t", imove); 
                if (idim==0) {
                    printf("x\t%e\t%e\t%20.15f\n", f_h[imove].x, ff, f_h[imove].x - ff);
                } else if (idim==1) {
                    printf("y\t%e\t%e\t%20.15f\n", f_h[imove].y, ff, f_h[imove].y - ff);
                } else {
                    printf("z\t%e\t%e\t%20.15f\n", f_h[imove].z, ff, f_h[imove].z - ff);
                }

            } //idim
        } //imove

        cudaFree(r_d);
        cudaFree(f_d);
        nclist.FreeOnDevice("native contacts");
        bondlist.FreeOnDevice("bonds");
        SaltBridgeList.FreeOnDevice("salt bridges");
        nl.FreeOnDevice("neighbor list");
        //nlo.FreeOnDevice("outer neighbor list");
        cudaDeviceReset();

        exit(0);
    }


//Production run
   
    printf("t\tTraj#\tE_TOTAL\t\tE_POTENTIAL\tE_SoftSpheres\tE_NatCont\tE_ElStat\tE_FENE\t\t~TEMP\t<v>*neighfreq/DeltaRcut\n");
    //float Delta=0.;
    int stride=neighfreq;
    
    for (int t=0;t<NumSteps;t+=stride) {
        
        bool CoordCopiedToHost=false;
        
//        if ((t % (3*neighfreq))==0) {
//            SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nlo,bondlist,N/ntraj);
//            checkCUDAError("Outer Neighbor List");
//        }
        
        if ((t % neighfreq)==0) {
            //SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl);
            //SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,N/ntraj);
            SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,N/ntraj);
            //SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nlo,nl);
            checkCUDAError("Neighbor List");
        }
        
//        nl.CopyToHost("Neighbor List");
//        for (int i=0; i<N; i++) {
//            
//            int Nneib=nl.count_h[i];                                                  //Number of neighbors of the i-th bead
//            printf("%d, %d neibs: ",i,Nneib);
//            for (int ineib=0;ineib<Nneib;ineib++)                                    //Loop over neighbors of the i-th bead
//                printf("%d ",nl.map_h[ineib*nl.N+i]);
//            printf("\n");
//        }
        
        if ((t % outputfreq)==0) {
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ekin");
            CoordCopiedToHost=true;
            
            float* Ekin;
            Ekin=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ekin[itraj]+=r_h[i].w;
            }
            
            FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Efene");
            
            float* Efene;
            Efene=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Efene[itraj]+=r_h[i].w;
            }
            
            SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ess");
            
            float* Ess;
            Ess=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ess[itraj]+=r_h[i].w;
            }
            
            
            NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist,sig_d);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Enss");
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ess[itraj]+=r_h[i].w;
            }
            
            
            NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Enat");
            
            float* Enat;
            Enat=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Enat[itraj]+=r_h[i].w;
            }
            
            //DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
            //cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            //checkCUDAError("Copy coordinates back for Eel");
            
            float* Eel;
            Eel=(float*)calloc(ntraj,sizeof(float));
            //for (int itraj=0; itraj<ntraj; itraj++) {
            //    for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
            //        Eel[itraj]+=r_h[i].w;
            //}
            float* Epot;
            Epot=(float*)malloc(ntraj*sizeof(float));
            float* Etot;
            Etot=(float*)malloc(ntraj*sizeof(float));
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                Epot[itraj]=(Efene[itraj]+Ess[itraj]+Enat[itraj]+Eel[itraj])/2.;
                Etot[itraj]=Epot[itraj]+Ekin[itraj];
                printf("%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t%f\t%f\n",t,itraj,Etot[itraj],Epot[itraj],Ess[itraj]/2.,Enat[itraj]/2.,Eel[itraj]/2.,Efene[itraj]/2.,Ekin[itraj]*ntraj/(N*6.*bd_h.hoz/503.6),sqrt(Ekin[itraj]*ntraj/N)*neighfreq/(ss_h.Rcut-ss_h.MaxSigma*ss_h.CutOffFactor));
            }
            
        }
        
        if ((t % trajfreq)==0) {
            if (!CoordCopiedToHost) {
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back");
                CoordCopiedToHost=true;
            }
            writexyz(traj,r_h,Naa,Nnuc,ntraj);
            
        }
        
        for (int tongpu=0; tongpu<stride; tongpu++) {
            
            force_flush<<<BLOCKS,THREADS>>>(f_d,N);
            checkCUDAError("Force flush");
            
            FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
            checkCUDAError("FENE");
            
            SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
            checkCUDAError("SoftSphere");
            
            NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,sig_d);
            checkCUDAError("Native subtract Soft Sphere");
            
            NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist);
            checkCUDAError("Native");
            
            //DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
            //checkCUDAError("DebyeHuckel");
            
            integrate<<<BLOCKS,THREADS>>>(r_d,f_d,N,RNGStates_d);
            checkCUDAError("Integrate");
            
        }
        
    }
    

    
    cudaFree(r_d);
    cudaFree(f_d);
    nclist.FreeOnDevice("native contacts");
    bondlist.FreeOnDevice("bonds");
    SaltBridgeList.FreeOnDevice("salt bridges");
    nl.FreeOnDevice("neighbor list");
    //nlo.FreeOnDevice("outer neighbor list");
    cudaDeviceReset();
}

