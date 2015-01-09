//
//  InteractionList.cuh
//  
//
//  Created by Pavel Zhuravlev on 5/8/14.
//
//
/*
 Interaction lists for FENE, native contacs, electrostatics, neighbor lists etc.
 N is number of particles that can possibly engage in this interaction;
 Nmax is the maximum number of counterparts of each particle;
 Each i-th of the N particles has count[i] counterparts in interaction: first, second ... count[i]
 map is array organized as follows: N first counterparts (with parameters) for each of N particles; then N second counterparts etc.
 map[j*N+i] is thus the record for j-th counterpart of the i-th particle.
 map is allocated for the size N*Nmax*sizeof(T)
 */

#ifndef ____InteractionList__
#define ____InteractionList__

#include "common.cuh"

template<typename T>
class InteractionList {
    
public:
    T* map_h;
    int* count_h;
    
    T* map_d;
    int* count_d;
    
    int N;
    int Nmax;
    
    void AllocateOnHost() {
        
        map_h=(T*)malloc(Nmax*N*sizeof(T));
        count_h=(int*)calloc(N, sizeof(int));
        
    }
    
    void FreeOnHost() {
        
        free(map_h);
        free(count_h);
        
    }
    
    void AllocateOnDevice(std::string s) {
        
        cudaMalloc((void**)&map_d, Nmax*N*sizeof(T));
        checkCUDAError(("Allocate "+s+" map").c_str());
        
        cudaMalloc((void**)&count_d, N*sizeof(int));
        checkCUDAError(("Allocate "+s+" count").c_str());
        
    }
    
    void FreeOnDevice(std::string s) {
        
        cudaFree(map_d);
        checkCUDAError(("Free "+s+" map").c_str());
        
        cudaFree(count_d);
        checkCUDAError(("Free "+s+" count").c_str());
        
    }
    
    void CopyToDevice(std::string s) {
        
        cudaMemcpy(map_d, map_h, Nmax*N*sizeof(T), cudaMemcpyHostToDevice);
        checkCUDAError(("Copy "+s+" map").c_str());
        
        cudaMemcpy(count_d, count_h, N*sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError(("Copy "+s+" count").c_str());
        
    }
    
    void CopyToHost(std::string s) {
        
        cudaMemcpy(map_h, map_d, Nmax*N*sizeof(T), cudaMemcpyDeviceToHost);
        checkCUDAError(("Copy "+s+" map").c_str());
        
        cudaMemcpy(count_h, count_d, N*sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError(("Copy "+s+" count").c_str());
        
    }
    
    void CheckNmaxHost (int i, std::string s) {
        if (count_h[i] > Nmax) {
            printf(("ERROR: Maximum number of "+s+" exceeded the limit of %d.\n").c_str(), Nmax);
            exit(0);
        }
    }
    
    void PrintReadingStatement(int N, std::string s) {
        printf(("Reading "+s+" list (%d)\n").c_str(),N);
    }
    
    void PrintReadError(int i, int N, std::string s) {
        printf(("Premature end of file at %d/%d "+s+" read\n").c_str(),i,N);
    }
    
    InteractionList<T>() {};

};

class InteractionListBond: public InteractionList<bond> {

public:
    
    InteractionListBond(FILE *ind, int N_in, int Nmax_in, int Nb, std::string msg, int ntraj) {
        
        N=N_in*ntraj; Nmax=Nmax_in;
        AllocateOnDevice(msg);
        AllocateOnHost();
        
        PrintReadingStatement(Nb,msg);
        for (int k=0; k<Nb; k++) {
            int i,j;
            bond bk;
            if (fscanf(ind,"%d %d %f %f %f", &i,&j,&(bk.l0),&(bk.k),&(bk.R0))==EOF)
                PrintReadError(k,Nb,msg);
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                
                bk.i2=j;
                map_h[N*count_h[i]+i]=bk;
                count_h[i]++;
                
                bk.i2=i;
                map_h[N*count_h[j]+j]=bk;
                count_h[j]++;
                
                CheckNmaxHost(i,msg);
                CheckNmaxHost(j,msg);
                
                i+=N_in;
                j+=N_in;
            }
        }
        
        CopyToDevice(msg);
        FreeOnHost();
    }
    
};

class InteractionListAngleVertex: public InteractionList<angle_vertex> {

public:
    
    InteractionListAngleVertex(int N_in, int Na, std::string msg, int ntraj) {
        
        N=N_in*ntraj;
        Nmax=3;
        AllocateOnDevice(msg);
        AllocateOnHost();
    }
    
    void Append(int i1, int iv, int i2, float k, float a0, std::string msg, int N_in, int ntraj) {

        angle_vertex a;
        a.i1 = i1;
        a.i2 = i2;
        a.k  = k;
        a.a0 = a0;

        for (int itraj=0; itraj<ntraj; itraj++) {
            map_h[N*count_h[iv]+iv]=a;
            count_h[iv]++;
                
            CheckNmaxHost(iv,msg);

            iv+=N_in;
        }
    }
};

class InteractionListAngleEnd: public InteractionList<angle_end> {

public:
    
    InteractionListAngleEnd(int N_in, int Na, std::string msg, int ntraj) {
        
        N=N_in*ntraj;
        Nmax=4;
        AllocateOnDevice(msg);
        AllocateOnHost();
    }
    
    void Append(int i1, int iv, int i2, float k, float a0, std::string msg, int N_in, int ntraj) {

        angle_end a;
        a.iv = iv;
        a.k  = k;
        a.a0 = a0;

        for (int itraj=0; itraj<ntraj; itraj++) {
            a.i2 = i2;
            map_h[N*count_h[i1]+i1]=a;
            count_h[i1]++;

            a.i2 = i1;
            map_h[N*count_h[i2]+i2]=a;
            count_h[i2]++;
                
            CheckNmaxHost(i1,msg);
            CheckNmaxHost(i2,msg);

            i1+=N_in;
            i2+=N_in;
        }
    }
};

class InteractionListNC: public InteractionList<nc> {
    
public:
    
    InteractionListNC(FILE *ind,int N_in, int Nmax_in, int Nnc, std::string msg, int ntraj) {
        
        N=N_in*ntraj;
        Nmax=Nmax_in;
        
        AllocateOnDevice(msg);
        AllocateOnHost();
        
        PrintReadingStatement(Nnc,msg);
        for (int k=0; k<Nnc; k++) {
            int i,j;
            float r0,eps;
            if (fscanf(ind,"%d %d %f %f", &i,&j,&r0,&eps)==EOF)
                PrintReadError(k,Nnc,msg);
            
            nc nck;
            nck.r02=r0*r0;
            nck.factor=12.0*eps/r0/r0;
            nck.epsilon=eps;
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                nck.i2=j;
                map_h[N*count_h[i]+i]=nck;
                count_h[i]++;
                
                nck.i2=i;
                map_h[N*count_h[j]+j]=nck;
                count_h[j]++;
                
                CheckNmaxHost(i,msg);
                CheckNmaxHost(j,msg);
                
                i+=N_in;
                j+=N_in;
            }
        }
        
        CopyToDevice(msg);
        FreeOnHost();

    }
};

class InteractionListSB: public InteractionList<bond> {
    
public:
    
    InteractionListSB(FILE *ind,int N_in, int Nmax_in, int Nsb, std::string msg, int ntraj) {
        
        N=N_in*ntraj; Nmax=Nmax_in;
        AllocateOnDevice(msg);
        AllocateOnHost();
        
        PrintReadingStatement(Nsb,msg);
        for (int k=0; k<Nsb; k++) {
            int i,j;
            float qiqj;
            bond sbk;
            sbk.k=999999999999.9; // not used (will be modified later)
            if (fscanf(ind,"%d %d %f", &i,&j,&qiqj)==EOF)
                PrintReadError(k,Nsb,msg);
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                
                sbk.l0=els_h.prefactor*qiqj;
                sbk.i2=j;
                
                map_h[N*count_h[i]+i]=sbk;
                count_h[i]++;
                
                sbk.i2=i;
                map_h[N*count_h[j]+j]=sbk;
                count_h[j]++;
                
                CheckNmaxHost(i,msg);
                CheckNmaxHost(j,msg);
                i+=N_in;
                j+=N_in;
            }
        }
        
        CopyToDevice(msg);
        FreeOnHost();
    }
};

#endif /* defined(____InteractionList__) */
