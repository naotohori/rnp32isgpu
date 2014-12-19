__global__ void WCAForce(float4 *r, float4 *forces, InteractionList<int> list, float *sig) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 f=forces[i];
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    //float sigi=sig[i];
    float sigi=tex1Dfetch(sig_t,i);                                             //Sigma of the i-th bead
    int Nneib=list.count_d[i];                                                  //Number of neighbors of the i-th bead
    for (int ineib=0;ineib<Nneib;ineib++) {                                     //Loop over neighbors of the i-th bead
        int j=list.map_d[ineib*list.N+i];                                       //Look up neibor in the neibor list
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        //float4 r2=tex1Dfetch(r_t,tex1Dfetch(neibmap_t,ineib*list.N+i);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        //float sigma2=(sigi+sig[j])/2.;
        float sigma2=(sigi+tex1Dfetch(sig_t,j))/2.;     // sigma of the other bead, and mixed into sigma_ij
        sigma2*=sigma2;
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);
        float r6inv=r2.w*r2.w*r2.w;
        r2.w=12.*ss_c.eps/sigma2*r2.w*r6inv*(1-r6inv);
        f.x+=r2.w*r2.x;
        f.y+=r2.w*r2.y;
        f.z+=r2.w*r2.z;
        
    }
    forces[i]=f;
}

__global__ void WCAEnergy(float4 *r, InteractionList<int> list, float *sig) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    //float sigi=sig[i];
    float sigi=tex1Dfetch(sig_t,i);                                             //Sigma of the i-th bead
    int Nneib=list.count_d[i];                                                  //Number of neighbors of the i-th bead
    for (int ineib=0;ineib<Nneib;ineib++) {                                     //Loop over neighbors of the i-th bead
        int j=list.map_d[ineib*list.N+i];                                       //Look up neibor in the neibor list
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        //float4 r2=tex1Dfetch(r_t,tex1Dfetch(neibmap_t,ineib*list.N+i);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        //float sigma2=(sigi+sig[j])/2.;
        float sigma2=(sigi+tex1Dfetch(sig_t,j))/2.;
        sigma2*=sigma2;
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z); // squared
        float r6inv=r2.w*r2.w*r2.w;
        energy+=ss_c.eps*(r6inv*(r6inv-2.0f)+1.0f);
    }
    r[i].w=energy;
}

__global__ void NativeSubtractWCAForce(float4* r, float4* forces, InteractionList<nc> list, float *sig) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nnc=list.count_d[i];
    //float sigi=sig[i];
    float sigi=tex1Dfetch(sig_t,i);
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        int j=ncij.i2;
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        //float sigma2=(sigi+sig[j])/2.;
        float sigma2=(sigi+tex1Dfetch(sig_t,j))/2.;
        sigma2*=sigma2;
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z); // squared
        float r6inv=r2.w*r2.w*r2.w;
        r2.w=12.*ss_c.eps/sigma2*r2.w*r6inv*(1-r6inv);
        f.x-=r2.w*r2.x;
        f.y-=r2.w*r2.y;
        f.z-=r2.w*r2.z;
    }
    forces[i]=f;
}

__global__ void NativeSubtractWCAForce(float4* r, float4* forces, InteractionList<nc> list, float *sig, float Delta) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nnc=list.count_d[i];
    //float sigi=sig[i];
    float sigi=tex1Dfetch(sig_t,i);
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        int j=ncij.i2;
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        //float sigma2=(sigi+sig[j])/2.;
        float sigma2=(sigi+tex1Dfetch(sig_t,j))/2.;
        sigma2*=sigma2;
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z); // squared
        float r6inv=r2.w*r2.w*r2.w;
        r2.w=Delta*12.*ss_c.eps/sigma2*r2.w*r6inv*(1-r6inv);
        f.x-=r2.w*r2.x;
        f.y-=r2.w*r2.y;
        f.z-=r2.w*r2.z;
    }
    forces[i]=f;
}

__global__ void NativeSubtractWCAEnergy(float4 *r, InteractionList<nc> list, float *sig) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    int Nnc=list.count_d[i];
    //float sigi=sig[i];
    float sigi=tex1Dfetch(sig_t,i);
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        int j=ncij.i2;
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        //float sigma2=(sigi+sig[j])/2.;
        float sigma2=(sigi+tex1Dfetch(sig_t,j))/2.;
        sigma2*=sigma2;
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);      // squared
        float r6inv=r2.w*r2.w*r2.w;
        energy-=ss_c.eps*(r6inv*(r6inv-2.0f)+1.0f);
    }
    r[i].w=energy;
}


__global__ void WCANeighborList(float4* r, InteractionList<int> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    int neighbors=0;
    for (int j=0;j<list.N;j++) {
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        if (
            (r2.w<ss_c.Rcut2) and
            (
             (abs(j-i)>1) or
             ((abs(j-i)>0) and ((i>=list.N/2) or (j>=list.N/2)))  //bb with ss or ss with ss on neighboring residues (this actually excludes terminal beads of different chains, that are not bound)
             ) and
            ((j+list.N/2)!=i) and                                 //exclude covalently bonded bb and ss beads
            ((i+list.N/2)!=j)
            ) {
            
            list.map_d[neighbors*list.N+i]=j;
            neighbors++;
        }
    }
    list.count_d[i]=neighbors;
    
}

__global__ void WCANeighborListMultTraj(float4* r, InteractionList<int> list, int Ntraj) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    int neighbors=0;
    for (int j=0;j<list.N;j++) {
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        if (
            (r2.w<ss_c.Rcut2) and
            (
             (abs(j-i)>1) or
             ((abs(j-i)>0) and ((i>=list.N/2) or (j>=list.N/2)))  //bb with ss or ss with ss on neighboring residues (this actually excludes terminal beads of different chains, that are not bound)
             ) and
            ((j+list.N/2)!=i) and                                 //exclude covalently bonded bb and ss beads
            ((i+list.N/2)!=j) and
            ((i/Ntraj)==(j/Ntraj))								 //make sure beads belong to the same trajectory/replica
            ) {
            
            list.map_d[neighbors*list.N+i]=j;
            neighbors++;
        }
    }
    list.count_d[i]=neighbors;
    
}

__global__ void WCANeighborList(float4* r, InteractionList<int> intlist, InteractionList<int> neiblist) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=intlist.N) return;
        
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    int Npartners=intlist.count_d[i];
    int neighbors=0;
    for (int ip=0;ip<Npartners;ip++) {
        int j=intlist.map_d[ip*intlist.N+i];
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        if (r2.w<ss_c.Rcut2) {
            neiblist.map_d[neighbors*neiblist.N+i]=j;
            neighbors++;
        }
    }
    neiblist.count_d[i]=neighbors;
}
