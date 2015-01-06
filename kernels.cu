__global__ void reduce(float* a, int N) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id<N) {
        a[2*id]+=a[2*id+1];
        __syncthreads();
    }
}

__global__ void force_flush (float4 *f, int N) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id>=N) return;
    f[id].x=0.;
    f[id].y=0.;
    f[id].z=0.;
}

__global__ void rand_init (int seed, curandStatePhilox4_32_10_t* states) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &states[id]);
}

__global__ void integrate(float4 *r, float4 *forces, int N, curandStatePhilox4_32_10_t* states) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id>=N) return;
    
    float4 f=forces[id];
    float4 wn = curand_normal4(&states[id]); //Gaussian white noise ~N(0,1)
    //float4 ri=r[id];
    float4 ri=tex1Dfetch(r_t, id);
    float3 dr;
    dr.x=bd_c.hoz*f.x+bd_c.Gamma*wn.x;
    dr.y=bd_c.hoz*f.y+bd_c.Gamma*wn.y;
    dr.z=bd_c.hoz*f.z+bd_c.Gamma*wn.z;
    ri.x+=dr.x;
    ri.y+=dr.y;
    ri.z+=dr.z;
    ri.w=dr.x*dr.x+dr.y*dr.y+dr.z*dr.z;     //Save velocity squared for calculation of diffusion constant / temperature / kinetic energy
    r[id]=ri;
}

__global__ void minimize(float4 *r, float4 *forces, int N, float alpha) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id>=N) return;
    
    float4 f=forces[id];
    //float4 ri=r[id];
    float4 ri=tex1Dfetch(r_t, id);
    ri.x+=alpha*f.x;
    ri.y+=alpha*f.y;
    ri.z+=alpha*f.z;
    r[id]=ri;
}

__global__ void FENEForce(float4* r, float4* forces, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nb=list.count_d[i];                 //Number of bonds of the i-th bead
    for (int ib=0; ib<Nb; ib++) {           //Loop over bonds of the i-th bead
        bond b=list.map_d[ib*list.N+i];     //Look up bond in the map
        //float4 l=r[b.i2];                 //Number of bead on the other end of the bond (i2) and its coordinates (l)
        float4 l=tex1Dfetch(r_t, b.i2);     //(reading from texture cache is faster than directly from r[])
        l.x-=ri.x;                          //Atom-to-bead vector
        l.y-=ri.y;
        l.z-=ri.z;
        l.w=sqrtf(l.x*l.x+l.y*l.y+l.z*l.z);
        l.w-=b.l0;
        float denom=(1.-l.w*l.w/fene_c.R02);
        l.w=fene_c.k*l.w/denom/(l.w+b.l0);
        f.x+=l.w*l.x;
        f.y+=l.w*l.y;
        f.z+=l.w*l.z;
    }
    forces[i]=f;
}

__global__ void FENEEnergy(float4* r, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    int Nb=list.count_d[i]; //Number of bonds of the i-th bead
    //printf("Atom %d (%f %f %f):\n",i,ri.x,ri.y,ri.z);
    for (int ib=0; ib<Nb; ib++) {           //Loop over bonds of the i-th bead
        bond b=list.map_d[ib*list.N+i];     //Look up bond in the map
        //float4 l=r[b.i2];                 //Number of bead on the other end of the bond (i2) and its coordinates (l)
        float4 l=tex1Dfetch(r_t, b.i2);     //(reading from texture cache is faster than directly from r[])
        l.x-=ri.x;                          //Atom-to-bead vector
        l.y-=ri.y;
        l.z-=ri.z;
        l.w=sqrtf(l.x*l.x+l.y*l.y+l.z*l.z);
        //printf("%d %d %f %f %f\n",i,b.i2,energy,b.l0,l.w);
        l.w-=b.l0;
        l.w=.5*fene_c.k*fene_c.R02*logf(1.-l.w*l.w/fene_c.R02);
        //printf("%f\n",l.w);
        energy+=l.w;
    }
    r[i].w=energy;
}


__global__ void SoftSphereForce(float4 *r, float4 *forces, InteractionList<int> list, float *sig) {
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
        r2.w=sigma2/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);    // squared
        if (r2.w>ss_c.CutOffFactor2inv) {               // Potential is cut off at rcut=CutOffFactor*sigma => sigma^2/r^2 should be > 1/CutOffFactor2
            r2.w*=r2.w;                                 // to the 4th
            r2.w*=r2.w;                                 // to the 8th
            r2.w=ss_c.Minus6eps/sigma2*(r2.w+ss_c.CutOffFactor8inv);
            f.x+=r2.x*r2.w;
            f.y+=r2.y*r2.w;
            f.z+=r2.z*r2.w;
        }
    }
    forces[i]=f;
}

__global__ void SoftSphereEnergy(float4 *r, InteractionList<int> list, float *sig) {
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
        if (r2.w>ss_c.CutOffFactor2inv)              // Potential is cut off at rcut=CutOffFactor*sigma => sigma^2/r^2 should be > 1/CutOffFactor2
            //energy+=ss_c.eps*r2.w*r2.w*r2.w;         // to the 6th
            energy+=ss_c.eps*(r2.w*r2.w*r2.w-ss_c.CutOffFactor6inv);
    }
    r[i].w=energy;
}

__global__ void NativeSubtractSoftSphereForce(float4* r, float4* forces, InteractionList<nc> list, float *sig) {
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
        if (r2.w>ss_c.CutOffFactor2inv) {
            r2.w*=r2.w;                                  // to the 4th
            r2.w*=r2.w;                                 // to the 8th
            r2.w=ss_c.Minus6eps/sigma2*(r2.w+ss_c.CutOffFactor8inv);
            f.x-=r2.x*r2.w;
            f.y-=r2.y*r2.w;
            f.z-=r2.z*r2.w;
        }
    }
    forces[i]=f;
}

__global__ void NativeSubtractSoftSphereForce(float4* r, float4* forces, InteractionList<nc> list, float *sig, float Delta) {
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
        if (r2.w>ss_c.CutOffFactor2inv) {
            r2.w*=r2.w;                                  // to the 4th
            r2.w*=r2.w;                                 // to the 8th
            r2.w=Delta*ss_c.Minus6eps/sigma2*(r2.w+ss_c.CutOffFactor8inv);
            f.x-=r2.x*r2.w;
            f.y-=r2.y*r2.w;
            f.z-=r2.z*r2.w;
        }
    }
    forces[i]=f;
}

__global__ void NativeSubtractSoftSphereEnergy(float4 *r, InteractionList<nc> list, float *sig) {
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
        if (r2.w>ss_c.CutOffFactor2inv)
            energy-=ss_c.eps*(r2.w*r2.w*r2.w-ss_c.CutOffFactor6inv);              // to the 6th
    }
    r[i].w=energy;
}

__global__ void NativeForce(float4* r, float4* forces, InteractionList<nc> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    float4 f=forces[i];
    int Nnc=list.count_d[i];
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        //float4 r2=r[ncij.i2];
        float4 r2=tex1Dfetch(r_t,ncij.i2);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=ncij.r02/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);
        float r6inv=r2.w*r2.w*r2.w;
        r2.w=ncij.factor*r2.w*r6inv*(1-r6inv);
        f.x+=r2.w*r2.x;
        f.y+=r2.w*r2.y;
        f.z+=r2.w*r2.z;
    }
    forces[i]=f;
}

__global__ void NativeForce(float4* r, float4* forces, InteractionList<nc> list, float Delta) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    float4 f=forces[i];
    int Nnc=list.count_d[i];
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        //float4 r2=r[ncij.i2];
        float4 r2=tex1Dfetch(r_t,ncij.i2);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=ncij.r02/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);
        float r6inv=r2.w*r2.w*r2.w;
        r2.w=Delta*ncij.factor*r2.w*r6inv*(1-r6inv);
        f.x+=r2.w*r2.x;
        f.y+=r2.w*r2.y;
        f.z+=r2.w*r2.z;
    }
    forces[i]=f;
}

__global__ void NativeEnergy(float4* r, InteractionList<nc> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    float energy=0.0f;
    int Nnc=list.count_d[i];
    for (int inc=0; inc<Nnc; inc++) {
        nc ncij=list.map_d[inc*list.N+i];
        //float4 r2=r[ncij.i2];
        float4 r2=tex1Dfetch(r_t,ncij.i2);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=ncij.r02/(r2.x*r2.x+r2.y*r2.y+r2.z*r2.z);
        float r6inv=r2.w*r2.w*r2.w;
        energy+=ncij.epsilon*r6inv*(r6inv-2.0f);
    }
    r[i].w=energy;
}

__global__ void DebyeHuckelForce(float4* r, float4* forces, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    float4 f=forces[i];
    int Nsb=list.count_d[i];
    for (int isb=0; isb<Nsb; isb++) {
        bond sbij=list.map_d[isb*list.N+i];
        //float4 r2=r[ncij.i2];
        float4 r2=tex1Dfetch(r_t,sbij.i2);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        float dist2=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        float dist=sqrtf(dist2);
        //if (dist<1.5*els_c.kappainv) {
            r2.w=expf(-dist/els_c.kappainv)*sbij.l0/dist2;
            f.x+=r2.w*r2.x;
            f.y+=r2.w*r2.y;
            f.z+=r2.w*r2.z;
        //}
    }
    forces[i]=f;
}

__global__ void DebyeHuckelEnergy(float4* r, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    float energy=0.f;
    int Nsb=list.count_d[i];
    for (int isb=0; isb<Nsb; isb++) {
        bond sbij=list.map_d[isb*list.N+i];
        //float4 r2=r[ncij.i2];
        float4 r2=tex1Dfetch(r_t,sbij.i2);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        float dist2=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        float dist=sqrtf(dist2);
        //if (dist<1.5*els_c.kappainv)
            energy+=expf(-dist/els_c.kappainv)*sbij.l0/dist;
    }
    r[i].w=energy;
}


__global__ void SoftSphereNeighborList(float4* r, InteractionList<int> list, int Ntraj) {
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
        if (r2.w>ss_c.Rcut2)
            continue;
        if (i==j)
            continue;
        bool atchainstart=false;
        for (int chain=1; chain<chainstarts_c[0]; chain++) {
            atchainstart+=(max(i,j)==chainstarts_c[chain]);
        }
        if (
            (
             (abs(j-i)>1) //include all-nonneighboring beads
            or
             ((max(i,j) % Ntraj)>=Ntraj/2) // if one is sidechain, it interacts with any other side chain or backbone
            or atchainstart //if one bead is the start of a chain, it interacts with any other bead
            )
            and
             (abs(j-i)!=Ntraj/2) //exclude bb and sc of the same residue
            and
             ((i/Ntraj)==(j/Ntraj)) //exclude beads belonging to different trajectories/replicas
            ) {
            list.map_d[neighbors*list.N+i]=j;
            neighbors++;
        }
    }
    list.count_d[i]=neighbors;
    
}

__global__ void SoftSphereNeighborList(float4* r, InteractionList<int> list, InteractionList<bond> blist, int Ntraj) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    //float4 ri=r[i];
    float4 ri=tex1Dfetch(r_t,i);
    int neighbors=0;
    int Nb=blist.count_d[i];
    for (int j=0;j<list.N;j++) {
        //float4 r2=r[j];
        float4 r2=tex1Dfetch(r_t,j);
        r2.x-=ri.x;
        r2.y-=ri.y;
        r2.z-=ri.z;
        r2.w=r2.x*r2.x+r2.y*r2.y+r2.z*r2.z;
        
        //Check that i and j are not bonded by looping over bonds of i
        bool nonbonded=true;
        for (int ib=0; ib<Nb; ib++) {
            bond b=blist.map_d[ib*blist.N+i];
            if (b.i2==j) nonbonded=false;
        }
        
        if ((r2.w<ss_c.Rcut2) and (i!=j) and (nonbonded) and (i/Ntraj)==(j/Ntraj)) {
            list.map_d[neighbors*list.N+i]=j;
            neighbors++;
        }
    }
    list.count_d[i]=neighbors;
    
}

__global__ void SoftSphereNeighborList(float4* r, InteractionList<int> intlist, InteractionList<int> neiblist) {
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
