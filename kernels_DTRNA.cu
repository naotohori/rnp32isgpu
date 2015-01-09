__device__ float4 operator-(const float4 & a, const float4 & b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ float dot_product(const float4 & a, const float4 & b) {
    return (a.x*b.x + a.y*b.y + a.z*b.z);
    // CAUTION: This function does not use neither a.w nor b.w.
}


__global__ void HarmonicBondForce(float4* r, float4* forces, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nb=list.count_d[i];                 //Number of bonds of the i-th bead

    for (int ib=0; ib<Nb; ib++) {           //Loop over bonds of the i-th bead
        bond b=list.map_d[ib*list.N+i];     //Look up bond in the map
        float4 l=tex1Dfetch(r_t, b.i2);     //(reading from texture cache is faster than directly from r[])
        l = l - ri;
        float dist = sqrt( dot_product(l,l) );
        l.w = dist - b.l0;
        l.w = 2.0 * b.k * l.w / dist;
        f.x+=l.w*l.x;
        f.y+=l.w*l.y;
        f.z+=l.w*l.z;
    }
    forces[i]=f;
}

__global__ void HarmonicBondEnergy(float4* r, InteractionList<bond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    int Nb=list.count_d[i]; //Number of bonds of the i-th bead

    for (int ib=0; ib<Nb; ib++) {           //Loop over bonds of the i-th bead
        bond b=list.map_d[ib*list.N+i];     //Look up bond in the map
        float4 l=tex1Dfetch(r_t, b.i2);     //(reading from texture cache is faster than directly from r[])
        l = l - ri;
        l.w=sqrt( dot_product(l,l) );
        l.w-=b.l0;
        l.w=b.k*l.w*l.w;
        energy+=l.w;
    }
    r[i].w=energy;
}


__global__ void AngleVertexForce(float4* r, float4* forces, InteractionList<angle_vertex> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Na=list.count_d[i];                 //Number of bonds of the i-th bead

    for (int iang=0; iang<Na; iang++) {     //Loop over bonds of the i-th bead
        angle_vertex a = list.map_d[iang*list.N+i]; //Look up bond in the map
        float4 l1=tex1Dfetch(r_t, a.i1);
        float4 l2=tex1Dfetch(r_t, a.i2);
        l1 = l1 - ri;
        l2 = l2 - ri;
        float l1l1 = dot_product(l1,l1);
        float l2l2 = dot_product(l2,l2);
        float l1l2 = dot_product(l1,l2);
        float ang = acos( l1l2 / sqrt( l1l1 * l2l2 ) );
        float fact = -2.0 * a.k * (ang - a.a0) / sqrt(l1l1 * l2l2 - l1l2 * l1l2);
        l1l1 = 1.0 / l1l1;
        l2l2 = 1.0 / l2l2;
        f.x += fact * (l1.x + l2.x - l1l2 * (l1.x * l1l1 + l2.x * l2l2));
        f.y += fact * (l1.y + l2.y - l1l2 * (l1.y * l1l1 + l2.y * l2l2));
        f.z += fact * (l1.z + l2.z - l1l2 * (l1.z * l1l1 + l2.z * l2l2));
        
    }
    forces[i]=f;
}


__global__ void AngleEndForce(float4* r, float4* forces, InteractionList<angle_end> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Na=list.count_d[i]; //Number of angles of the i-th angle
    for (int iang=0; iang<Na; iang++) {           //Loop over bonds of the i-th bead
        angle_end a = list.map_d[iang*list.N+i]; //Look up bond in the map
        float4 lv=tex1Dfetch(r_t, a.iv);
        float4 l2=tex1Dfetch(r_t, a.i2);
        float4 l1 = ri - lv;
        l2 = l2 - lv;
        float l1l2 = dot_product(l1,l2);
        float l1l1 = dot_product(l1,l1);
        float l2l2 = dot_product(l2,l2);
        float ang = acos( l1l2 / sqrt( l1l1 * l2l2 ) );
        float fact = 2.0 * a.k * (ang - a.a0) / sqrt( l1l1 * l2l2 - l1l2 * l1l2);

        l1l2 = l1l2 / l1l1;
        f.x += fact * (l2.x - l1l2 * l1.x);
        f.y += fact * (l2.y - l1l2 * l1.y);
        f.z += fact * (l2.z - l1l2 * l1.z);
    }
    forces[i]=f;
}
        

__global__ void AngleEnergy(float4* r, InteractionList<angle_vertex> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    int Na=list.count_d[i]; //Number of angles of the i-th angle

    for (int iang=0; iang<Na; iang++) {           //Loop over bonds of the i-th bead
        angle_vertex a = list.map_d[iang*list.N+i]; //Look up bond in the map
        float4 l1=tex1Dfetch(r_t, a.i1);
        float4 l2=tex1Dfetch(r_t, a.i2);
        l1 = l1 - ri;
        l2 = l2 - ri;
        float ang = acos( dot_product(l1,l2) / sqrt( dot_product(l1,l1) * dot_product(l2,l2) ) );
        ang -= a.a0;
        energy += a.k * ang * ang;
    }
    r[i].w=energy;
}
