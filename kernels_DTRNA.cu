#include <math_constants.h>

__device__ float4 operator-(const float4 & a, const float4 & b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ float dot_product(const float4 & a, const float4 & b) {
    return (a.x*b.x + a.y*b.y + a.z*b.z);
    // CAUTION: This function does not use a.w or b.w.
}

__device__ float4 cross_product(const float4 & a, const float4 & b) {
    return make_float4(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x,
                       0.f);
    // CAUTION: This function does not use a.w or b.w, and return 0.0 for w.
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
    r[i].w=0.5*energy; // To eliminate duplication
}


__global__ void AngleVertexForce(float4* r, float4* forces, InteractionList<angle> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Na=list.count_d[i];                 //Number of bonds of the i-th bead

    for (int iang=0; iang<Na; iang++) {     //Loop over bonds of the i-th bead
        angle a = list.map_d[iang*list.N+i]; //Look up bond in the map
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


__global__ void AngleEndForce(float4* r, float4* forces, InteractionList<angle> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Na=list.count_d[i]; //Number of angles of the i-th angle
    for (int iang=0; iang<Na; iang++) {           //Loop over bonds of the i-th bead
        angle a = list.map_d[iang*list.N+i]; //Look up bond in the map
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
        

__global__ void AngleEnergy(float4* r, InteractionList<angle> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;
    
    float4 ri=tex1Dfetch(r_t, i);
    float energy=0.0f;
    int Na=list.count_d[i]; //Number of angles of the i-th angle

    for (int iang=0; iang<Na; iang++) {           //Loop over bonds of the i-th bead
        angle a = list.map_d[iang*list.N+i]; //Look up bond in the map
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


// This function should be called for a list of P2 particles
__global__ void StackEnergy(float4* r, InteractionList<stack> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float denom = 1.0f;
    stack st = list.map_d[i];
    float4 rP1 = tex1Dfetch(r_t, st.iP1);
    float4 rS1 = tex1Dfetch(r_t, st.iS1);
    float4 rB1 = tex1Dfetch(r_t, st.iB1);
    float4 rP2 = tex1Dfetch(r_t, st.iP2); // <=== i
    float4 rS2 = tex1Dfetch(r_t, st.iS2);
    float4 rB2 = tex1Dfetch(r_t, st.iB2);
    float4 rP3 = tex1Dfetch(r_t, st.iP3);

    float4 B2B1 = rB2 - rB1;  // 21
    float4 P1S1 = rP1 - rS1;  // 34
    float4 P2S1 = rP2 - rS1;  // 54 
    float4 P2S2 = rP2 - rS2;  // 56 
    float4 P3S2 = rP3 - rS2;  // 76

    // length between B1 and B2
    float delta = sqrt(dot_product(B2B1,B2B1)) - st.l0;
    denom += st.kl * delta * delta;

    // dihedral of P1-S1-P2-S2
    float4 m = cross_product(P1S1, P2S1); // normal vectors
    float4 n = cross_product(P2S1, P2S2);
    delta = atan2( dot_product(P1S1,n) * sqrt(dot_product(P2S1,P2S1)), dot_product(m,n)) - st.phi10;
    if (delta > CUDART_PI_F) { 
       delta = delta - 2.*CUDART_PI_F;
    } else if (delta < -CUDART_PI_F) {
       delta = delta + 2.*CUDART_PI_F;
    }
    denom += st.kphi1 * delta * delta;

    // dihedral of P3-S2-P2-S1
    m = cross_product(P3S2, P2S2);
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;
    delta = atan2( dot_product(P3S2,n) * sqrt(dot_product(P2S2,P2S2)), dot_product(m,n)) - st.phi20;
    if (delta > CUDART_PI_F) { 
       delta = delta - 2.*CUDART_PI_F;
    } else if (delta < -CUDART_PI_F) {
       delta = delta + 2.*CUDART_PI_F;
    }
    denom += st.kphi2 * delta * delta;

    r[i].w = st.U0 / denom;
}


__global__ void StackP13Force(float4* r, float4* forces, InteractionList<stack> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nst=list.count_d[i];
    for (int ist=0; ist<Nst; ist++) {

        float denom = 1.0f;
        stack st = list.map_d[ist*list.N+i];

        float4 rP1 = tex1Dfetch(r_t, st.iP1);
        float4 rS1 = tex1Dfetch(r_t, st.iS1);
        float4 rB1 = tex1Dfetch(r_t, st.iB1);
        float4 rP2 = tex1Dfetch(r_t, st.iP2);
        float4 rS2 = tex1Dfetch(r_t, st.iS2);
        float4 rB2 = tex1Dfetch(r_t, st.iB2);
        float4 rP3 = tex1Dfetch(r_t, st.iP3);
    
        float4 B2B1 = rB2 - rB1;  // 21
        float4 P1S1 = rP1 - rS1;  // 34
        float4 P2S1 = rP2 - rS1;  // 54 
        float4 P2S2 = rP2 - rS2;  // 56 
        float4 P3S2 = rP3 - rS2;  // 76

        // length between B1 and B2
        float delta = sqrt(dot_product(B2B1,B2B1)) - st.l0;
        denom += st.kl * delta * delta;

        // dihedral of P1-S1-P2-S2
        float4 m = cross_product(P1S1, P2S1); // normal vectors
        float4 n = cross_product(P2S1, P2S2);
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi10;
        if (delta > CUDART_PI_F) { 
            delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
            delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi1 * delta * delta;

        float fact = 2.0f * st.kphi1 * delta * absP2S1 / dot_product(m,m);
        float4 ft;
        ft.x = fact * m.x;
        ft.y = fact * m.y;
        ft.z = fact * m.z;

        // dihedral of P3-S2-P2-S1
        m = cross_product(P3S2, P2S2);
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        delta = atan2( dot_product(P3S2,n) * sqrt(dot_product(P2S2,P2S2)), dot_product(m,n)) - st.phi20;
        if (delta > CUDART_PI_F) { 
           delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
           delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi2 * delta * delta;

        fact = st.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}

__global__ void StackP2Force(float4* r, float4* forces, InteractionList<stack> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nst=list.count_d[i];
    for (int ist=0; ist<Nst; ist++) {

        float denom = 1.0f;
        stack st = list.map_d[ist*list.N+i];

        float4 rP1 = tex1Dfetch(r_t, st.iP1);
        float4 rS1 = tex1Dfetch(r_t, st.iS1);
        float4 rB1 = tex1Dfetch(r_t, st.iB1);
        float4 rP2 = tex1Dfetch(r_t, st.iP2);
        float4 rS2 = tex1Dfetch(r_t, st.iS2);
        float4 rB2 = tex1Dfetch(r_t, st.iB2);
        float4 rP3 = tex1Dfetch(r_t, st.iP3);
    
        float4 B2B1 = rB2 - rB1;  // 21
        float4 P1S1 = rP1 - rS1;  // 34
        float4 P2S1 = rP2 - rS1;  // 54 
        float4 P2S2 = rP2 - rS2;  // 56 
        float4 P3S2 = rP3 - rS2;  // 76

        // length between B1 and B2
        float delta = sqrt(dot_product(B2B1,B2B1)) - st.l0;
        denom += st.kl * delta * delta;

        // dihedral of P1-S1-P2-S2
        float4 m = cross_product(P1S1, P2S1); // normal vectors
        float4 n = cross_product(P2S1, P2S2);
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi10;
        if (delta > CUDART_PI_F) { 
            delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
            delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi1 * delta * delta;

        float fact = 2.0f * st.kphi1 * delta * absP2S1;
        float fact_i = + fact / dot_product(m,m);
        float fact_j = - fact / dot_product(n,n);
        float4 fi;
        float4 fj;
        fi.x = fact_i * m.x;
        fi.y = fact_i * m.y;
        fi.z = fact_i * m.z;
        fj.x = fact_j * n.x;
        fj.y = fact_j * n.y;
        fj.z = fact_j * n.z;

        fact_i =       - dot_product(P1S1,P2S1) / dot_product(P2S1,P2S1);
        fact_j = -1.0f + dot_product(P2S2,P2S1) / dot_product(P2S1,P2S1);
        float4 ft;
        ft.x = fact_i * fi.x + fact_j * fj.x;
        ft.y = fact_i * fi.y + fact_j * fj.y;
        ft.z = fact_i * fi.z + fact_j * fj.z;

        // dihedral of P3-S2-P2-S1
        m = cross_product(P3S2, P2S2); // normal vectors
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi20;
        if (delta > CUDART_PI_F) { 
           delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
           delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi2 * delta * delta;

        fact = 2.0f * st.kphi2 * delta * absP2S2;
        fact_i = + fact / dot_product(m,m);
        fact_j = - fact / dot_product(n,n);
        fi.x = fact_i * m.x;
        fi.y = fact_i * m.y;
        fi.z = fact_i * m.z;
        fj.x = fact_j * n.x;
        fj.y = fact_j * n.y;
        fj.z = fact_j * n.z;

        fact_i =       - dot_product(P3S2,P2S2) / dot_product(P2S2,P2S2);
        fact_j = -1.0f + dot_product(P2S1,P2S2) / dot_product(P2S2,P2S2);
        ft.x += fact_i * fi.x + fact_j * fj.x;
        ft.y += fact_i * fi.y + fact_j * fj.y;
        ft.z += fact_i * fi.z + fact_j * fj.z;

        fact = st.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}

__global__ void StackSForce(float4* r, float4* forces, InteractionList<stack> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nst=list.count_d[i];
    for (int ist=0; ist<Nst; ist++) {

        float denom = 1.0f;
        stack st = list.map_d[ist*list.N+i];

        float4 rP1 = tex1Dfetch(r_t, st.iP1);
        float4 rS1 = tex1Dfetch(r_t, st.iS1);
        float4 rB1 = tex1Dfetch(r_t, st.iB1);
        float4 rP2 = tex1Dfetch(r_t, st.iP2);
        float4 rS2 = tex1Dfetch(r_t, st.iS2);
        float4 rB2 = tex1Dfetch(r_t, st.iB2);
        float4 rP3 = tex1Dfetch(r_t, st.iP3);
    
        float4 B2B1 = rB2 - rB1;  // 21
        float4 P1S1 = rP1 - rS1;  // 34
        float4 P2S1 = rP2 - rS1;  // 54 
        float4 P2S2 = rP2 - rS2;  // 56 
        float4 P3S2 = rP3 - rS2;  // 76

        // length between B1 and B2
        float delta = sqrt(dot_product(B2B1,B2B1)) - st.l0;
        denom += st.kl * delta * delta;

        // dihedral of P1-S1-P2-S2
        float4 m = cross_product(P1S1, P2S1); // normal vectors
        float4 n = cross_product(P2S1, P2S2);
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi10;
        if (delta > CUDART_PI_F) { 
            delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
            delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi1 * delta * delta;

        float fact = 2.0f * st.kphi1 * delta * absP2S1;
        float fact_i = + fact / dot_product(m,m);
        float fact_j = - fact / dot_product(n,n);
        float4 fi;
        float4 fj;
        fi.x = fact_i * m.x;
        fi.y = fact_i * m.y;
        fi.z = fact_i * m.z;
        fj.x = fact_j * n.x;
        fj.y = fact_j * n.y;
        fj.z = fact_j * n.z;

        fact_i = -1.0f + dot_product(P1S1,P2S1) / dot_product(P2S1,P2S1);
        fact_j =       - dot_product(P2S2,P2S1) / dot_product(P2S1,P2S1);
        float4 ft;
        ft.x = fact_i * fi.x + fact_j * fj.x;
        ft.y = fact_i * fi.y + fact_j * fj.y;
        ft.z = fact_i * fi.z + fact_j * fj.z;

        // dihedral of P3-S2-P2-S1
        m = cross_product(P3S2, P2S2); // normal vectors
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi20;
        if (delta > CUDART_PI_F) { 
           delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
           delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi2 * delta * delta;

        fact = -2.0f * st.kphi2 * delta * absP2S2 / dot_product(n,n);

        ft.x +=  fact * n.x;
        ft.y +=  fact * n.y;
        ft.z +=  fact * n.z;

        fact = st.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}

__global__ void StackBForce(float4* r, float4* forces, InteractionList<stack> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 ri=tex1Dfetch(r_t, i);
    float4 f=forces[i];
    int Nst=list.count_d[i];
    for (int ist=0; ist<Nst; ist++) {

        float denom = 1.0f;
        stack st = list.map_d[ist*list.N+i];

        float4 rP1 = tex1Dfetch(r_t, st.iP1);
        float4 rS1 = tex1Dfetch(r_t, st.iS1);
        float4 rB1 = tex1Dfetch(r_t, st.iB1);
        float4 rP2 = tex1Dfetch(r_t, st.iP2);
        float4 rS2 = tex1Dfetch(r_t, st.iS2);
        float4 rB2 = tex1Dfetch(r_t, st.iB2);
        float4 rP3 = tex1Dfetch(r_t, st.iP3);
    
        float4 B2B1 = rB2 - rB1;  // 21
        float4 P1S1 = rP1 - rS1;  // 34
        float4 P2S1 = rP2 - rS1;  // 54 
        float4 P2S2 = rP2 - rS2;  // 56 
        float4 P3S2 = rP3 - rS2;  // 76

        // length between B1 and B2
        float absB2B1 = sqrt(dot_product(B2B1,B2B1));
        float delta = absB2B1 - st.l0;
        denom += st.kl * delta * delta;

        float fact = 2.0f * st.kl * delta / absB2B1;
        float4 ft;
        ft.x = - fact * B2B1.x;
        ft.y = - fact * B2B1.y;
        ft.z = - fact * B2B1.z;

        // dihedral of P1-S1-P2-S2
        float4 m = cross_product(P1S1, P2S1); // normal vectors
        float4 n = cross_product(P2S1, P2S2);
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi10;
        if (delta > CUDART_PI_F) { 
            delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
            delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi1 * delta * delta;

        // dihedral of P3-S2-P2-S1
        m = cross_product(P3S2, P2S2); // normal vectors
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi20;
        if (delta > CUDART_PI_F) { 
           delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
           delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi2 * delta * delta;

  
        fact = st.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}


__global__ void HydrogenBondEnergy(float4* r, InteractionList<hydrogenbond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float energy = 0.0f;
    int Nhb=list.count_d[i];

    for (int ihb=0; ihb<Nhb; ihb++) {
        float denom = 1.0f;
        hydrogenbond hb = list.map_d[ihb*list.N+i];

        float4 r1 = tex1Dfetch(r_t, hb.i1);
        float4 r2 = tex1Dfetch(r_t, hb.i2);
        float4 r3 = tex1Dfetch(r_t, hb.i3);
        float4 r4 = tex1Dfetch(r_t, hb.i4);
        float4 r5 = tex1Dfetch(r_t, hb.i5);
        float4 r6 = tex1Dfetch(r_t, hb.i6);

        float4 v12 = r1 - r2;
        float4 v13 = r1 - r3;
        float4 v53 = r5 - r3;
        float4 v42 = r4 - r2;
        float4 v46 = r4 - r6;

        float d1212 = dot_product(v12,v12);

        // Distance
        float d = sqrt(d1212) - hb.l0;
        denom += hb.kl * d * d;

        // Angle of 3-1=2
        float cos_theta = dot_product(v13,v12) / sqrt(dot_product(v13,v13) * d1212);
        d = acos(cos_theta) - hb.theta10;
        denom += hb.ktheta1 * d * d;

        // Angle of 1=2-4
        cos_theta = dot_product(v12,v42) / sqrt(d1212 * dot_product(v42,v42));
        d = acos(cos_theta) - hb.theta20;
        denom += hb.ktheta2 * d * d;

        // Dihedral of 4-2=1-3
        float4 c4212 = cross_product(v42, v12);
        float4 c1213 = cross_product(v12, v13);
        d = atan2(dot_product(v42,c1213) * sqrt(dot_product(v12,v12)),
                  dot_product(c4212,c1213))
            - hb.psi0;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi * d * d;

        // Dihedral of 5-3-1=2
        float4 m = cross_product(v53, v13);
        float4 n;
        n.x = -c1213.x;
        n.y = -c1213.y;
        n.z = -c1213.z;
        d = atan2(dot_product(v53,n) * sqrt(dot_product(v13,v13)),
                  dot_product(m,n))
            - hb.psi10;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi1 * d * d;

        // Dihedral of 1=2-4-6
        m.x = -c4212.x;
        m.y = -c4212.y;
        m.z = -c4212.z;
        n = cross_product(v42,v46);
        d = atan2(dot_product(v12,n) * sqrt(dot_product(v42,v42)),
                  dot_product(m,n))
            - hb.psi20;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi2 * d * d;

        energy += hb.U0 / denom;
    }

    r[i].w = 0.5*energy; // To eliminate duplication
}



__global__ void HydrogenBondInForce(float4* r, float4* forces, InteractionList<hydrogenbond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 f=forces[i];
    int Nhb=list.count_d[i];

    for (int ihb=0; ihb<Nhb; ihb++) {
        float denom = 1.0f;
        hydrogenbond hb = list.map_d[ihb*list.N+i];

        float4 r1 = tex1Dfetch(r_t, hb.i1);
        float4 r2 = tex1Dfetch(r_t, hb.i2);
        float4 r3 = tex1Dfetch(r_t, hb.i3);
        float4 r4 = tex1Dfetch(r_t, hb.i4);
        float4 r5 = tex1Dfetch(r_t, hb.i5);
        float4 r6 = tex1Dfetch(r_t, hb.i6);

        float4 v12 = r1 - r2;
        float4 v13 = r1 - r3;
        float4 v53 = r5 - r3;
        float4 v42 = r4 - r2;
        float4 v46 = r4 - r6;

        float d1212 = dot_product(v12,v12);
        float d1213 = dot_product(v12,v13);
        float d1313 = dot_product(v13,v13);
        float d4242 = dot_product(v42,v42);
        float d1242 = dot_product(v12,v42);

        float a12 = sqrt(d1212);
        float a13 = sqrt(d1313);
        float a42 = sqrt(d4242);

        float d1213over1212 = d1213 / d1212;
        float d1213over1313 = d1213 / d1313;
        float d1242over1212 = d1242 / d1212;
        float d1353over1313 = dot_product(v13,v53) / d1313;

        // Distance
        float d = a12 - hb.l0;
        denom += hb.kl * d * d;
        float fact = 2.f * hb.kl * d / a12;
        float4 ft;
        ft.x = fact * v12.x;
        ft.y = fact * v12.y;
        ft.z = fact * v12.z;

        // Angle of 3-1=2
        float cos_theta = d1213 / (a13 * a12);
        d = acos(cos_theta) - hb.theta10;
        denom += hb.ktheta1 * d * d;
        fact = -2.f * hb.ktheta1 * d / sqrt(d1313 * d1212 - d1213 * d1213);
        ft.x += fact * (v12.x - d1213over1313 * v13.x + v13.x - d1213over1212 * v12.x);
        ft.y += fact * (v12.y - d1213over1313 * v13.y + v13.y - d1213over1212 * v12.y);
        ft.z += fact * (v12.z - d1213over1313 * v13.z + v13.z - d1213over1212 * v12.z);

        // Angle of 1=2-4
        cos_theta = d1242 / (a12 * a42);
        d = acos(cos_theta) - hb.theta20;
        denom += hb.ktheta2 * d * d;
        fact = -2.f * hb.ktheta2 * d / sqrt(d1212 * d4242 - d1242 * d1242);
        ft.x += fact * (v42.x - d1242over1212 * v12.x);
        ft.y += fact * (v42.y - d1242over1212 * v12.y);
        ft.z += fact * (v42.z - d1242over1212 * v12.z);

        // Dihedral of 4-2=1-3
        float4 c4212 = cross_product(v42, v12);
        float4 c1213 = cross_product(v12, v13);
        d = atan2(dot_product(v42,c1213) * a12, dot_product(c4212,c1213)) - hb.psi0;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi * d * d;
        fact = 2.f * hb.kpsi * d * a12;
        float c4212_abs2 = dot_product(c4212,c4212);
        float c1213_abs2 = dot_product(c1213,c1213);
        ft.x += (1.f - d1213over1212) * fact / c1213_abs2 * c1213.x
                     - d1242over1212  * fact / c4212_abs2 * c4212.x; 
        ft.y += (1.f - d1213over1212) * fact / c1213_abs2 * c1213.y
                     - d1242over1212  * fact / c4212_abs2 * c4212.y; 
        ft.z += (1.f - d1213over1212) * fact / c1213_abs2 * c1213.z
                     - d1242over1212  * fact / c4212_abs2 * c4212.z; 

        // Dihedral of 5-3-1=2
        float4 m = cross_product(v53, v13);
        float4 n;
        n.x = -c1213.x;
        n.y = -c1213.y;
        n.z = -c1213.z;
        float dmm = dot_product(m,m);
        float dnn = c1213_abs2;
        d = atan2(dot_product(v53,n) * a13, dot_product(m,n)) - hb.psi10;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi1 * d * d;
        fact = 2.f * hb.kpsi1 * d * a13;
        ft.x += (1.f - d1213over1313) * fact / dnn * n.x
                     - d1353over1313  * fact / dmm * m.x;
        ft.y += (1.f - d1213over1313) * fact / dnn * n.y
                     - d1353over1313  * fact / dmm * m.y;
        ft.z += (1.f - d1213over1313) * fact / dnn * n.z
                     - d1353over1313  * fact / dmm * m.z;

        // Dihedral of 1=2-4-6
        m.x = -c4212.x;
        m.y = -c4212.y;
        m.z = -c4212.z;
        n = cross_product(v42,v46);
        dmm = dot_product(m,m);
        d = atan2(dot_product(v12,n) * a42, dot_product(m,n))
            - hb.psi20;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi2 * d * d;
        fact = 2.f * hb.kpsi2 * d * a42;
        ft.x += fact / dmm * m.x;
        ft.y += fact / dmm * m.y;
        ft.z += fact / dmm * m.z;


        fact = hb.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}

__global__ void HydrogenBondMidForce(float4* r, float4* forces, InteractionList<hydrogenbond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 f=forces[i];
    int Nhb=list.count_d[i];

    for (int ihb=0; ihb<Nhb; ihb++) {
        float denom = 1.0f;
        hydrogenbond hb = list.map_d[ihb*list.N+i];

        float4 r1 = tex1Dfetch(r_t, hb.i1);
        float4 r2 = tex1Dfetch(r_t, hb.i2);
        float4 r3 = tex1Dfetch(r_t, hb.i3);
        float4 r4 = tex1Dfetch(r_t, hb.i4);
        float4 r5 = tex1Dfetch(r_t, hb.i5);
        float4 r6 = tex1Dfetch(r_t, hb.i6);

        float4 v12 = r1 - r2;
        float4 v13 = r1 - r3;
        float4 v53 = r5 - r3;
        float4 v42 = r4 - r2;
        float4 v46 = r4 - r6;

        float d1212 = dot_product(v12,v12);
        float d1213 = dot_product(v12,v13);
        float d1313 = dot_product(v13,v13);
        float d4242 = dot_product(v42,v42);
        float d1242 = dot_product(v12,v42);
        float d1353 = dot_product(v13,v53);

        float a12 = sqrt(d1212);
        float a13 = sqrt(d1313);
        float a42 = sqrt(d4242);

        float d1213over1313 = d1213 / d1313;
        float d1353over1313 = d1353 / d1313;

        // Distance
        float d = a12 - hb.l0;
        denom += hb.kl * d * d;

        // Angle of 3-1=2
        float cos_theta = d1213 / (a13 * a12);
        d = acos(cos_theta) - hb.theta10;
        denom += hb.ktheta1 * d * d;
        float fact = 2.f * hb.ktheta1 * d / sqrt(d1313 * d1212 - d1213 * d1213);
        float4 ft;
        ft.x = fact * (v12.x - d1213over1313 * v13.x);
        ft.y = fact * (v12.y - d1213over1313 * v13.y);
        ft.z = fact * (v12.z - d1213over1313 * v13.z);

        // Angle of 1=2-4
        cos_theta = d1242 / (a12 * a42);
        d = acos(cos_theta) - hb.theta20;
        denom += hb.ktheta2 * d * d;

        // Dihedral of 4-2=1-3
        float4 c4212 = cross_product(v42, v12);
        float4 c1213 = cross_product(v12, v13);
        d = atan2(dot_product(v42,c1213) * a12,
                  dot_product(c4212,c1213))
            - hb.psi0;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi * d * d;
        fact = -2.f * hb.kpsi * d * a12;
        float c4212_abs2 = dot_product(c4212,c4212);
        float c1213_abs2 = dot_product(c1213,c1213);
        ft.x += fact / c1213_abs2 * c1213.x;
        ft.y += fact / c1213_abs2 * c1213.y;
        ft.z += fact / c1213_abs2 * c1213.z;

        // Dihedral of 5-3-1=2
        float4 m = cross_product(v53, v13);
        float4 n;
        n.x = -c1213.x;
        n.y = -c1213.y;
        n.z = -c1213.z;
        float dmm = dot_product(m,m);
        float dnn = c1213_abs2;
        d = atan2(dot_product(v53,n) * a13, dot_product(m,n)) - hb.psi10;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi1 * d * d;
        fact = 2.f * hb.kpsi1 * d * a13;
        ft.x += (-1.f + d1353over1313) * fact / dmm * m.x
                     +  d1213over1313  * fact / dnn * n.x;
        ft.y += (-1.f + d1353over1313) * fact / dmm * m.y
                     +  d1213over1313  * fact / dnn * n.y;
        ft.z += (-1.f + d1353over1313) * fact / dmm * m.z
                     +  d1213over1313  * fact / dnn * n.z;

        // Dihedral of 1=2-4-6
        m.x = -c4212.x;
        m.y = -c4212.y;
        m.z = -c4212.z;
        n = cross_product(v42,v46);
        d = atan2(dot_product(v12,n) * sqrt(dot_product(v42,v42)),
                  dot_product(m,n))
            - hb.psi20;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi2 * d * d;

        fact = hb.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}

__global__ void HydrogenBondOutForce(float4* r, float4* forces, InteractionList<hydrogenbond> list) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=list.N) return;

    float4 f=forces[i];
    int Nhb=list.count_d[i];

    for (int ihb=0; ihb<Nhb; ihb++) {
        float denom = 1.0f;
        hydrogenbond hb = list.map_d[ihb*list.N+i];

        float4 r1 = tex1Dfetch(r_t, hb.i1);
        float4 r2 = tex1Dfetch(r_t, hb.i2);
        float4 r3 = tex1Dfetch(r_t, hb.i3);
        float4 r4 = tex1Dfetch(r_t, hb.i4);
        float4 r5 = tex1Dfetch(r_t, hb.i5);
        float4 r6 = tex1Dfetch(r_t, hb.i6);

        float4 v12 = r1 - r2;
        float4 v13 = r1 - r3;
        float4 v53 = r5 - r3;
        float4 v42 = r4 - r2;
        float4 v46 = r4 - r6;

        float a12 = sqrt(dot_product(v12,v12));
        float a13 = sqrt(dot_product(v13,v13));

        // Distance
        float d = a12 - hb.l0;
        denom += hb.kl * d * d;

        // Angle of 3-1=2
        float cos_theta = dot_product(v12,v13) / (a13 * a12);
        d = acos(cos_theta) - hb.theta10;
        denom += hb.ktheta1 * d * d;

        // Angle of 1=2-4
        cos_theta = dot_product(v12,v42) / (a12 * sqrt(dot_product(v42,v42)));
        d = acos(cos_theta) - hb.theta20;
        denom += hb.ktheta2 * d * d;

        // Dihedral of 4-2=1-3
        float4 c4212 = cross_product(v42, v12);
        float4 c1213 = cross_product(v12, v13);
        d = atan2(dot_product(v42,c1213) * a12, dot_product(c4212,c1213)) - hb.psi0;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi * d * d;

        // Dihedral of 5-3-1=2
        float4 m = cross_product(v53, v13);
        float4 n;
        n.x = -c1213.x;
        n.y = -c1213.y;
        n.z = -c1213.z;
        float dmm = dot_product(m,m);
        d = atan2(dot_product(v53,n) * a13, dot_product(m,n)) - hb.psi10;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi1 * d * d;
        float fact = 2.f * hb.kpsi1 * d * a13;
        float4 ft;
        ft.x = fact / dmm * m.x;
        ft.y = fact / dmm * m.y;
        ft.z = fact / dmm * m.z;

        // Dihedral of 1=2-4-6
        m.x = -c4212.x;
        m.y = -c4212.y;
        m.z = -c4212.z;
        n = cross_product(v42,v46);
        d = atan2(dot_product(v12,n) * sqrt(dot_product(v42,v42)),
                  dot_product(m,n))
            - hb.psi20;
        if (d > CUDART_PI_F) { 
            d = d - 2.*CUDART_PI_F;
        } else if (d < -CUDART_PI_F) {
            d = d + 2.*CUDART_PI_F;
        }
        denom += hb.kpsi2 * d * d;


        fact = hb.U0 / (denom * denom);

        f.x += fact * ft.x;
        f.y += fact * ft.y;
        f.z += fact * ft.z;
    }
    forces[i]=f;
}
