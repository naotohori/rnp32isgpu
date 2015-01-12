#include <math_constants.h>

__device__ float4 operator-(const float4 & a, const float4 & b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device__ float dot_product(const float4 & a, const float4 & b) {
    return (a.x*b.x + a.y*b.y + a.z*b.z);
    // CAUTION: This function does not use a.w or b.w.
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


// This function should be called for every P2 particles
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
    float4 m,n;  // normal vectors
    m.x = P1S1.y * P2S1.z - P1S1.z * P2S1.y;
    m.y = P1S1.z * P2S1.x - P1S1.x * P2S1.z;
    m.z = P1S1.x * P2S1.y - P1S1.y * P2S1.x;
    n.x = P2S1.y * P2S2.z - P2S1.z * P2S2.y;
    n.y = P2S1.z * P2S2.x - P2S1.x * P2S2.z;
    n.z = P2S1.x * P2S2.y - P2S1.y * P2S2.x;
    delta = atan2( dot_product(P1S1,n) * sqrt(dot_product(P2S1,P2S1)), dot_product(m,n)) - st.phi01;
    if (delta > CUDART_PI_F) { 
       delta = delta - 2.*CUDART_PI_F;
    } else if (delta < -CUDART_PI_F) {
       delta = delta + 2.*CUDART_PI_F;
    }
    denom += st.kphi1 * delta * delta;

    // dihedral of P3-S2-P2-S1
    m.x = P3S2.y * P2S2.z - P3S2.z * P2S2.y;
    m.y = P3S2.z * P2S2.x - P3S2.x * P2S2.z;
    m.z = P3S2.x * P2S2.y - P3S2.y * P2S2.x;
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;
    delta = atan2( dot_product(P3S2,n) * sqrt(dot_product(P2S2,P2S2)), dot_product(m,n)) - st.phi02;
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
        float4 m,n;  // normal vectors
        m.x = P1S1.y * P2S1.z - P1S1.z * P2S1.y;
        m.y = P1S1.z * P2S1.x - P1S1.x * P2S1.z;
        m.z = P1S1.x * P2S1.y - P1S1.y * P2S1.x;
        n.x = P2S1.y * P2S2.z - P2S1.z * P2S2.y;
        n.y = P2S1.z * P2S2.x - P2S1.x * P2S2.z;
        n.z = P2S1.x * P2S2.y - P2S1.y * P2S2.x;
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi01;
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
        m.x = P3S2.y * P2S2.z - P3S2.z * P2S2.y;
        m.y = P3S2.z * P2S2.x - P3S2.x * P2S2.z;
        m.z = P3S2.x * P2S2.y - P3S2.y * P2S2.x;
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        delta = atan2( dot_product(P3S2,n) * sqrt(dot_product(P2S2,P2S2)), dot_product(m,n)) - st.phi02;
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
        float4 m,n;  // normal vectors
        m.x = P1S1.y * P2S1.z - P1S1.z * P2S1.y;
        m.y = P1S1.z * P2S1.x - P1S1.x * P2S1.z;
        m.z = P1S1.x * P2S1.y - P1S1.y * P2S1.x;
        n.x = P2S1.y * P2S2.z - P2S1.z * P2S2.y;
        n.y = P2S1.z * P2S2.x - P2S1.x * P2S2.z;
        n.z = P2S1.x * P2S2.y - P2S1.y * P2S2.x;
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi01;
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
        m.x = P3S2.y * P2S2.z - P3S2.z * P2S2.y;
        m.y = P3S2.z * P2S2.x - P3S2.x * P2S2.z;
        m.z = P3S2.x * P2S2.y - P3S2.y * P2S2.x;
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi02;
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
        float4 m,n;  // normal vectors
        m.x = P1S1.y * P2S1.z - P1S1.z * P2S1.y;
        m.y = P1S1.z * P2S1.x - P1S1.x * P2S1.z;
        m.z = P1S1.x * P2S1.y - P1S1.y * P2S1.x;
        n.x = P2S1.y * P2S2.z - P2S1.z * P2S2.y;
        n.y = P2S1.z * P2S2.x - P2S1.x * P2S2.z;
        n.z = P2S1.x * P2S2.y - P2S1.y * P2S2.x;
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi01;
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
        m.x = P3S2.y * P2S2.z - P3S2.z * P2S2.y;
        m.y = P3S2.z * P2S2.x - P3S2.x * P2S2.z;
        m.z = P3S2.x * P2S2.y - P3S2.y * P2S2.x;
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi02;
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
        float4 m,n;  // normal vectors
        m.x = P1S1.y * P2S1.z - P1S1.z * P2S1.y;
        m.y = P1S1.z * P2S1.x - P1S1.x * P2S1.z;
        m.z = P1S1.x * P2S1.y - P1S1.y * P2S1.x;
        n.x = P2S1.y * P2S2.z - P2S1.z * P2S2.y;
        n.y = P2S1.z * P2S2.x - P2S1.x * P2S2.z;
        n.z = P2S1.x * P2S2.y - P2S1.y * P2S2.x;
        float absP2S1 = sqrt(dot_product(P2S1,P2S1));
        delta = atan2( dot_product(P1S1,n) * absP2S1, dot_product(m,n)) - st.phi01;
        if (delta > CUDART_PI_F) { 
            delta = delta - 2.*CUDART_PI_F;
        } else if (delta < -CUDART_PI_F) {
            delta = delta + 2.*CUDART_PI_F;
        }
        denom += st.kphi1 * delta * delta;

        // dihedral of P3-S2-P2-S1
        m.x = P3S2.y * P2S2.z - P3S2.z * P2S2.y;
        m.y = P3S2.z * P2S2.x - P3S2.x * P2S2.z;
        m.z = P3S2.x * P2S2.y - P3S2.y * P2S2.x;
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        float absP2S2 = sqrt(dot_product(P2S2,P2S2));
        delta = atan2( dot_product(P3S2,n) * absP2S2, dot_product(m,n)) - st.phi02;
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
