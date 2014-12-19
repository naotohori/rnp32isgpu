#include <stdio.h>
#include <vector_types.h>


//Output frame in XYZ format
void writexyz(FILE* traj, float4* r, int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}

//Output frame in XYZ format multiple trajectiories
void writexyz(FILE** traj, float4* r, int Naa, int Nnuc, int ntraj) {
    for (int itraj=0; itraj<ntraj; itraj++) {
        fprintf(traj[itraj],"%d\nAtoms\n",2*Naa+3*Nnuc);
        for (int i=(2*Naa+3*Nnuc)*itraj;i<(2*Naa+3*Nnuc)*itraj+Naa;i++) {
            fprintf(traj[itraj],"CA %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
        for (int i=(2*Naa+3*Nnuc)*itraj+Naa;i<(2*Naa+3*Nnuc)*itraj+2*Naa;i++) {
            fprintf(traj[itraj],"CB %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
        for (int i=(2*Naa+3*Nnuc)*itraj+2*Naa;i<(2*Naa+3*Nnuc)*itraj+2*Naa+Nnuc;i++) {
            fprintf(traj[itraj],"P %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
        for (int i=(2*Naa+3*Nnuc)*itraj+2*Naa+Nnuc;i<(2*Naa+3*Nnuc)*itraj+2*Naa+2*Nnuc;i++) {
            fprintf(traj[itraj],"S %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
        for (int i=(2*Naa+3*Nnuc)*itraj+2*Naa+2*Nnuc;i<(2*Naa+3*Nnuc)*itraj+2*Naa+3*Nnuc;i++) {
            fprintf(traj[itraj],"B %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
    }
}

//Output frame in XYZ format with translation on vector t (typically, center-of-mass)
void writexyz(FILE* traj, float4* r, float3 t,int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
}

//Write forces (needed only for debugging)
void writeforces(FILE* traj, float4* r, int Naa) {
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}

//Read coordinates from input file
void readcoord(FILE* ind, float4* r, int N) {
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%f %f %f", &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
    }
}

//Read coordinates from input file, multiple trajectories
void readcoord(FILE* ind, float4* r, int N, int ntraj) {
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%f %f %f", &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
        for (int itraj=1; itraj<ntraj; itraj++)
            r[itraj*N+i]=r[i];
    }
}

//Read coordinates from XYZ file
void readxyz(FILE* ind, float4* r, int N) {
    char name[80];
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%s %f %f %f", name, &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
    }
}

//Read coordinates from XYZ file, multiple trajectories
void readxyz(FILE* ind, float4* r, int N, int ntraj) {
    char name[80];
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%s %f %f %f", name, &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
        for (int itraj=1; itraj<ntraj; itraj++)
            r[itraj*N+i]=r[i];
    }
}
