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
        SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,harmonicbondlist,N/ntraj);
        checkCUDAError("Neighbor List");

        //Force
        force_flush<<<BLOCKS,THREADS>>>(f_d,N);
        checkCUDAError("Force flush");
       
        /**
        FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
        checkCUDAError("FENE");

        HarmonicBondForce<<<BLOCKS,THREADS>>>(r_d,f_d,harmonicbondlist);
        checkCUDAError("HarmonicBond");

        AngleVertexForce<<<BLOCKS,THREADS>>>(r_d,f_d,anglevertexlist);
        checkCUDAError("AngleVertex");

        AngleEndForce<<<BLOCKS,THREADS>>>(r_d,f_d,angleendlist);
        checkCUDAError("AngleEnd");
        **/

        StackP13Force<<<BLOCKS,THREADS>>>(r_d,f_d,stackP13list);
        checkCUDAError("StackP13");

        StackP2Force<<<BLOCKS,THREADS>>>(r_d,f_d,stackP2list);
        checkCUDAError("StackP2");
        
        StackSForce<<<BLOCKS,THREADS>>>(r_d,f_d,stackSlist);
        checkCUDAError("StackS");

        StackBForce<<<BLOCKS,THREADS>>>(r_d,f_d,stackBlist);
        checkCUDAError("StackB");
        
        /**
        SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
        checkCUDAError("SoftSphere");
        
        NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,sig_d);
        checkCUDAError("Native subtract Soft Sphere");
        
        NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist);
        checkCUDAError("Native");
        
        DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
        checkCUDAError("DebyeHuckel");
        **/
        
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

                SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,harmonicbondlist,N/ntraj);
                checkCUDAError("Neighbor List");
    
                // Energy (plus)
                double Efene=0.0;
                double Ehb=0.0;
                double Eang=0.0;
                double Est=0.0;
                double Ess=0.0;
                double Enat=0.0;
                double Eel=0.0;
                /**
                FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Efene");
                       
                for (int i=0; i<N; i++)
                    Efene+=r_h[i].w;

                HarmonicBondEnergy<<<BLOCKS,THREADS>>>(r_d,harmonicbondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ehb");
                       
                for (int i=0; i<N; i++)
                    Ehb+=r_h[i].w;
                        
                AngleEnergy<<<BLOCKS,THREADS>>>(r_d,anglevertexlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Eang");
                       
                for (int i=0; i<N; i++) {
                    Eang+=(double)r_h[i].w;
                }
                **/

                StackEnergy<<<BLOCKS,THREADS>>>(r_d,stackP2list);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Est");
                       
                for (int i=0; i<N; i++) {
                    Est+=(double)r_h[i].w;
                }

                /**
                SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ess");
                    
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
                        
                for (int i=0; i<N; i++)
                    Enat+=r_h[i].w;
                        
                DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Eel");
                        
                for (int i=0; i<N; i++)
                    Eel+=r_h[i].w;
                **/
    
                double Epot_plus=(Efene+Ess+Enat+Eel)/2. + Ehb + Eang + Est;
            
                if (idim==0) r_h[imove].x -= 2.0 * small;
                if (idim==1) r_h[imove].y -= 2.0 * small;
                if (idim==2) r_h[imove].z -= 2.0 * small;
    
                //Copy coordinates to device
                cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);

                SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl,bondlist,harmonicbondlist,N/ntraj);
                checkCUDAError("Neighbor List");
    
                // Energy (minus)
                Efene=0.0;
                Ehb=0.0;
                Eang=0.0;
                Est=0.0;
                Ess=0.0;
                Enat=0.0;
                Eel=0.0;
                /**
                FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Efene");
                       
                for (int i=0; i<N; i++)
                    Efene+=r_h[i].w;

                HarmonicBondEnergy<<<BLOCKS,THREADS>>>(r_d,harmonicbondlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ehb");
                       
                for (int i=0; i<N; i++)
                    Ehb+=r_h[i].w;
                        
                AngleEnergy<<<BLOCKS,THREADS>>>(r_d,anglevertexlist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Eang");
                       
                for (int i=0; i<N; i++) {
                    Eang+=(double)r_h[i].w;
                }
                **/

                StackEnergy<<<BLOCKS,THREADS>>>(r_d,stackP2list);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Est");
                       
                for (int i=0; i<N; i++) {
                    Est+=(double)r_h[i].w;
                }

                /**
                SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Ess");
                    
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
                        
                for (int i=0; i<N; i++)
                    Enat+=r_h[i].w;
                        
                DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Eel");
                        
                for (int i=0; i<N; i++)
                    Eel+=r_h[i].w;
                **/
    
                double Epot_minus=(Efene+Ess+Enat+Eel)/2. + Ehb + Eang + Est;
    
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
