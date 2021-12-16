#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include "cuComplex.h"

//07/03/15
//Sistema de três níveis em cascata com efeito Doppler
//Plota duas dopplers
//Paralelização nos grupos de átomos

#define Pi 3.141592653589793

#define pontosfreqdiodo 1601
#define blocks 600
#define threads 32
#define Tr 12.5e-9
#define Tp 150e-15
#define pontosFemto 10
#define pontosDecaimento 200
#define OmegaDiodo 2*Pi*0.5*6e6
#define ProfundidadeOptica 2.0
__constant__ int pulsos = 50; //1250
__constant__ float passovelocidade = 1.04*60/blocks;

__constant__ float OmegaFemto = 2*Pi*9e9;
__constant__ float Bf = 0;
__constant__ float Bd = 0;

#define CUDA_ERROR_CHECK
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

__constant__ float q22 = 2*Pi*6e6;
__constant__ float q33 = 2*Pi*6.6e5;
__constant__ float q12 = 2*Pi*0.5*6e6;
__constant__ float q13 = 2*Pi*0.5*6.6e5;
__constant__ float q23 = 2*Pi*0.5*(6e6+6.6e5);
__constant__ float deltaf = 0;
__constant__ int nucleos = blocks*threads;

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
		system("pause");
        exit(-1);
    }
#endif
 
    return;
}

__device__ void f(float a11, float a22, float a33, float a12, float b12, float a13, 
				  float b13, float a23, float b23, float deltad,
				  int j, float &saida, float Ad, float Af, float dop_d, float dop_f)  //sistema de 3 níveis
{
  /*a11*/ if (j==1) saida =  2*Ad*b12 - 2*Bd*a12 + q22*a22;
  /*a22*/ if (j==2) saida = -2*Ad*b12 + 2*Bd*a12 + 2*Af*b23 - 2*Bf*a23 - q22*a22 + q33*a33;
  /*a33*/ if (j==3) saida = -2*Af*b23 + 2*Bf*a23 - q33*a33;
    
  /*a12*/ if (j==4) saida = -q12*a12 - (deltad+dop_d)*b12 + Af*b13 - Bf*a13 - Bd*(a22-a11); //a12
  /*b12*/ if (j==5) saida = -q12*b12 + (deltad+dop_d)*a12 - Af*a13 - Bf*b13 + Ad*(a22-a11); //b12
  /*a13*/ if (j==6) saida = -q13*a13 - (deltad+deltaf+dop_d+dop_f)*b13 + Af*b12 - Ad*b23 + Bf*a12 - Bd*a23;    //a13
  /*b13*/ if (j==7) saida = -q13*b13 + (deltad+deltaf+dop_d+dop_f)*a13 - Af*a12 + Ad*a23 + Bf*b12 - Bd*b23;    //b13
  /*a23*/ if (j==8) saida = -q23*a23 - (deltaf+dop_f)*b23 - Ad*b13 + Bd*a13 - Bf*(a33-a22);  //a23
  /*b23*/ if (j==9) saida = -q23*b23 + (deltaf+dop_f)*a23 + Ad*a13 + Bd*b13 + Af*(a33-a22);  //b23
}

__global__ void Kernel(float *a11, float *a22, float *a33, float *a12, float *b12, float *a13, float *b13, float *a23, float *b23, int *freqd, float *diodo)
{
	const double h1 = Tp/pontosFemto;
	const double h2 = Tr/pontosDecaimento;

	//Paralelização nos grupos de átomos (variável v)

	int n,j,k,pontosTempo;
	float teste;
	float k11,k12,k13,k14;
	float k21,k22,k23,k24;
	float k31,k32,k33,k34;
	float ka12_1,ka12_2,ka12_3,ka12_4, kb12_1,kb12_2,kb12_3,kb12_4;
	float ka13_1,ka13_2,ka13_3,ka13_4, kb13_1,kb13_2,kb13_3,kb13_4;
	float ka23_1,ka23_2,ka23_3,ka23_4, kb23_1,kb23_2,kb23_3,kb23_4;
	float h,Af,deltad,v;
	float dop_d,dop_f;
	float Ad = diodo[0];
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	deltad = (2*Pi)*1e6*freqd[0];

	v = (i-nucleos/2)*passovelocidade;

	dop_d = 2*Pi*v/780e-9;
	dop_f = 2*Pi*v/776e-9;

	for (n=0;n<=2*pulsos+1;n++) //loop pulsos
	{
		if (n % 2 == 0)
		{
			h = h1;
			pontosTempo = pontosFemto;
			Af = OmegaFemto;
		}
		if (n % 2 == 1)
		{
			h = h2;
			pontosTempo = pontosDecaimento;
			Af = 0;
		}
		//{
		
		for (k=1;k<=pontosTempo;k++)     //loop tempo
		{
			//t=t+h;                
                                       
			for (j=1;j<=9;j++)
			{
				f(a11[i],a22[i],a33[i],a12[i],b12[i],a13[i],b13[i],a23[i],b23[i],deltad,j,teste,Ad,Af,dop_d,dop_f);
				
				if (j==1) k11=teste;    if (j==2) k21=teste;    if (j==3) k31=teste;    
				if (j==4) ka12_1=teste; if (j==5) kb12_1=teste; if (j==6) ka13_1=teste; if (j==7)   kb13_1=teste;
				if (j==8) ka23_1=teste; if (j==9) kb23_1=teste;
			}
                                                           
			for (j=1;j<=9;j++)
			{
				f(a11[i]+k11*h/2,a22[i]+k21*h/2,a33[i]+k31*h/2,a12[i]+ka12_1*h/2,b12[i]+kb12_1*h/2,a13[i]+ka13_1*h/2,b13[i]+kb13_1*h/2,
											    a23[i]+ka23_1*h/2,b23[i]+kb23_1*h/2,deltad,j,teste,Ad,Af,dop_d,dop_f);
				if (j==1) k12=teste;    if (j==2) k22=teste;    if (j==3) k32=teste;    
				if (j==4) ka12_2=teste; if (j==5) kb12_2=teste; if (j==6) ka13_2=teste; if (j==7)   kb13_2=teste;
				if (j==8) ka23_2=teste; if (j==9) kb23_2=teste;
			}
                 
			for (j=1;j<=9;j++)
			{
				f(a11[i]+k12*h/2,a22[i]+k22*h/2,a33[i]+k32*h/2,a12[i]+ka12_2*h/2,b12[i]+kb12_2*h/2,a13[i]+ka13_2*h/2,b13[i]+kb13_2*h/2,
									            a23[i]+ka23_2*h/2,b23[i]+kb23_2*h/2,deltad,j,teste,Ad,Af,dop_d,dop_f);
				if (j==1) k13=teste;    if (j==2) k23=teste;    if (j==3) k33=teste;
				if (j==4) ka12_3=teste; if (j==5) kb12_3=teste; if (j==6) ka13_3=teste; if (j==7)   kb13_3=teste;
				if (j==8) ka23_3=teste; if (j==9) kb23_3=teste;
			}
                 
			for (j=1;j<=9;j++)
			{
				f(a11[i]+k13*h,a22[i]+k23*h,a33[i]+k33*h,a12[i]+ka12_3*h,b12[i]+kb12_3*h,a13[i]+ka13_3*h,b13[i]+kb13_3*h,
										    a23[i]+ka23_3*h,b23[i]+kb23_3*h,deltad,j,teste,Ad,Af,dop_d,dop_f);
				if (j==1) k14=teste;    if (j==2) k24=teste;    if (j==3) k34=teste;
				if (j==4) ka12_4=teste; if (j==5) kb12_4=teste; if (j==6) ka13_4=teste; if (j==7)   kb13_4=teste;
				if (j==8) ka23_4=teste; if (j==9) kb23_4=teste;
			}
			a11[i] = a11[i] + h*(k11/6+k12/3+k13/3+k14/6);			   a22[i] = a22[i] + h*(k21/6+k22/3+k23/3+k24/6);
			a33[i] = a33[i] + h*(k31/6+k32/3+k33/3+k34/6);			   		
			a12[i] = a12[i] + h*(ka12_1/6+ka12_2/3+ka12_3/3+ka12_4/6); b12[i] = b12[i] + h*(kb12_1/6+kb12_2/3+kb12_3/3+kb12_4/6);
			a13[i] = a13[i] + h*(ka13_1/6+ka13_2/3+ka13_3/3+ka13_4/6); b13[i] = b13[i] + h*(kb13_1/6+kb13_2/3+kb13_3/3+kb13_4/6); 
			a23[i] = a23[i] + h*(ka23_1/6+ka23_2/3+ka23_3/3+ka23_4/6); b23[i] = b23[i] + h*(kb23_1/6+kb23_2/3+kb23_3/3+kb23_4/6);
 
		}  //loop tempo

	}	//loop pulsos

}

int main()
{
	clock_t begin, end;
	double time_spent;
	begin = clock();

	FILE *arquivo;
	arquivo=fopen("dados.dat","w");
	const int nucleos = blocks*threads;

	cudaDeviceProp prop;
	int count,d;

	cudaGetDeviceCount(&count);
	for (int s=0;s<count;s++)
	{
		cudaGetDeviceProperties(&prop,s);
		printf("General information\n");
		printf("%s", prop.name);
		printf(", Compute Capability %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d MHz\n", int(prop.clockRate/1000));
		printf("Kernel execition timeout:  ");
		if   (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
									   else  printf("Disabled\n");
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d)\n", prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared memory per MP: %ld kb\n", prop.sharedMemPerBlock/1024);
		printf("Registers per MP: %d\n", prop.regsPerBlock);
		printf("Warp size: %ld\n\n", prop.warpSize);
		//printf("Total Global Memory: %d\n\n", prop.totalGlobalMem);
	}

	printf("Blocks = %d\n", blocks);
	printf("Threads = %d\n\n", threads);

	printf("Calculando...\n");

	float H11, H22, H33;
	float H12, B12, H13, B13;
	float H23, B23;

	float a11[nucleos]; float a22[nucleos];
	float a33[nucleos]; 
	float a12[nucleos]; float b12[nucleos];
	float a23[nucleos]; float b23[nucleos];
	float a13[nucleos]; float b13[nucleos];
	int freqd[1]; float diodo[1],campodiodo[pontosfreqdiodo];
	float doppler,soma,alpha,N;

	float *dev_a11; float *dev_a22;
	float *dev_a33;
	float *dev_a12; float *dev_b12;
	float *dev_a13; float *dev_b13;
	float *dev_a23; float *dev_b23;
	float *dev_diodo;
	int   *dev_freqd;

	cudaMalloc((void**)&dev_a11, nucleos * sizeof(float)); cudaMalloc((void**)&dev_a22, nucleos * sizeof(float));
	cudaMalloc((void**)&dev_a33, nucleos * sizeof(float)); 
	cudaMalloc((void**)&dev_a12, nucleos * sizeof(float)); cudaMalloc((void**)&dev_b12, nucleos * sizeof(float));
	cudaMalloc((void**)&dev_a23, nucleos * sizeof(float)); cudaMalloc((void**)&dev_b23, nucleos * sizeof(float));
	cudaMalloc((void**)&dev_a13, nucleos * sizeof(float)); cudaMalloc((void**)&dev_b13, nucleos * sizeof(float));
	cudaMalloc((void**)&dev_diodo, 1 * sizeof(float));
	cudaMalloc((void**)&dev_freqd, 1 * sizeof(float));

    // Choose which GPU to run on, change this on a multi-GPU system.
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    //cudaStatus = cudaMalloc((void**)&dev_a, nucleos * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

	alpha=1;
	N = ProfundidadeOptica;
	fprintf(arquivo, "diode_frequency rho11 rho22 rho33 soma\n");
	for (d=0;d<=pontosfreqdiodo-1;d++)  //loop diodo
	{
		freqd[0] = d-(pontosfreqdiodo-1)/2;

		alpha = N;	

		campodiodo[d] =  exp(      -alpha*exp( -0.5*freqd[0]*freqd[0]/200/200 ) );
		doppler = 0.25/(sqrt(2.0*Pi)*200)*exp( -0.5*freqd[0]*freqd[0]/200/200 )*60/blocks;
		diodo[0] = OmegaDiodo*campodiodo[d];

		for (int q=0;q<=nucleos-1;q++)
		{
			a11[q] = 1; a22[q] = 0;
			a33[q] = 0; 
			a12[q] = 0; b12[q] = 0;
			a13[q] = 0; b13[q] = 0;
			a23[q] = 0; b23[q] = 0;
		}

		cudaMemcpy(dev_a11, a11, nucleos * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(dev_a22, a22, nucleos * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_a33, a33, nucleos * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_a12, a12, nucleos * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(dev_b12, b12, nucleos * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_a23, a23, nucleos * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(dev_b23, b23, nucleos * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_a13, a13, nucleos * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(dev_b13, b13, nucleos * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_freqd, freqd,   1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_diodo, diodo,   1 * sizeof(float), cudaMemcpyHostToDevice);

		Kernel<<<blocks,threads>>>(dev_a11, dev_a22, dev_a33, dev_a12, dev_b12, dev_a13, dev_b13, dev_a23, dev_b23, dev_freqd, dev_diodo);
		CudaCheckError();
    
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaDeviceSynchronize();

		cudaMemcpy(a11, dev_a11, nucleos * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(a22, dev_a22, nucleos * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(a33, dev_a33, nucleos * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(a12, dev_a12, nucleos * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(b12, dev_b12, nucleos * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(a13, dev_a13, nucleos * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(b13, dev_b13, nucleos * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(a23, dev_a23, nucleos * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(b23, dev_b23, nucleos * sizeof(float), cudaMemcpyDeviceToHost);

		for (int q=0;q<=nucleos-1;q++)
		{
			H11 = H11 + a11[q]*doppler; H22 = H22 + a22[q]*doppler;
			H33 = H33 + a33[q]*doppler;
			H12 = H12 + a12[q]*doppler; B12 = B12 + b12[q]*doppler;
			H13 = H13 + a13[q]*doppler; B13 = B13 + b13[q]*doppler;
			H23 = H23 + a23[q]*doppler; B23 = B23 + b23[q]*doppler;
		}

		soma = H11 + H22 + H33;
		printf(          "%d %.9f  %.10f %.13f %.13f\n", freqd[0], H11, H22, H33, soma);
		fprintf(arquivo, "%d %.9f  %.10f %.13f %.13f\n", freqd[0], H11, H22, H33, soma);

		H11 = H22 = H33 = 0;
		H12 = B12 = H13 = B13 = 0;
		H23 = B23 = 0;
		
	} //loop diodo

	cudaFree(dev_a11);cudaFree(dev_a22);
	cudaFree(dev_a33);
	cudaFree(dev_a12);cudaFree(dev_b12);
	cudaFree(dev_a13);cudaFree(dev_b13);
	cudaFree(dev_a23);cudaFree(dev_b23);
	cudaFree(dev_freqd);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (time_spent<=60) printf("\nTempo de execucao = %f s\n\n", time_spent);
	if (time_spent> 60 && time_spent<= 3600) printf("\nTempo de execucao = %f min\n\n", time_spent/60);
	if (time_spent>3600) printf("\nTempo de execucao = %f h\n\n", time_spent/3600);

	printf("\a");
	system ("pause");
	fclose(arquivo);
}