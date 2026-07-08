#include <CL/cl.h>
#include <stdio.h>
int main(){
  cl_uint np=0; clGetPlatformIDs(0,NULL,&np); printf("platforms=%u\n",np);
  cl_platform_id p[8]; if(np>8)np=8; clGetPlatformIDs(np,p,NULL);
  for(cl_uint i=0;i<np;i++){
    char name[256]=""; clGetPlatformInfo(p[i],CL_PLATFORM_NAME,256,name,NULL);
    cl_uint nd=0; cl_int e=clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,0,NULL,&nd);
    printf("platform %u: '%s'  gpu_devices=%u (getDeviceIDs err %d)\n",i,name,nd,e);
    if(nd>0){
      cl_device_id d[8]; if(nd>8)nd=8; clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,nd,d,NULL);
      char dn[256]=""; clGetDeviceInfo(d[0],CL_DEVICE_NAME,256,dn,NULL);
      cl_context_properties props[]={CL_CONTEXT_PLATFORM,(cl_context_properties)p[i],0};
      cl_int ce; cl_context ctx=clCreateContext(props,1,d,NULL,NULL,&ce);
      printf("  device0='%s'  clCreateContext err=%d %s\n",dn,ce,ce==0?"SUCCESS":"FAIL");
      if(ce==0){ cl_int qe; clCreateCommandQueue(ctx,d[0],0,&qe); printf("  clCreateCommandQueue err=%d\n",qe);}
      // also try the no-properties form
      cl_int ce2; clCreateContext(NULL,1,d,NULL,NULL,&ce2);
      printf("  clCreateContext(NULL props) err=%d %s\n",ce2,ce2==0?"SUCCESS":"FAIL");
    }
  }
  return 0;
}
