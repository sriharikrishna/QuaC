
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
operator *qubits;
Vec rho_init,rho_tmp;
FILE *f_pop;


int main(int argc,char **args){
    PetscReal time_max,dt,*gamma_1,*gamma_2,*omega,*sigma_x,*sigma_y,*sigma_z;
    PetscReal gate_time_step,theta,fidelity,t1,t2;
    PetscScalar mat_val;
    PetscInt  steps_max;
    Vec rho;
    int num_qubits,i,j,h_dim,system,gate_count;
    circuit teleportation,teleportation2;
    Mat circ_mat,circ_mat2;
    PetscViewer    mat_view;
    
    /* Initialize QuaC */
    QuaC_initialize(argc,args);
    
    num_qubits = 5;
    theta = 0.0;
    PetscOptionsGetReal(NULL,NULL,"-gam1",&gamma_1[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-gam2",&gamma_1[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-gam3",&gamma_1[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-gam4",&gamma_1[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-gam5",&gamma_1[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-dep1",&gamma_2[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-dep2",&gamma_2[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-dep3",&gamma_2[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-dep4",&gamma_2[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-dep5",&gamma_2[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-sigx1",&sigma_x[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigx2",&sigma_x[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigx3",&sigma_x[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigx4",&sigma_x[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigx5",&sigma_x[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-sigy1",&sigma_y[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy2",&sigma_y[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy3",&sigma_y[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy4",&sigma_y[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy5",&sigma_y[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-sigz1",&sigma_z[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigz2",&sigma_z[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigz3",&sigma_z[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy4",&sigma_y[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-sigy5",&sigma_y[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-om1",&omega[0],NULL);
    PetscOptionsGetReal(NULL,NULL,"-om2",&omega[1],NULL);
    PetscOptionsGetReal(NULL,NULL,"-om3",&omega[2],NULL);
    PetscOptionsGetReal(NULL,NULL,"-om4",&omega[3],NULL);
    PetscOptionsGetReal(NULL,NULL,"-om5",&omega[4],NULL);
    
    PetscOptionsGetReal(NULL,NULL,"-theta",&theta,NULL);
    PetscOptionsGetInt(NULL,NULL,"-num_qubits",&num_qubits,NULL);
    qubits  = malloc(num_qubits*sizeof(struct operator)); //Only need 3 qubits for teleportation
    gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    gamma_2 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    sigma_x = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    sigma_y = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    sigma_z = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    omega   = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
    for (i=0;i<num_qubits;i++){
        create_op(2,&qubits[i]);
        omega[i]   = 0.0;
        gamma_1[i] = 0;
        gamma_2[i] = 0;
        sigma_x[i] = 0;
        sigma_y[i] = 0;
        sigma_z[i] = 0;
    }
    
    /* Add terms to the hamiltonian */
    for (i=0;i<num_qubits;i++){
        add_to_ham(omega[i],qubits[i]->n);
        /* qubit decay */
        add_lin(gamma_1[i],qubits[i]);
        add_lin(gamma_2[i],qubits[i]->n);
        add_lin(sigma_x[i],qubits[i]->sig_x);
        add_lin(sigma_y[i],qubits[i]->sig_y);
        add_lin(sigma_z[i],qubits[i]->sig_z);
    }
    
    printf("gam: %f %f %f %f %f\n",gamma_1[0],gamma_1[1],gamma_1[2],gamma_1[3],gamma_1[4]);
    printf("dep: %f %f %f %f %f\n",gamma_2[0],gamma_2[1],gamma_2[2],gamma_2[3],gamma_2[4]);
    printf("sigx: %f %f %f %f %f\n",sigma_x[0],sigma_x[1],sigma_x[2],sigma_x[3],sigma_x[4]);
    printf("sigy: %f %f %f %f %f\n",sigma_y[0],sigma_y[1],sigma_y[2],sigma_y[3],sigma_y[4]);
    printf("sigz: %f %f %f %f %f\n",sigma_z[0],sigma_z[1],sigma_z[2],sigma_z[3],sigma_z[4]);
    printf("om: %f %f %f %f %f\n",omega[0],omega[1],omega[2],omega[3],omega[4]);
    printf("theta: %f\n",theta);
    
    time_max  = 9;
    dt        = 0.01;
    steps_max = 1000;
    
    /* Set the ts_monitor to print results at each time step */
    set_ts_monitor(ts_monitor);
    /* Open file that we will print to in ts_monitor */
    if (nid==0){
        f_pop = fopen("pop","w");
        fprintf(f_pop,"#Time Populations\n");
    }
    
    create_full_dm(&rho);
    
    create_dm(&rho_init,4);
    create_dm(&rho_tmp,4);
    printf("create_circuit gatelist length %d", num_qubits+((num_qubits+1)*num_qubits)/2);
    create_circuit(&teleportation,num_qubits+5*((num_qubits+1)*num_qubits)/2);
    
    //Set the initial DM
    mat_val = cos(theta)*cos(theta);
    mat_val = 0.5;
    add_value_to_dm(rho,0,0,mat_val);
    /* mat_val = cos(theta)*sin(theta); */
    mat_val = -0.5;
    add_value_to_dm(rho,4,0,mat_val);
    add_value_to_dm(rho,0,4,mat_val);
    /* mat_val = sin(theta)*sin(theta); */
    mat_val = 0.5;
    add_value_to_dm(rho,4,4,mat_val);
    assemble_dm(rho);
    
    /* print_dm(rho,8); */
    h_dim = pow(2,num_qubits);
    gate_time_step = 1.0;
    gate_count=0;
    
    for(i=0;i<num_qubits;i++){
        add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,RX,i);
        printf("RX %d gate_count %d\n",i,gate_count);

    }
    
    //QFT for num_qubits
    add_gate_to_circuit(&teleportation,gate_count++*gate_time_step,HADAMARD,0);
    printf("HADAMARD %d gate_count %d\n",0,gate_count);
    for(i=1;i<num_qubits;i++){
        for(j=0;j<i;j++){
          add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,U3, i,0,0,PETSC_PI/pow(2,(i-j-1)));
          printf("U3 %d PetscPi/%lf gate_count%d \n",i,pow(2,(i-j-1)),gate_count);
            
          add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,CNOT,i,j);
          printf("CNOT %d %d gate_count%d \n",i,j,gate_count);
            
          add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,U3, j,0,0,-PETSC_PI/pow(2,(i-j-1)));
          printf("U3 %d PetscPi/%lf gate_count%d \n",j,-pow(2,(i-j-1)),gate_count);
            
          add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,CNOT,i,j);
          printf("CNOT %d %d gate_count%d \n",i,j,gate_count);
            
          add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,U3, j,0,0,PETSC_PI/pow(2,(i-j-1)));
          printf("U3 %d PetscPi/%lf gate_count%d \n",j,pow(2,(i-j-1)),gate_count);

        }
        add_gate_to_circuit(&teleportation,(gate_count++)*gate_time_step,HADAMARD,i);
        printf("HADAMARD %d gate_count %d\n",i,gate_count);
    }
    
    combine_circuit_to_mat(&circ_mat,teleportation);
    start_circuit_at_time(&teleportation,0.0);
    
    time_step(rho,0.0,time_max,dt,steps_max);
    
    //  steady_state(rho);:
    partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
    //partial_trace_over(rho,rho_tmp,3,qubits[0],qubits[1],qubits[2]);
    
    get_fidelity(rho_init,rho_tmp,&fidelity);
    if(nid==0) printf("Final fidelity: %f\n",fidelity);
    
    //  partial_trace_over(rho,rho_init,2,qubits[0],qubits[1]);
    //print_dm(rho,h_dim);
    //if(nid==0) printf("Final PTraced DM: C?\n");
    //print_dm(rho_tmp,1);
    //if(nid==0) printf("Final PTraced DM: 0\n");
    //partial_trace_over(rho,rho_tmp,1,qubits[0]);
    //print_dm(rho_tmp,1);
    //if(nid==0) printf("Final PTraced DM: 1\n");
    //partial_trace_over(rho,rho_tmp,1,qubits[1]);
    //print_dm(rho_tmp,1);
    
    destroy_dm(rho_init);
    destroy_dm(rho_tmp);
    for (i=0;i<num_qubits;i++){
        destroy_op(&qubits[i]);
    }
    free(qubits);
    destroy_dm(rho);
    QuaC_finalize();
    return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
    double fidelity,*populations;
    int num_pop,i;
    
    num_pop = get_num_populations();
    populations = malloc(num_pop*sizeof(double));
    get_populations(rho,&populations);
    if (nid==0){
        /* Print populations to file */
        fprintf(f_pop,"%e",time);
        for(i=0;i<num_pop;i++){
            fprintf(f_pop," %e ",populations[i]);
        }
        fprintf(f_pop,"\n");
    }
    
    //partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
    partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
    //partial_trace_over(rho,rho_tmp,3,qubits[0],qubits[1],qubits[2]);

    //partial_trace_over(rho,rho_tmp,1,qubits[1]);
    
    get_fidelity(rho_init,rho_tmp,&fidelity);
    if(nid==0) printf("%f %f\n",time,fidelity);
    free(populations);
    //  print_dm(dm,2);
    PetscFunctionReturn(0);
    
}

