import os
from ANNarchy import * 
import scipy.spatial.distance
import time
import numpy as np
import statistics
import optuna
import pandas as pd


def simular(tiempoExp, umbralActividad, ruidoPM, ruidoTalamo, ruidoSNr, ruidoGPe, ruidoStrD2, ruidoStrD1, ruidoSTN, ruidoTonos, ruidoObjetivos, tauObjStrD2, tauObjStrD1, tauTonesStrD2, tauTonesStrD1, tauTonesSTN, tauStrD2GPe, tauGPeSNr, tauStrD1SNr, tauSTNSNr, tauSNrVA, tauVAPM):
    
    clear()

    #Organización de carpetas
    #creación carpeta de salidas
    output_dir = os.path.join(os.getcwd(), "salidas")
    os.makedirs(output_dir, exist_ok=True)

    #Creación de carpetas para archivos sin uso
    unused_dir = os.path.join(output_dir, "Sin uso")
    os.makedirs(unused_dir, exist_ok=True)



    tiempoTrial = tiempoExp           #indica el tiempo que se puede esperar cada trial antes que se considere nulo por no presionar el botón
    umbralRespuesta = umbralActividad     #indica la actividad necesaria como meta para inciciar el proceso de softmax para presionar un botón
    multSTN = 100.0               #multiplicador de potencia de tono no-go sobre activación de STN



    contadorCoincidencia1NoRev = 0
    contadorNoCoincide1NoRev = 0
    contadorSkipMalo1NoRev = 0
    contadorCoincidencia1Rev = 0
    contadorNoCoincide1Rev = 0
    contadorSkipMalo1Rev = 0
    contadorCoincidencia2 = 0
    contadorNoCoincide2 = 0
    contadorSkipMalo2 = 0
    contadorCoincidencia3 = 0
    contadorNoCoincide3 = 0
    contadorSkipBueno3 = 0
    contadorSkipMalo3 = 0
    contadorSkipRespondido3 = 0

    #lista de tiempos para condición de tiempos exp 1 rev>noRev 
    tiemposNoRev = []
    tiemposRev = []

    listaTiemposGlobal1norev = []
    listaTiemposGlobal1Rev = []
    listaTiemposGlobal2 = []
    listaTiemposGlobal3 = []
    listaTonos = []
    listaCoincidencia1Rev = []
    listaCoincidencia1noRev = []
    listaCoincidencia2 = []
    listaCoincidencia3 = []


    #General networks parameters
    baseline_dopa = 0.1


    # PREGUNTAS
    #Neuron models

    LinearNeuron = Neuron(
        parameters= """
            tau = 10.0
            baseline = 0.0
            noise = 0.0
            lesion = 1.0

        """,
        equations="""
            tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
            r = lesion*pos(mp)
        """
    )


    # NO SE USA AL PARECER
    '''LinearNeuron_saturated = Neuron(
        parameters= """
            mean_point = 0.0
            slope = 3.0
            tau = 10.0
            baseline = 0.0
            noise = 0.0
        """,
        equations="""
            tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
            r = sigmoid(slope*(mp-mean_point))
        """,
        functions = """
            sigmoid(x) = 1.0 / (1.0 - x))
        """
    )'''
    # Traza
    LinearNeuron_trace = Neuron(
        parameters= """
            tau = 10.0
            baseline = 0.0
            noise = 0.0
            tau_trace = 120.0
            lesion = 1.0
        """,
        equations="""
            tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
            r = lesion*pos(mp)
            tau_trace*dtrace/dt + trace = r
        """
    )
    # Ecuacion de la Dopamina
    DopamineNeuron = Neuron(
        parameters="""
            tau = 10.0
            firing = 0
            inhibition = 0.0
            baseline = 0.0
            exc_threshold = 0.0
            factor_inh = 10.0
        """,
        equations="""
            ex_in = if (sum(exc)>exc_threshold): 1 else: 0
            s_inh = sum(inh)
            aux = if (firing>0): (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh)+baseline)  else: baseline
            tau*dmp/dt + mp =  aux
            r = if (mp>0.0): mp else: 0.0
        """
    )
    # ESTIMULOS? 
    InputNeuron = Neuron(
        parameters="""
            tau = 1.5
            baseline = 0.0
        """,
        equations="""
            tau*dmp/dt + mp = baseline
        r = if (mp>0.0): mp else: 0.0
        """


    )
    # NO SE USA AL PARECER

    '''InputNeuron_preference = Neuron(
        parameters="""
            A = 1.0
            a = 15.
            distance = 0.0
            firing = 0.0
        """,
        equations="""
        r = if (firing>0): A*(-(distance*distance)/a) else: 0
            
        """

    )'''

    ###################################################################################################################################################
    ###################################################################################################################################################
    # ECUACIONES PARA LAS CONEXIONES?
    #Synapse models
    # POS EN LA post sinaptica
    # NO SE UTILIZA
    '''PostCovariance = Synapse(
        parameters="""
            tau = 1000.0
            tau_alpha = 10.0 
            regularization_threshold = 1.0
            threshold_post = 0.0
            threshold_pre = 0.0
        """,
        equations="""
            tau_alpha*dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) 


            trace = (pre.r - mean(pre.r) - threshold_pre) * pos(post.r - mean(post.r) - threshold_post)
        delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post)*w)
            tau*dw/dt = delta : min=0
    """
    )'''
    # Pre en la presinaptica
    '''PreCovariance = Synapse(
        parameters="""
            tau = 1000.0
            tau_alpha = 10.0 
            regularization_threshold = 1.0
            threshold_post = 0.0
            threshold_pre = 0.0
        """,
        equations="""
            tau_alpha*dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) 


            trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
        delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post)*w)
            tau*dw/dt = delta : min=0
    """
    )'''
    # CUAndo traspasa ese umbral en vez de  inhibirse se excitan
    ReversedSynapse = Synapse(
        parameters="""
            reversal = 0.3
        """,
        psp="""
            w*pos(reversal-pre.r)
        """    

    )

    #DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
    DAPostCovarianceNoThreshold = Synapse(
        parameters="""
            tau=1000.0
            tau_alpha=10.0 
            regularization_threshold=1.0 
            baseline_dopa = 0.1
            K_burst = 1.0
            K_dip = 0.4
            DA_type = 1 
            threshold_pre=0.0
            threshold_post=0.0
        """,
        equations="""
            tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold) 
            dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

            trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)

        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0

            dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum

            

            delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
            tau*dw/dt = delta : min=0 
        """


    )


    DAPostCovarianceNoThreshold_trace = Synapse(
        parameters="""
            tau=1000.0
            tau_alpha=10.0 
            regularization_threshold=1.0 
            baseline_dopa = 0.1
            K_burst = 1.0
            K_dip = 0.4
            DA_type = 1 
            threshold_pre=0.0
            threshold_post=0.0
        """,
        equations="""
            tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold) 
            dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)

            trace = pos(post.trace -  mean(post.trace) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)

        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0

            dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum

            

            delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
            tau*dw/dt = delta : min=0 
        """


    )

    #tau*dw/dt = delta : min=0 

    #Excitatory synapses STN -> SNr
    DAPreCovariance_excitatory = Synapse(
        parameters="""
        tau=1000.0
        tau_alpha=10.0 
        regularization_threshold=1.0 
        baseline_dopa = 0.1  
        K_burst = 1.0
        K_dip = 0.4
        DA_type= 1
        threshold_pre=0.0
        threshold_post=0.0
        """,
        equations = """
            tau_alpha*dalpha/dt  = pos( post.mp - regularization_threshold) - alpha
            dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

            trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
            aux = if (trace<0.0): 1 else: 0
            dopa_mod = if (dopa_sum>0): K_burst * dopa_sum else: K_dip * dopa_sum * aux
            delta = dopa_mod * trace - alpha * pos(trace)
            tau*dw/dt = delta : min=0 

            
        """

    )


    #Inhibitory synapses SNr -> SNr and STRD2 -> GPe
    DAPreCovariance_inhibitory = Synapse(
        parameters="""
        tau=1000.0
        tau_alpha=10.0 
        regularization_threshold=1.0 
        baseline_dopa = 0.1    
        K_burst = 1.0
        K_dip = 0.4
        DA_type= 1
        threshold_pre=0.0
        threshold_post=0.0
        negg = 1
        """,
        equations="""
            tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
            dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

            trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
            aux = if (trace>0): negg else: 0

            dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
            trace2 = trace

            delta = dopa_mod * trace2 - alpha * pos(trace2)
            tau*dw/dt = delta : min=0 
        """


    )


    DAPreCovariance_inhibitory_trace = Synapse(
        parameters="""
        tau=1000.0
        tau_alpha=10.0 
        regularization_threshold=1.0 
        baseline_dopa = 0.1 
        K_burst = 1.0
        K_dip = 0.4
        DA_type= 1
        threshold_pre=0.0
        threshold_post=0.0
        """,
        equations="""
            tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
            dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa) 

            trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.trace) - post.trace  - threshold_post)
            aux = if (trace>0): 1 else: 0

            dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
            trace2 = trace

            delta = dopa_mod * trace2 - alpha * pos(trace2)
            tau*dw/dt = delta : min=0 
        """


    )
    # trace2 = trace - (1-aux)*(trace/2.)

    DAPrediction = Synapse(
        parameters="""
            tau = 100000.0
            baseline_dopa = 0.1
    """,
    equations="""
        aux = if (post.sum(exc)>0): 1.0 else: 3.0
        delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
        tau*dw/dt = delta : min=0 
    """


    )



    ###################################################################################################################################################

    ###################################################################################################################################################

    #CORTICAL NEURONS

    # ESTIMULOS ?
    Input_neurons = Population(name='InputNeurons',geometry=40,neuron=InputNeuron)
    #Input_neurons.tau = 40.0

    Input_neurons_tones = Population(name='ToneInputNeurones',geometry=4,neuron=InputNeuron)
    #Input_neurons_tones.tau = 40.0#10.0
    #Input_neurons_tones.noise = 0.01
    #Input_neurons.tau = 40.0

    Input_neurons_reversal_group_strd1 = Population(name='ReversalGrouSTRD1Input',geometry=3,neuron=InputNeuron)
    Input_neurons_reversal_group_VA = Population(name='ReversalGroupVAInput',geometry=3,neuron=InputNeuron)

    # NO TOMARLO EN CUENTA
    Context = Population(name='Context',geometry=2,neuron=InputNeuron)

    # PM
    PM = Population(name="PM", geometry = 2, neuron=LinearNeuron)
    PM.tau = 30.0
    PM.noise = 0.01

    # ESTADO DEL MAPA
    #Neurons to represent the state of the joystick
    Propio = Population(name="Propio", geometry=3, neuron=LinearNeuron)
    Propio.tau = 10.0
    Propio.noise = 0.01
    # OBJETIVOS
    #Neurons to represent the different objectives
    # PUTAMEN
    Objectives = Population(name="Objectives", geometry=3, neuron=LinearNeuron)
    Objectives.tau = 40.0#10.0
    Objectives.noise = 0.01

    # CAUDATE
    Objectives_extra = Population(name="ObjectivesExtra", geometry=15, neuron=LinearNeuron)
    Objectives_extra.tau = 40.0#10.0
    Objectives_extra.noise = 0.01


    #New IL
    # El atajo
    IL = Population(name='IL',geometry=2,neuron=LinearNeuron)
    IL.tau = 10
    IL.baseline = 0.0
    IL.noise = 0.1

    Saturation = Population(name='Saturation',geometry=1,neuron=LinearNeuron)
    Saturation.baseline = 0.0
    Saturation.tau=10.0

    #ASSOCIATIVE LOOP
    # CAUDATE = DORSOMEDIAL
    # PUTAMEN = DORSOLATERAl

    # PORQUE HAY DOS TIPO DE D1?

    # Striatum direct pathway
    StrD1_caudate0 = Population(name="StrD1_caudate0", geometry=(2,2),neuron = LinearNeuron)
    StrD1_caudate0.tau = 10.0
    StrD1_caudate0.noise = 0.3 #0.08
    StrD1_caudate0.baseline = 0.0
    #StrD1_caudate.slope = 3.0
    #StrD1_caudate.mean_point = 0.8

    StrD1_caudate1 = Population(name="StrD1_caudate1", geometry=(2,2),neuron = LinearNeuron)
    StrD1_caudate1.tau = 10.0
    StrD1_caudate1.noise = 0.3 #0.08
    StrD1_caudate1.baseline = 0.0
    #StrD1_caudate.slope = 3.0
    #StrD1_caudate.mean_point = 0.8


    # Striatum indirect pathway
    StrD2_caudate0 = Population(name="StrD2_caudate0", geometry = (3,3), neuron=LinearNeuron)
    StrD2_caudate0.tau = 10.0
    StrD2_caudate0.noise = 0.01
    StrD2_caudate0.baseline = 0.0#0.4 

    StrD2_caudate1 = Population(name="StrD2_caudate1", geometry = (3,3), neuron=LinearNeuron)
    StrD2_caudate1.tau = 10.0
    StrD2_caudate1.noise = 0.01
    StrD2_caudate1.baseline = 0.0#0.4 

    # Striatum feedback pathway
    StrThal_caudate = Population(name="StrThal_caudate", geometry = 2, neuron=LinearNeuron)
    StrThal_caudate.tau = 10.0
    StrThal_caudate.noise = 0.01
    StrThal_caudate.baseline = 0.4

    # PORQUE TRAZA?
    # SNr
    SNr_caudate = Population(name="SNr_caudate", geometry = 2, neuron=LinearNeuron_trace)
    SNr_caudate.tau = 10.0
    SNr_caudate.noise = 0.3
    SNr_caudate.baseline = 1.5 
    SNr_caudate.tau_trace = 200.
    #SNr_caudate.slope =1.0

    # STN
    STN_caudate0 = Population(name="STN_caudate0", geometry = (4,4), neuron=LinearNeuron)
    STN_caudate0.tau = 10.0
    STN_caudate0.noise = 0.01
    STN_caudate0.baseline = 0.0

    STN_caudate1 = Population(name="STN_caudate1", geometry = (4,4), neuron=LinearNeuron)
    STN_caudate1.tau = 10.0
    STN_caudate1.noise = 0.01
    STN_caudate1.baseline = 0.0

    # GPe
    GPe_caudate = Population(name="GPe_caudate", geometry = 2, neuron=LinearNeuron)
    GPe_caudate.tau = 10.0
    GPe_caudate.noise = 0.05
    GPe_caudate.baseline = 1.0



    # VA TALAMO EN CAUDATE
    VA_caudate = Population(name="VA_caudate", geometry=2, neuron=LinearNeuron)
    VA_caudate.tau = 10.0
    VA_caudate.noise = 0.05
    VA_caudate.baseline = 0.0

    # PFC INPUT DEL TALAMO
    # Pre frontal cortex
    PFC_caudate = Population(name="PFC_caudate", geometry=2, neuron=LinearNeuron)
    PFC_caudate.tau = 10.0
    PFC_caudate.noise = 0.05
    PFC_caudate.baselie = 0.0

    #MOTOR LOOP

    # Striatum direct pathway
    StrD1_putamen = Population(name="StrD1_putamen", geometry=(2),neuron = LinearNeuron_trace)
    StrD1_putamen.tau = 10.0 
    StrD1_putamen.noise = 0.1/2. 
    StrD1_putamen.baseline = 0.0

    # Striatum indirect pathway
    StrD2_putamen = Population(name="StrD2_putamen", geometry = (2,2), neuron=LinearNeuron)
    StrD2_putamen.tau = 10.0
    StrD2_putamen.noise = 0.1/2.
    StrD2_putamen.baseline = 0.0

    # Striatum feedback pathway
    # A que corresponde esto?
    StrThal_putamen = Population(name="StrThal_putamen", geometry = 2, neuron=LinearNeuron)
    StrThal_putamen.tau = 5.0#10.0
    StrThal_putamen.noise = 0.01
    StrThal_putamen.baseline = 0.4

    # SNr
    SNr_putamen = Population(name="SNr_putamen", geometry =2, neuron=LinearNeuron_trace)
    SNr_putamen.tau = 5.0 #10.0
    SNr_putamen.noise = 0.005 
    SNr_putamen.baseline = 1.1 #2.0
    SNr_putamen.tau_trace = 200.

    # STN
    STN_putamen = Population(name="STN_putamen", geometry = (2,2), neuron=LinearNeuron)
    STN_putamen.tau = 10.0
    STN_putamen.noise = 0.01
    STN_putamen.baseline = 0.0

    # GPe
    GPe_putamen = Population(name="GPe_putamen", geometry = 2, neuron=LinearNeuron)
    GPe_putamen.tau = 10.0
    GPe_putamen.noise = 0.001
    GPe_putamen.baseline = 1.0

    # VA TALAMO EN PUTAMEN
    VA_putamen = Population(name="VA_putamen", geometry=2, neuron=LinearNeuron)
    VA_putamen.tau = 8.0
    VA_putamen.noise = 0.0
    VA_putamen.baseline = 0.3


    #REWARD
    #  Sustancia negra
    SNc_put = Population(name='SNc_put',geometry=2,neuron=DopamineNeuron)
    SNc_put.exc_threshold=1.5 #0.8
    SNc_put.baseline = baseline_dopa
    SNc_put.factor_inh = 1.0

    SNc_caud = Population(name='SNc_cau',geometry=2,neuron=DopamineNeuron)
    SNc_caud.baseline = baseline_dopa

    PPTN = Population(name="PPTN", geometry=2, neuron=InputNeuron)
    PPTN.tau = 1.0


    Hippo = Population(name='Hippocampus',geometry=1,neuron=LinearNeuron)
    Hippo.tau = 30
    Hippo.noise = 0.0
    Hippo.baseline =0.0



    ####################################################################################################################################################
    ####################################################################################################################################################


    #SYNAPSES

    #NEW CORTICO-THALAMIC CONNECTION
    #CorticoThalamic = Projection(pre=Input_neurons[0:10],post=VA_putamen,target='exc',synapse=PostCovariance)
    #CorticoThalamic.connect_all_to_all( weights = 0) 
    #CorticoThalamic.tau = 80000
    #CorticoThalamic.regularization_threshold = 3.0
    #CorticoThalamic.threshold_pre = 0.0
    #CorticoThalamic.threshold_post = 0.0

    #Associative loop

    VAPFC_11 = Projection(pre=VA_caudate[0],post=PFC_caudate[0],target='exc')
    VAPFC_11.connect_all_to_all(weights=1.0)
    VAPFC_22 = Projection(pre=VA_caudate[1],post=PFC_caudate[1],target='exc')
    VAPFC_22.connect_all_to_all(weights=1.0)

    PFCVA_11 = Projection(pre=PFC_caudate[0],post=VA_caudate[0],target="exc")
    PFCVA_11.connect_all_to_all(weights = 0.35) #0.15
    PFCVA_22 = Projection(pre=PFC_caudate[1],post=VA_caudate[1],target="exc")
    PFCVA_22.connect_all_to_all(weights = 0.35)

    ITPFC = Projection(pre=Input_neurons,post=PFC_caudate,target="exc")#,synapse=PostCovariance)
    ITPFC.connect_all_to_all( weights = Uniform(0.2,0.3)) 
    #ITPFC.tau = 250000
    #ITPFC.regularization_threshold = 3.5
    #ITPFC.tau_alpha = 1.0
    #ITPFC.threshold_post = 0.15
    #ITPFC.threshold_pre = 0.35
    # DIRECTA
    # ? IT es una parte especifica de la corteza
    # BASICAMENTE NOS BRINDA EL RESULTADO DE UN ESTIMULO 
    ITStrD1_caudate0 = Projection(pre=Input_neurons,post=StrD1_caudate0,target='exc',synapse=DAPostCovarianceNoThreshold)
    ITStrD1_caudate0.connect_all_to_all(weights = Normal(0.1,0.02)) #Uniform(0.0,0.2)) 
    ITStrD1_caudate0.tau = 100  
    ITStrD1_caudate0.regularization_threshold = 1.0
    ITStrD1_caudate0.tau_alpha = 2.0
    ITStrD1_caudate0.baseline_dopa = baseline_dopa
    ITStrD1_caudate0.K_dip = 0.05
    ITStrD1_caudate0.K_burst = 1.0
    ITStrD1_caudate0.DA_type = 1
    ITStrD1_caudate0.threshold_pre = 0.35 
    ITStrD1_caudate0.threshold_post = 0.0#0.15

    ITStrD1_caudate1 = Projection(pre=Input_neurons,post=StrD1_caudate1,target='exc',synapse=DAPostCovarianceNoThreshold)
    ITStrD1_caudate1.connect_all_to_all(weights = Normal(0.1,0.02)) #Uniform(0.0,0.2)) 
    ITStrD1_caudate1.tau = 100  
    ITStrD1_caudate1.regularization_threshold =  1.0
    ITStrD1_caudate1.tau_alpha = 2.0
    ITStrD1_caudate1.baseline_dopa = baseline_dopa
    ITStrD1_caudate1.K_dip = 0.05
    ITStrD1_caudate1.K_burst = 1.0
    ITStrD1_caudate1.DA_type = 1
    ITStrD1_caudate1.threshold_pre = 0.35 
    ITStrD1_caudate1.threshold_post = 0.0#0.15

    #OBjetivo con Striatum directo D1
    ObjStrD1_caudate0 = Projection(pre=Objectives_extra[0:5],post=StrD1_caudate0,target='exc')
    ObjStrD1_caudate0.connect_all_to_all(weights=0.2)

    ObjStrD1_caudate1 = Projection(pre=Objectives_extra[5:10],post=StrD1_caudate1,target='exc')
    ObjStrD1_caudate1.connect_all_to_all(weights=0.2)

    ObjStrD1_caudate01 = Projection(pre=Objectives_extra[0:5],post=StrD1_caudate1,target='inh')
    ObjStrD1_caudate01.connect_all_to_all(weights=0.6)
    ObjStrD1_caudate10 = Projection(pre=Objectives_extra[5:10],post=StrD1_caudate0,target='inh')
    ObjStrD1_caudate10.connect_all_to_all(weights=0.6)
    # Porque lo inhibe?
    # inhibicion de striaum a sustancia negra
    StrD1SNr_caudate0 = Projection(pre=StrD1_caudate0,post=SNr_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
    StrD1SNr_caudate0.connect_all_to_all(weights=Normal(0.2,0.01))# Uniform(0.05,0.15)) 
    StrD1SNr_caudate0.tau = 550 
    StrD1SNr_caudate0.regularization_threshold = 10.5
    StrD1SNr_caudate0.tau_alpha = 20.0
    StrD1SNr_caudate0.baseline_dopa = 2*baseline_dopa
    StrD1SNr_caudate0.K_dip = 0.9
    StrD1SNr_caudate0.K_burst = 1.0#1.2 
    StrD1SNr_caudate0.threshold_post = 0.3 
    StrD1SNr_caudate0.threshold_pre = 0.15
    StrD1SNr_caudate0.DA_type=1
    StrD1SNr_caudate0.negg = 5.0

    StrD1SNr_caudate1 = Projection(pre=StrD1_caudate1,post=SNr_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
    StrD1SNr_caudate1.connect_all_to_all(weights=Normal(0.2,0.01)) #Uniform(0.05,0.15)) 
    StrD1SNr_caudate1.tau = 550 
    StrD1SNr_caudate1.regularization_threshold = 10.5
    StrD1SNr_caudate1.tau_alpha = 20.0
    StrD1SNr_caudate1.baseline_dopa = 2*baseline_dopa
    StrD1SNr_caudate1.K_dip = 0.9
    StrD1SNr_caudate1.K_burst = 1.0#1.2 
    StrD1SNr_caudate1.threshold_post = 0.3 
    StrD1SNr_caudate1.threshold_pre = 0.15
    StrD1SNr_caudate1.DA_type=1
    StrD1SNr_caudate1.negg = 5.0
    # INDIRECTA
    #? 
    ITStrD2_caudate0 = Projection(pre=Input_neurons,post=StrD2_caudate0,target='exc',synapse=DAPostCovarianceNoThreshold)
    ITStrD2_caudate0.connect_all_to_all(weights = Normal(0.01,0.005)) #Uniform(0.01,0.015)) 
    ITStrD2_caudate0.tau = 10.0
    ITStrD2_caudate0.regularization_threshold = 1.5
    ITStrD2_caudate0.tau_alpha = 1.0
    ITStrD2_caudate0.baseline_dopa = baseline_dopa
    ITStrD2_caudate0.K_dip = 0.2#0.2
    ITStrD2_caudate0.K_burst = 1.0#1.0
    ITStrD2_caudate0.DA_type = -1
    ITStrD2_caudate0.threshold_pre = 0.2 #0.25
    ITStrD2_caudate0.threshold_post = 0.05

    ITStrD2_caudate1 = Projection(pre=Input_neurons,post=StrD2_caudate1,target='exc',synapse=DAPostCovarianceNoThreshold)
    ITStrD2_caudate1.connect_all_to_all(weights = Normal(0.01,0.005)) #Uniform(0.01,0.015)) 
    ITStrD2_caudate1.tau = 10.0
    ITStrD2_caudate1.regularization_threshold = 1.5
    ITStrD2_caudate1.tau_alpha = 1.0
    ITStrD2_caudate1.baseline_dopa = baseline_dopa
    ITStrD2_caudate1.K_dip = 0.2#0.2
    ITStrD2_caudate1.K_burst = 1.0#1.0
    ITStrD2_caudate1.DA_type = -1
    ITStrD2_caudate1.threshold_pre = 0.2 #0.25
    ITStrD2_caudate1.threshold_post = 0.05
    # Objetivo con Striatum indirecto D2
    ObjStrD2_caudate0 = Projection(pre=Objectives_extra[0:5],post=StrD2_caudate0,target='exc')
    ObjStrD2_caudate0.connect_all_to_all(weights = 0.2)
    ObjStrD2_caudate1 = Projection(pre=Objectives_extra[0:5],post=StrD2_caudate1,target='exc')
    ObjStrD2_caudate1.connect_all_to_all(weights = 0.2)

    # Inhibicion de Striatum indirecto D2 a GPe
    StrD2GPe_caudate0 = Projection(pre=StrD2_caudate0,post=GPe_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
    StrD2GPe_caudate0.connect_all_to_all(weights=0.01) 
    StrD2GPe_caudate0.tau = 600 #600.0
    StrD2GPe_caudate0.regularization_threshold = 1.5
    StrD2GPe_caudate0.tau_alpha = 20.0
    StrD2GPe_caudate0.baseline_dopa = 2*baseline_dopa
    StrD2GPe_caudate0.K_dip = 0.1#0.1
    StrD2GPe_caudate0.K_burst = 1.2#1.2
    StrD2GPe_caudate0.threshold_post = 0.0
    StrD2GPe_caudate0.threshold_pre = 0.2
    StrD2GPe_caudate0.DA_type = -1

    StrD2GPe_caudate1 = Projection(pre=StrD2_caudate1,post=GPe_caudate,target='inh',synapse=DAPreCovariance_inhibitory)
    StrD2GPe_caudate1.connect_all_to_all(weights=0.01) 
    StrD2GPe_caudate1.tau = 600 #600.0
    StrD2GPe_caudate1.regularization_threshold = 1.5
    StrD2GPe_caudate1.tau_alpha = 20.0
    StrD2GPe_caudate1.baseline_dopa = 2*baseline_dopa
    StrD2GPe_caudate1.K_dip = 0.1#0.1
    StrD2GPe_caudate1.K_burst = 1.2#1.2
    StrD2GPe_caudate1.threshold_post = 0.0
    StrD2GPe_caudate1.threshold_pre = 0.2
    StrD2GPe_caudate1.DA_type = -1
    # Hiperdirecta
    ITSTN_caudate0 = Projection(pre=Input_neurons, post=STN_caudate0, target='exc')#,synapse=DAPostCovarianceNoThreshold)
    ITSTN_caudate0.connect_all_to_all(weights = Uniform(0.0,0.001)) 
    ITSTN_caudate0.tau = 1500.0 #1000
    ITSTN_caudate0.regularization_threshold = 1.0
    ITSTN_caudate0.tau_alpha = 1.0
    ITSTN_caudate0.baseline_dopa = baseline_dopa
    ITSTN_caudate0.K_dip = 0.4#0.4
    ITSTN_caudate0.K_burst = 1.0#1.0
    ITSTN_caudate0.DA_type = 1
    ITSTN_caudate0.threshold_pre = 0.15

    ITSTN_caudate1 = Projection(pre=Input_neurons, post=STN_caudate1, target='exc')#,synapse=DAPostCovarianceNoThreshold)
    ITSTN_caudate1.connect_all_to_all(weights = Uniform(0.0,0.001)) 
    ITSTN_caudate1.tau = 1500.0 #1000
    ITSTN_caudate1.regularization_threshold = 1.0
    ITSTN_caudate1.tau_alpha = 1.0
    ITSTN_caudate1.baseline_dopa = baseline_dopa
    ITSTN_caudate1.K_dip = 0.4#0.4
    ITSTN_caudate1.K_burst = 1.0#1.0
    ITSTN_caudate1.DA_type = 1
    ITSTN_caudate1.threshold_pre = 0.15
    # Objetivos con STN
    ObjSTN_caudate0 = Projection(pre=Objectives_extra[0:5], post=STN_caudate0, target='exc')
    ObjSTN_caudate0.connect_all_to_all(weights = 0.2)
    ObjSTN_caudate1 = Projection(pre=Objectives_extra[5:10], post=STN_caudate1, target='exc',synapse=DAPostCovarianceNoThreshold)
    ObjSTN_caudate1.connect_all_to_all(weights = 0.2)




    STNSNr_caudate0 = Projection(pre=STN_caudate0,post=SNr_caudate,target='exc')#,synapse=DAPreCovariance_excitatory)
    STNSNr_caudate0.connect_all_to_all(weights=Uniform(0.0012,0.0014)) 
    STNSNr_caudate0.tau = 9000# 8000 900.0
    STNSNr_caudate0.regularization_threshold = 1.5
    STNSNr_caudate0.tau_alpha = 1.0
    STNSNr_caudate0.baseline_dopa = 2*baseline_dopa
    STNSNr_caudate0.K_dip = 0.4#0.4
    STNSNr_caudate0.K_burst = 1.0#1.0
    STNSNr_caudate0.thresholdpost =-0.15
    STNSNr_caudate0.DA_type = 1

    STNSNr_caudate1 = Projection(pre=STN_caudate1,post=SNr_caudate,target='exc')#,synapse=DAPreCovariance_excitatory)
    STNSNr_caudate1.connect_all_to_all(weights=Uniform(0.0012,0.0014)) 
    STNSNr_caudate1.tau = 9000# 8000 900.0
    STNSNr_caudate1.regularization_threshold = 1.5
    STNSNr_caudate1.tau_alpha = 1.0
    STNSNr_caudate1.baseline_dopa = 2*baseline_dopa
    STNSNr_caudate1.K_dip = 0.4#0.4
    STNSNr_caudate1.K_burst = 1.0#1.0
    STNSNr_caudate1.thresholdpost =-0.15
    STNSNr_caudate1.DA_type = 1

    weight_local_inh = 0.8
    StrD1StrD1_caudate0 = Projection(pre=StrD1_caudate0,post=StrD1_caudate0,target='inh')
    StrD1StrD1_caudate0.connect_all_to_all(weights = weight_local_inh)
    StrD1StrD1_caudate1 = Projection(pre=StrD1_caudate1,post=StrD1_caudate1,target='inh')
    StrD1StrD1_caudate1.connect_all_to_all(weights = weight_local_inh)

    weight_stn_inh = 0.3
    STNSTN_caudate0 = Projection(pre=STN_caudate0,post=STN_caudate0,target='inh')
    STNSTN_caudate0.connect_all_to_all(weights = weight_stn_inh)
    STNSTN_caudate1 = Projection(pre=STN_caudate1,post=STN_caudate1,target='inh')
    STNSTN_caudate1.connect_all_to_all(weights = weight_stn_inh)







    PFCPFC_caudate = Projection(pre=PFC_caudate,post = PFC_caudate,target='inh')
    PFCPFC_caudate.connect_all_to_all(weights = 0.12)

    weight_inh_sd2 = 0.5 #0.5
    StrD2StrD2_caudate0 = Projection(pre=StrD2_caudate0,post=StrD2_caudate0,target='inh')
    StrD2StrD2_caudate0.connect_all_to_all(weights=weight_inh_sd2)
    StrD2StrD2_caudate1 = Projection(pre=StrD2_caudate1,post=StrD2_caudate1,target='inh')
    StrD2StrD2_caudate1.connect_all_to_all(weights=weight_inh_sd2)

    StrThalStrThal_caudate = Projection(pre=StrThal_caudate,post=StrThal_caudate,target='inh')
    StrThalStrThal_caudate.connect_all_to_all(weights=0.5)

    SNrSNr_caudate = Projection(pre=SNr_caudate,post=SNr_caudate,target='exc',synapse=ReversedSynapse)
    SNrSNr_caudate.connect_all_to_all(weights=0.8)
    SNrSNr_caudate.reversal = 0.4

    VAVA_caudate = Projection(pre=VA_caudate,post=VA_caudate,target='inh')
    VAVA_caudate.connect_all_to_all(weights=1.1)







    GPeSNr_caudate = Projection(pre=GPe_caudate,post=SNr_caudate,target='inh')
    GPeSNr_caudate.connect_one_to_one(weights=1.0) #0.9

    StrThalGPe_caudate = Projection(pre=StrThal_caudate,post=GPe_caudate,target='inh')
    StrThalGPe_caudate.connect_one_to_one(weights=0.3) 

    StrThalSNr_caudate = Projection(pre=StrThal_caudate,post=SNr_caudate,target='inh')
    StrThalSNr_caudate.connect_one_to_one(weights=0.85) #1.1

    SNrVA_caudate = Projection(pre=SNr_caudate,post=VA_caudate,target='inh')
    SNrVA_caudate.connect_one_to_one(weights=2.0)

    VAObj_caudate = Projection(pre=VA_caudate,post=Objectives[1:3],target='exc')
    VAObj_caudate.connect_one_to_one(weights=2.0) #1.6

    ObjStrThal_caudate = Projection(pre=Objectives[1:3],post=StrThal_caudate,target='exc')
    ObjStrThal_caudate.connect_one_to_one(weights=1.2)
    #VAStrThal_caudate = Projection(pre=VA_caudate,post=StrThal_caudate,target='exc')
    #VAStrThal_caudate.connect_one_to_one(weights=1.2)



    #Motor loop
    # SEGUNDO LOOP
    # Objetivos con Striatum indirecto D2
    ObjStrD2_putamen = Projection(pre=Objectives[1:3],post=StrD2_putamen,target='exc',synapse=DAPostCovarianceNoThreshold)
    ObjStrD2_putamen.connect_all_to_all(weights = Uniform(0.3,0.4))
    #ObjStrD2_putamen.connect_all_to_all(weights = Uniform(0.15,0.2)) 
    ObjStrD2_putamen.tau = 60.0
    ObjStrD2_putamen.regularization_threshold = 1.0
    ObjStrD2_putamen.tau_alpha = 1.0
    ObjStrD2_putamen.baseline_dopa = 2*baseline_dopa
    ObjStrD2_putamen.K_dip = 0.4
    ObjStrD2_putamen.K_burst = 1.0
    ObjStrD2_putamen.DA_type = -1

    # CONEXION ESTIMULOS SONOROS CON STRIATUM DIRECTO, INDIRECTO E HIPERDIRECTO
    # SEGUNDO LOOP
    #########################################################################################
    TonesStrD2_putamen = Projection(pre=Input_neurons_tones[1:4],post=StrD2_putamen,target='exc',synapse=DAPostCovarianceNoThreshold)
    TonesStrD2_putamen.connect_all_to_all(weights = Uniform(0.3,0.4))
    #TonesStrD2_putamen.connect_all_to_all(weights = Uniform(0.15,0.2)) 
    TonesStrD2_putamen.tau = 60.0
    TonesStrD2_putamen.regularization_threshold = 1.0
    TonesStrD2_putamen.tau_alpha = 1.0
    TonesStrD2_putamen.baseline_dopa = 2*baseline_dopa
    TonesStrD2_putamen.K_dip = 0.4
    TonesStrD2_putamen.K_burst = 1.0
    TonesStrD2_putamen.DA_type = -1


    TonesSTN_putamen = Projection(
        pre=Input_neurons_tones[1:4],
        post=STN_putamen,
        target='exc',
        synapse=DAPostCovarianceNoThreshold
    )
    num_pre = 3   # tonos 1, 2, 3
    num_post = STN_putamen.size  # 4 (2x2)

    # Crear matriz
    weights_matrix = np.random.uniform(0.2, 0.3, (num_post, num_pre))

    # Aumentar la influencia del tono 3 (columna 2)
    weights_matrix[:, 2] *= multSTN
    TonesSTN_putamen.connect_from_matrix(weights_matrix)

    # Parametros adicionales
    TonesSTN_putamen.tau = 1000.0
    TonesSTN_putamen.regularization_threshold = 0.4
    TonesSTN_putamen.tau_alpha = 1.0
    TonesSTN_putamen.baseline_dopa = 2 * baseline_dopa
    TonesSTN_putamen.K_dip = 0.4
    TonesSTN_putamen.K_burst = 0.6
    TonesSTN_putamen.DA_type = 1
    TonesSTN_putamen.threshold_pre = 0.15




    TonesStrD1_putamen = Projection(pre=Input_neurons_tones[1:4],post=StrD1_putamen,target='exc',synapse=DAPostCovarianceNoThreshold_trace)
    TonesStrD1_putamen.connect_all_to_all(weights = Normal(0.7,0.01))
    TonesStrD1_putamen.tau = 600.0
    TonesStrD1_putamen.regularization_threshold = 0.9
    TonesStrD1_putamen.tau_alpha = 15
    TonesStrD1_putamen.baseline_dopa = 2*baseline_dopa
    TonesStrD1_putamen.K_dip = 0.025
    TonesStrD1_putamen.K_burst = 0.6 
    TonesStrD1_putamen.DA_type = 1
    TonesStrD1_putamen.threshold_pre = 0.1
    TonesStrD1_putamen.threshold_post = 0.0



    #########################################################################################

    # Objetivos con Nucleo Subtalamico Hiperdirecto
    ObjSTN_putamen = Projection(pre=Objectives[1:3], post=STN_putamen, target='exc',synapse=DAPostCovarianceNoThreshold)
    ObjSTN_putamen.connect_all_to_all(weights = Uniform(0.2,0.3))
    #ObjSTN_putamen.connect_all_to_all(weights = Uniform(0.1,0.15)) 
    ObjSTN_putamen.tau = 1000.0
    ObjSTN_putamen.regularization_threshold = 0.4
    ObjSTN_putamen.tau_alpha = 1.0
    ObjSTN_putamen.baseline_dopa = 2*baseline_dopa
    ObjSTN_putamen.K_dip = 0.4
    ObjSTN_putamen.K_burst = 0.6
    ObjSTN_putamen.DA_type = 1
    ObjSTN_putamen.threshold_pre = 0.15

    # Inhibiciones entre las mismas poblaciones
    StrD1StrD1_putamen = Projection(pre=StrD1_putamen,post=StrD1_putamen,target='inh')
    StrD1StrD1_putamen.connect_all_to_all(weights = 0.95) #1.0

    STNSTN_putamen = Projection(pre=STN_putamen,post=STN_putamen,target='inh')
    STNSTN_putamen.connect_all_to_all(weights = 0.3)

    StrD2StrD2_putamen = Projection(pre=StrD2_putamen,post=StrD2_putamen,target='inh')
    StrD2StrD2_putamen.connect_all_to_all(weights=0.3)

    StrThalStrThal_putamen = Projection(pre=StrThal_putamen,post=StrThal_putamen,target='inh')
    StrThalStrThal_putamen.connect_all_to_all(weights=0.9)

    #Porque excita a la sustancia negra?
    SNrSNr_putamen = Projection(pre=SNr_putamen,post=SNr_putamen,target='exc',synapse=ReversedSynapse)
    SNrSNr_putamen.connect_all_to_all(weights=0.9) #0.2 #0.7
    # 0.6
    # VAVA?
    VAVA_putamen = Projection(pre=VA_putamen,post=VA_putamen,target='inh')
    VAVA_putamen.connect_all_to_all(weights=0.3) #0.2
    #0.9 id 0
    #0.6 id 
    #0.3 id 1
    #0.2 id 
    # Conexion Objetivo con Striatum directo D1
    ObjStrD1_putamen = Projection(pre=Objectives[1:3],post=StrD1_putamen,target='exc',synapse=DAPostCovarianceNoThreshold_trace)
    ObjStrD1_putamen.connect_all_to_all(weights = Normal(0.55,0.01))  #Normal(0.25,0.01))
    #ObjStrD1_putamen.connect_all_to_all(weights = Normal(0.35,0.01)) 
    ObjStrD1_putamen.tau = 600.0
    ObjStrD1_putamen.regularization_threshold = 0.9
    ObjStrD1_putamen.tau_alpha = 15
    ObjStrD1_putamen.baseline_dopa = 2*baseline_dopa
    ObjStrD1_putamen.K_dip = 0.025
    ObjStrD1_putamen.K_burst = 0.6 
    ObjStrD1_putamen.DA_type = 1
    ObjStrD1_putamen.threshold_pre = 0.1
    ObjStrD1_putamen.threshold_post = 0.0#0.4

    # 50 50 entre la eleccion 
    # que los pesos tengan sentido


    # A QUE CORRESPONDE EL OBJETIVO 0?
    Obj0StrD1 = Projection(pre=Objectives[0],post=StrD1_putamen,target='exc')
    Obj0StrD1.connect_all_to_all(weights = 0.0) #Weird
    # Una matriz 2x2 con elementos entre 0 - 1 (la division por dos hace que sean entre 0.5 y 0)
    weights_snr = np.random.rand(2,2)/2.
    #actions = np.random.randint(4,size=16)
    # ACCIONES EN ESTE CASO DEBERIA SER UN RANDOM ENTRE 0 Y 1   
    # weights_snr= [[0.5,0.5],[0.5,0.5] como maximo puede agarrar ese valor
    actions = [0,1]
    # actions puede ser [0,1] o [1,0]
    actions = np.random.permutation(actions)
    # range(2) = [0,1]
    # actions = [0,1] o [1,0]
    # Operacion a base de indexación de numpy 
    # Puedes obtener un aumento de 0.6 en las posiciones [0,0] y [1,1] de la matriz
    # O un aumento de 0.6 en las posiciones [0,1] y [1,0] de la matriz
    weights_snr[actions,range(2)] += 0.1 # 0.3 estaba antes
    # entonces lo maximo que puede tener es 0.6

    #weights_snr=[[0.6,0.6],[0.6,0.6]]

    # Inhibicion de Striatum directo D1 a sustancia negra
    StrD1SNr_putamen = Projection(pre=StrD1_putamen,post=SNr_putamen,target='inh',synapse=DAPreCovariance_inhibitory_trace)
    #StrD1SNr_putamen.connect_all_to_all(weights=Uniform(0.0,2.0)) #0.45,0.55
    StrD1SNr_putamen.connect_from_matrix(weights_snr)
    StrD1SNr_putamen.tau = 850.0 #250
    StrD1SNr_putamen.regularization_threshold = 10.5
    StrD1SNr_putamen.tau_alpha = 20.0
    StrD1SNr_putamen.baseline_dopa = 2*baseline_dopa
    StrD1SNr_putamen.K_dip = 0.005#0.01
    StrD1SNr_putamen.K_burst = 0.5#1.0
    StrD1SNr_putamen.threshold_post = 0.0#0.28#0.01
    StrD1SNr_putamen.threshold_pre = 0.1
    StrD1SNr_putamen.DA_type=1
    #StrD1SNr_putamen.connect_one_to_one(weights=2.0)



    # Exitacion de STN a sustancia negra
    STNSNr_putamen = Projection(pre=STN_putamen,post=SNr_putamen,target='exc',synapse=DAPreCovariance_excitatory)
    STNSNr_putamen.connect_all_to_all(weights=Uniform(0.2,0.225)) 
    STNSNr_putamen.tau = 1000.0
    STNSNr_putamen.regularization_threshold = 1.3
    STNSNr_putamen.tau_alpha = 1.0
    STNSNr_putamen.baseline_dopa = 2*baseline_dopa
    STNSNr_putamen.K_dip = 0.4#0.4
    STNSNr_putamen.K_burst = 0.8 #1.0
    STNSNr_putamen.thresholdpost = 0.15
    STNSNr_putamen.DA_type = 1
    #STNSNr_putamen.connect_one_to_one(weights=0.8)

    # Inhibicion de D2 a GPe indirecta
    StrD2GPe_putamen = Projection(pre=StrD2_putamen,post=GPe_putamen,target='inh',synapse=DAPreCovariance_inhibitory)
    StrD2GPe_putamen.connect_all_to_all(weights=Uniform(0.0,0.0001)) 
    StrD2GPe_putamen.tau = 300.0
    StrD2GPe_putamen.regularization_threshold = 2.0
    StrD2GPe_putamen.tau_alpha = 1.0
    StrD2GPe_putamen.baseline_dopa = 2*baseline_dopa
    StrD2GPe_putamen.K_dip = 0.1#0.4
    StrD2GPe_putamen.K_burst = 1.2#1.0
    StrD2GPe_putamen.threshold_post = 0.05 #0.15
    StrD2GPe_putamen.DA_type = -1
    #StrD2GPe_putamen.connect_one_to_one(weights=0.8)

    # Inhibicion de GPE a sustancia negra indirecta
    GPeSNr_putamen = Projection(pre=GPe_putamen,post=SNr_putamen,target='inh')
    GPeSNr_putamen.connect_one_to_one(weights=0.1)

    # ??? 
    StrThalGPe_putamen = Projection(pre=StrThal_putamen,post=GPe_putamen,target='inh')
    StrThalGPe_putamen.connect_one_to_one(weights=0.15) 

    StrThalSNr_putamen = Projection(pre=StrThal_putamen,post=SNr_putamen,target='inh')
    StrThalSNr_putamen.connect_one_to_one(weights=0.8) #OPTUNA
    #0.1
    # ???
    SNrVA_putamen = Projection(pre=SNr_putamen,post=VA_putamen,target='inh')
    SNrVA_putamen.connect_one_to_one(weights=1.0) #0.95

    GroupStrD1_putamen = Projection(pre=Input_neurons_reversal_group_strd1[1:3],post=StrD1_putamen,target='exc')
    GroupStrD1_putamen.connect_one_to_one(weights = 1.0)

    VAPM_putamen = Projection(pre=VA_putamen,post=PM,target='exc')
    VAPM_putamen.connect_one_to_one(weights=0.8)

    VAStrThal_putamen = Projection(pre=PM,post=StrThal_putamen,target='exc')
    VAStrThal_putamen.connect_one_to_one(weights=1.0)

    #Reward system
    # A QUE CORRESPONDE EL REWARD SYSTEM? SNc?
    SNcStrD1_put = Projection(pre=SNc_put,post=StrD1_putamen,target='dopa')
    SNcStrD1_put.connect_all_to_all(weights=1.0)

    SNcStrD2_put = Projection(pre=SNc_put,post=StrD2_putamen,target='dopa')
    SNcStrD2_put.connect_all_to_all(weights=1.0)

    SNcSNr_put = Projection(pre=SNc_put,post=SNr_putamen,target='dopa')
    SNcSNr_put.connect_all_to_all(weights=1.0)

    SNcSTN_put = Projection(pre=SNc_put,post=STN_putamen,target='dopa')
    SNcSTN_put.connect_all_to_all(weights=1.0)

    SNcGPe_put = Projection(pre=SNc_put,post=GPe_putamen,target='dopa')
    SNcGPe_put.connect_all_to_all(weights=1.0)


    #### fijarse aqui ####
    PropSNc = Projection(pre=Propio[0:2],post=SNc_put,target='exc')
    PropSNc.connect_one_to_one(weights=3.0) #???




    StrD1SNc_put = Projection(pre=StrD1_putamen,post=SNc_put,target='inh',synapse=DAPrediction)
    StrD1SNc_put.connect_all_to_all(weights=0.0)
    StrD1SNc_put.tau = 12000 
    StrD1SNc_put.baseline_dopa = 0.1

    SNcStrD1_caud0 = Projection(pre=SNc_caud[0],post=StrD1_caudate0,target='dopa')
    SNcStrD1_caud0.connect_all_to_all(weights=1.0)
    SNcStrD1_caud1 = Projection(pre=SNc_caud[1],post=StrD1_caudate1,target='dopa')
    SNcStrD1_caud1.connect_all_to_all(weights=1.0)

    SNcStrD2_caud0 = Projection(pre=SNc_caud[0],post=StrD2_caudate0,target='dopa')
    SNcStrD2_caud0.connect_all_to_all(weights=1.0)
    SNcStrD2_caud1 = Projection(pre=SNc_caud[1],post=StrD2_caudate1,target='dopa')
    SNcStrD2_caud1.connect_all_to_all(weights=1.0)


    SNcSNr_caud = Projection(pre=SNc_caud,post=SNr_caudate,target='dopa')
    SNcSNr_caud.connect_all_to_all(weights=1.0)

    SNcSTN_caud0 = Projection(pre=SNc_caud[0],post=STN_caudate0,target='dopa')
    SNcSTN_caud0.connect_all_to_all(weights=1.0)
    SNcSTN_caud1 = Projection(pre=SNc_caud[1],post=STN_caudate1,target='dopa')
    SNcSTN_caud1.connect_all_to_all(weights=1.0)

    SNcGPe_caud = Projection(pre=SNc_caud,post=GPe_caudate,target='dopa')
    SNcGPe_caud.connect_all_to_all(weights=1.0)

    PPTNSNc = Projection(pre=PPTN,post=SNc_caud,target='exc')
    PPTNSNc.connect_one_to_one(weights=1.0)

    StrD1SNc_caud0 = Projection(pre=StrD1_caudate0,post=SNc_caud[0],target='inh',synapse=DAPrediction)
    StrD1SNc_caud0.connect_all_to_all(weights=0.0)
    StrD1SNc_caud0.tau = 3000 #2000 1200 8000.0

    StrD1SNc_caud1 = Projection(pre=StrD1_caudate1,post=SNc_caud[1],target='inh',synapse=DAPrediction)
    StrD1SNc_caud1.connect_all_to_all(weights=0.0)
    StrD1SNc_caud1.tau = 3000 #2000 1200 8000.0

    #Cortical connections

    #PropioObj = Projection(pre=Propio[0:2],post=Input_neurons_tones[1:3],target='exc')
    PropioObj = Projection(pre=Propio[0:2],post=Objectives[1:3],target='exc')
    PropioObj.connect_one_to_one(weights=1.2)

    ObjObj = Projection(pre=Objectives[1:3],post=Objectives[1:3],target='inh')
    ObjObj.connect_all_to_all(weights=0.5)

    #TonesTones = Projection(pre=Input_neurons_tones[1:3],post=Input_neurons_tones[1:3],target='inh')
    #TonesTones.connect_all_to_all(weights=0.5)

    PMPM = Projection(pre=PM,post=PM,target='inh')
    PMPM.connect_all_to_all(weights = 0.4)#0.03


    #Hippo
    HippoVA = Projection(pre=Hippo,post=VA_putamen[1],target='exc')
    HippoVA.connect_all_to_all(weights = 1.0)


    #IL
    ILVA = Projection(pre=IL,post=VA_putamen,target='exc')
    ILVA.connect_one_to_one(weights=1.0)

    VAIL = Projection(pre=VA_putamen,post=IL,target='exc')#,synapse=PreCovariance)
    VAIL.connect_one_to_one(weights=0.05)
    #VAIL.tau = 5000 #120000
    #VAIL.threshold_pre = 0.0
    #VAIL.threshold_post = 0.0
    #VAIL.regularization_threshold = 3.0

    # AQUI SE UTILIZA LA PRE covarianza
    CorticoIL = Projection(pre=Input_neurons[0:10],post=IL,target='exc')#,synapse=PreCovariance)
    CorticoIL.connect_all_to_all( weights = 0.6) 
    CorticoIL.tau = 9000
    CorticoIL.regularization_threshold = 3.0
    CorticoIL.threshold_pre = 0.0
    CorticoIL.threshold_post = 0.0

    SaturationIL0 = Projection(pre=Saturation,post=IL[0],target='exc')
    SaturationIL0.connect_all_to_all(weights = 0.4)
    SaturationIL1 = Projection(pre=Saturation,post=IL[1],target='inh')
    SaturationIL1.connect_all_to_all(weights = 0.4)

    ######################################################################################################################################################

    #bloque ruidos y tau

    PM.noise = ruidoPM
    VA_putamen.noise = ruidoTalamo
    SNr_putamen.noise = ruidoSNr
    GPe_putamen.noise = ruidoGPe
    StrD2_putamen.noise = ruidoStrD2
    StrD1_putamen.noise = ruidoStrD1
    STN_putamen.noise = ruidoSTN
    Input_neurons_tones.noise = ruidoTonos
    Objectives.noise = ruidoObjetivos

    
    ObjStrD2_putamen.tau = tauObjStrD2
    ObjStrD1_putamen.tau = tauObjStrD1
    TonesStrD2_putamen.tau = tauTonesStrD2
    TonesStrD1_putamen.tau = tauTonesStrD1
    TonesSTN_putamen.tau = tauTonesSTN
    StrD2GPe_putamen.tau = tauStrD2GPe
    GPeSNr_putamen.tau = tauGPeSNr
    StrD1SNr_putamen.tau = tauStrD1SNr
    STNSNr_putamen.tau = tauSTNSNr
    SNrVA_putamen.tau = tauSNrVA
    VAPM_putamen.tau = tauVAPM


    compile()



    # Total de experimentos
    num_trials = 701

    
    stim = 0
    trial_stim = [0,0]





    # Contador que indica si el trial fue correcto
    a=0
    cond=0
    cond_nogo=0   #cuando esta en 1 , el siguiente trial lleva tono medio, indicando no-go trial

    weight_trial=0
    grupo=0# Grupo no reversible 0, reversible 1


    for trial in range(num_trials):
        
        # SE ESTABLECE TODO EN 0
        Input_neurons.baseline = 0.0
        # NO TOMARLO EN CUENTA
        #Context.baseline = 0.0
        Input_neurons.r = 0.0
        Input_neurons_tones.r = 0.0
        Input_neurons_tones.baseline = 0.0
        Input_neurons_reversal_group_strd1.baseline = 0.0
        Input_neurons_reversal_group_VA.baseline = 0.0
        # OBJ 2 (dorsolateral) 
        Objectives.r = 0.0
        #Objectives.baseline = 0.3
        Objectives.baseline = 0
        # OBJ 1 (dorsomedial)
        Objectives_extra.baseline=0.0
        Propio.baseline = 0.0
        SNc_caud.baseline = baseline_dopa
        Saturation.baseline = 0.0
        PM.baseline= 0.0



        # implementacion reseteo ruido
        PM.noise = 0.0
        VA_putamen.noise = 0.0
        SNr_putamen.noise = 0.0
        GPe_putamen.noise = 0.0
        StrD2_putamen.noise = 0.0
        StrD1_putamen.noise = 0.0
        STN_putamen.noise = 0.0
        Input_neurons_tones.noise = 0.0
        Objectives.noise = 0.0


        
        simulate(3000)    #cooldown entre trials
        

        
        if(trial==weight_trial):
            weight_trial+=10
        
        
        if(trial==200):   #ultimo ensayo de adquisicion
                
                disable_learning()
                
                
                if TonesStrD1_putamen.w[0][0]>TonesStrD1_putamen.w[0][1] and TonesStrD1_putamen.w[1][1]>TonesStrD1_putamen.w[1][0]:  #mapeo 00_11

                    # AZUL 
                    # REVERSIBLE
                    # AZUL (1) -> ROJO (2)
                    # ROJO (2) -> AZUL (1)
                    
                    # NO REVERSIBLE
                    # AZUL (1) -> AZUL (1)
                    # ROJO (2) -> ROJO (2)
                    #group_map=1
                    time.sleep(3)
                else:           # MAPEO 01_10
                    
                    # REVERSIBLE
                    # ROJO (2) -> ROJO (2)
                    time.sleep(3)
                    # NO REVERSIBLE
                    # ROJO (2) -> ROJO (1)
                
                #break
                pass
        
        elif(trial<200): #Ensayos de adquisicion
            
        
            # estimulo visual
            Input_neurons[15:18].baseline = 0.7

            # OBjetivo del primer loop
            Objectives_extra[0:5].baseline=1.2

            # POSICION DE LA MANO (señales del cerebro que indican la posicion de la mano)
            Propio.r = 0
            Propio[2].baseline = 1.0

            simulate(600)

            # SALIDA DEL SEGUNDO LOOP
            response = PM.r

            # Distribución de probabilidades
            softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
            r = np.random.random()
            # Se elige una accion
            action = -1
            sum_probs = 0
            
            for i in range(2):
                sum_probs += softmax[i]  
                if r< sum_probs and action<0:
                    action = i
                if sum_probs==2:
                    action = np.random.choice([0, 1], p=[0.5, 0.5])

            Propio[2].baseline=0.0      # Indica cual es el estado alcanzado
            
            # Se modifica el estado del propio
            Propio[action].baseline=1.0

            # FEEDback del PM
            PM[action].baseline = 1.0
            
            # ESTIMULO SONORO
            # si la accion es 0 se activa el estimulo 1
            # si la accion es 1 se activa el estimulo 2    

            #se anota los estimulos recibidos por el sujeto cada vez que escoge uno
            if(action==0):
                Input_neurons_tones[1].baseline = 1.7#1.0
            else:
                Input_neurons_tones[2].baseline = 1.7#1.0


            simulate(40)

            # Si el objetivo alcanzado y el estimulos son iguales
            # Si stim == 1 significa que el trial fue equivocado
            if(stim==0):
                PPTN[action].baseline = 0.5
                
            else:
                PPTN.baseline = 0.0

            
            SNc_caud.firing = 1
            SNc_put.firing = 1
        



            simulate(100)
    
            #reset
            PPTN.baseline = 0.0
            SNc_caud.firing = 0
            SNc_put.firing = 0

            
            
        elif (trial<301):       # Experimento 1  - GRUPO NO REVERSIBLE  

            # reimplementacion ruido
            PM.noise = ruidoPM
            VA_putamen.noise = ruidoTalamo
            SNr_putamen.noise = ruidoSNr
            GPe_putamen.noise = ruidoGPe
            StrD2_putamen.noise = ruidoStrD2
            StrD1_putamen.noise = ruidoStrD1
            STN_putamen.noise = ruidoSTN
            Input_neurons_tones.noise = ruidoTonos
            Objectives.noise = ruidoObjetivos

            # estimulo visual
            Input_neurons[15:18].baseline = 0.7

            # OBjetivo del primer loop
            Objectives_extra[0:5].baseline=1.2

            # Se elige un estimulo
            if cond==0:
                cond=1
                Input_neurons_tones[1].baseline = 1.7# activa tono azul 
                tonoActual = 0
                if TonesStrD1_putamen.w[0][0]>TonesStrD1_putamen.w[0][1] and TonesStrD1_putamen.w[1][1]>TonesStrD1_putamen.w[1][0]:
                    
                    Input_neurons_reversal_group_strd1[1].baseline = 1.6 #2.05
                    pass
                        
                else:
                    Input_neurons_reversal_group_strd1[2].baseline = 1.6
                    pass
                
            else:
                cond=0
                Input_neurons_tones[2].baseline = 1.7 # activa tono rojo 
                tonoActual = 1
                if TonesStrD1_putamen.w[0][0]>TonesStrD1_putamen.w[0][1] and TonesStrD1_putamen.w[1][1]>TonesStrD1_putamen.w[1][0]:

                    Input_neurons_reversal_group_strd1[2].baseline = 1.6
                    pass
                        
                else:
                    Input_neurons_reversal_group_strd1[1].baseline = 1.6
                    pass
                
            responded = False
            
            for t in range(tiempoTrial):
                simulate(1)
                response = PM.r
                if np.any(response>= umbralRespuesta):
                    listaTiemposGlobal1norev.append(t)
                    tiemposNoRev.append(t)
                    responded = True
                    break
            response = PM.r

            if responded:

                # Distribución de probabilidades
                softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
                r = np.random.random()
                # Se elige una accion
                action = -1
                sum_probs = 0
                
                for i in range(2):
                    sum_probs += softmax[i]  
                    if r< sum_probs and action<0:
                        action = i
                    if sum_probs==2:
                        action = np.random.choice([0, 1], p=[0.5, 0.5])
                
                if(action == tonoActual):
                    listaCoincidencia1noRev.append("coincide")
                    contadorCoincidencia1NoRev += 1
                else:
                    listaCoincidencia1noRev.append("no coincide")
                    contadorNoCoincide1NoRev += 1
                    if contadorNoCoincide1NoRev > 10:
                        return -10000

            else:
                listaCoincidencia1noRev.append("skip malo")
                contadorSkipMalo1NoRev += 1
                listaTiemposGlobal1norev.append(tiempoTrial)
                action = 2
                if contadorSkipMalo1NoRev > 5:
                    return -10000
                
            if trial == 300 and contadorCoincidencia1NoRev < 88:
                return -10000

        elif (trial<401):       # Experimento 1  - GRUPO REVERSIBLE     

            # reimplementacion ruido
            PM.noise = ruidoPM
            VA_putamen.noise = ruidoTalamo
            SNr_putamen.noise = ruidoSNr
            GPe_putamen.noise = ruidoGPe
            StrD2_putamen.noise = ruidoStrD2
            StrD1_putamen.noise = ruidoStrD1
            STN_putamen.noise = ruidoSTN
            Input_neurons_tones.noise = ruidoTonos
            Objectives.noise = ruidoObjetivos

            # estimulo visual
            Input_neurons[15:18].baseline = 0.7

            # OBjetivo del primer loop
            Objectives_extra[0:5].baseline=1.2
                
            # Se elige un estimulo
            if cond==0:
                cond=1
                Input_neurons_tones[1].baseline = 1.7# activa tono azul 
                tonoActual = 0
                if TonesStrD1_putamen.w[0][0]>TonesStrD1_putamen.w[0][1] and TonesStrD1_putamen.w[1][1]>TonesStrD1_putamen.w[1][0]:
                    
                    Input_neurons_reversal_group_strd1[2].baseline = 1.6 #2.05
                    pass
                    
                else:
                    Input_neurons_reversal_group_strd1[1].baseline = 1.6
                    pass
                
            else:
                cond=0
                Input_neurons_tones[2].baseline = 1.7 # activa tono rojo 
                tonoActual = 1
                if TonesStrD1_putamen.w[0][0]>TonesStrD1_putamen.w[0][1] and TonesStrD1_putamen.w[1][1]>TonesStrD1_putamen.w[1][0]:

                    Input_neurons_reversal_group_strd1[1].baseline = 1.6
                    pass
                    
                else:
                    Input_neurons_reversal_group_strd1[2].baseline = 1.6
                    pass
                
            responded = False
                        
            for t in range(tiempoTrial):
                simulate(1)
                response = PM.r

                if np.any(response>= umbralRespuesta):
                    listaTiemposGlobal1Rev.append(t)
                    tiemposNoRev.append(t)
                    responded = True    
                    break      
                
            response = PM.r

            if responded:
                # Distribución de probabilidades
                softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
                r = np.random.random()
                # Se elige una accion
                action = -1
                sum_probs = 0
                
                for i in range(2):
                    sum_probs += softmax[i]  
                    if r< sum_probs and action<0:
                        action = i
                    if sum_probs==2:
                        action = np.random.choice([0, 1], p=[0.5, 0.5])

                if(action == tonoActual):
                    listaCoincidencia1Rev.append("coincide")
                    contadorCoincidencia1Rev += 1
                    if contadorCoincidencia1Rev > 10:
                        return -10000
                else:
                    listaCoincidencia1Rev.append("no coincide")

            else:
                listaCoincidencia1Rev.append("skip malo")
                contadorSkipMalo1Rev += 1
                listaTiemposGlobal1Rev.append(tiempoTrial)
                action = 2
                if contadorSkipMalo1Rev > 5:
                    return -10000
                
            if trial == 400:
                if contadorCoincidencia1Rev > 12:
                    return -10000
                promedioTiemposNoRev = statistics.mean(tiemposNoRev)
                promedioTiemposRev = statistics.mean(tiemposRev)
                if promedioTiemposNoRev > promedioTiemposRev:
                    return -10000

            pass

        elif (trial<501):       # Experimento 2

            # reimplementacion ruido
            PM.noise = ruidoPM
            VA_putamen.noise = ruidoTalamo
            SNr_putamen.noise = ruidoSNr
            GPe_putamen.noise = ruidoGPe
            StrD2_putamen.noise = ruidoStrD2
            StrD1_putamen.noise = ruidoStrD1
            STN_putamen.noise = ruidoSTN
            Input_neurons_tones.noise = ruidoTonos
            Objectives.noise = ruidoObjetivos       
        
            # estimulo visual
            Input_neurons[15:18].baseline = 0.7

            # OBjetivo del primer loop
            Objectives_extra[0:5].baseline=1.2
                
                
            if cond==0:
                cond=1
                Input_neurons_tones[1].baseline = 1.7# activa tono azul 
                tonoActual = 0
                    
            else:
                cond=0
                Input_neurons_tones[2].baseline = 1.7 # activa tono rojo 
                tonoActual = 1

            responded = False
                    
            for t in range(tiempoTrial):
                simulate(1)
                response = PM.r

                if np.any(response>= umbralRespuesta):
                    listaTiemposGlobal2.append(t)
                    responded = True
                    break
                
            response = PM.r

            if responded:

                # Distribución de probabilidades
                softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
                r = np.random.random()
                # Se elige una accion
                action = -1
                sum_probs = 0
                
                for i in range(2):
                    sum_probs += softmax[i]  
                    if r< sum_probs and action<0:
                        action = i
                    if sum_probs==2:
                        action = np.random.choice([0, 1], p=[0.5, 0.5])

                if(action == tonoActual):
                    listaCoincidencia1Rev.append("coincide")
                    contadorCoincidencia2 += 1
                else:
                    listaCoincidencia1Rev.append("no coincide")

            else:
                listaCoincidencia2.append("skip malo")
                contadorSkipMalo2 += 1
                listaTiemposGlobal2.append(tiempoTrial)
                if contadorSkipMalo2 > 5:
                    return -10000
                
            if trial == 500:
                if contadorCoincidencia2 > 64 or contadorCoincidencia2 < 44:
                    return -10000
                

        else:
            if (trial<701):                                 #Experimento 3

                # reimplementacion ruido
                PM.noise = ruidoPM
                VA_putamen.noise = ruidoTalamo
                SNr_putamen.noise = ruidoSNr
                GPe_putamen.noise = ruidoGPe
                StrD2_putamen.noise = ruidoStrD2
                StrD1_putamen.noise = ruidoStrD1
                STN_putamen.noise = ruidoSTN
                Input_neurons_tones.noise = ruidoTonos
                Objectives.noise = ruidoObjetivos


                # estimulo visual
                Input_neurons[15:18].baseline = 0.7

                # OBjetivo del primer loop
                Objectives_extra[0:5].baseline=1.2
                
                # Se elige un estimulo
                if cond_nogo == 1:
                    cond_nogo = 0
                    Input_neurons_tones[3].baseline = 1.7# activa tono medio/no-go trial
                    tonoActual = 2
                
                else:
                    if cond==0:
                        cond=1
                        cond_nogo = 1
                        Input_neurons_tones[1].baseline = 1.7# activa tono azul 
                        tonoActual = 0
                    
                    else:
                        cond=0
                        cond_nogo = 1
                        Input_neurons_tones[2].baseline = 1.7 # activa tono rojo 
                        tonoActual = 1
                    
                responded = False


                for t in range(tiempoTrial):
                    simulate(1)
                    response = PM.r

                    if np.any(response >= umbralRespuesta):                 #umbral de actividad para presionar algun boton
                        listaTiemposGlobal3.append(t)
                        responded = True
                        break
                response = PM.r

                if responded:
                    # Distribución de probabilidades
                    softmax = (response+0.0000001)/(np.sum(response)+0.0000001)
                    r = np.random.random()

                    # Se elige una accion
                    action = -1
                    sum_probs = 0
                
                    for i in range(2):
                        sum_probs += softmax[i]  
                        if r< sum_probs and action<0:
                            action = i
                        if sum_probs==2:
                            action = np.random.choice([0, 1], p=[0.5, 0.5])
                    #print("la accion escogida es: " + str(action))

                else:
                    listaTiemposGlobal3.append(tiempoTrial)
                    action = 2


                if(action == 2): #Si no responde, se revisa si saltó un trial que tenía que saltar(skip bueno), o si saltó uno que no debía(skip malo)
                    if(tonoActual == 2):
                        listaCoincidencia3.append("skip bueno")
                        contadorSkipBueno3 += 1
                    else:
                        listaCoincidencia3.append("skip malo")
                        contadorSkipMalo3 += 1
                        if contadorSkipMalo3 > 10:
                            return -10000

                else:           #si responde
                    if(tonoActual == 2): #se revisa si respondió en lo que debería haber sido un skip
                        listaCoincidencia3.append("skip respondido")
                        contadorSkipRespondido3 += 1
                        if contadorSkipRespondido3 > 10:
                            return -10000

                    elif(action == tonoActual):   #se revisa si la respuesta corresponde con el tono grave / agudo
                        listaCoincidencia3.append("coincide")
                        contadorCoincidencia3 += 1
                    else:
                        listaCoincidencia3.append("no coincide")

                if trial == 700:
                    if contadorCoincidencia3 > 70 or contadorCoincidencia3 < 50:
                        return -10000
                    if contadorCoincidencia3 < contadorCoincidencia2:
                        return -10000
                
                
            pass
        

    return {
        "exp1": {
            "noRev":{
                "coincidenciaNoRev": contadorCoincidencia1NoRev,
                "skipsNoRev": contadorSkipMalo1NoRev,
                "tiempo": promedioTiemposNoRev
                },
            "rev": {
                "coincidenciaRev": contadorCoincidencia1Rev,
                "skipsRev": contadorSkipMalo1Rev,
                "tiempo": promedioTiemposRev
            }
        },
        "exp2": {
            "coincidencia": contadorCoincidencia2,
            "skips": contadorSkipMalo2
        },
        "exp3": {
            "coincidencia": contadorCoincidencia3,
            "skipsBuenos": contadorSkipBueno3,
            "skipsMalos": contadorSkipMalo3,
            "skipsRespondidos": contadorSkipRespondido3
        }
    }

def calculoScore(metrics):

    SCORE_FAIL = -10000
    score = 2000

    # ======================
    # EXPERIMENTO 1 - NO REV
    # ======================

    m1 = metrics["exp1"]["noRev"]["coincidenciaNoRev"]
    s1 = metrics["exp1"]["noRev"]["skipsNoRev"]

    # score
    score -= abs(m1 - 98) * 10
    score -= s1 * 20


    # ======================
    # EXPERIMENTO 1 - REV
    # ======================

    m1r = metrics["exp1"]["rev"]["coincidenciaRev"]
    s1r = metrics["exp1"]["rev"]["skipsRev"]

    # mismas reglas por ahora

    score -= abs(m1r - 2) * 10
    score -= s1r * 20

    # ======================
    # EXPERIMENTO 2
    # ======================

    m2 = metrics["exp2"]["coincidencia"]
    s2 = metrics["exp2"]["skips"]

    # objetivo 54 (asumo % *100)
    score -= abs(m2 - 54) * 10
    score -= s2 * 20

    # ======================
    # EXPERIMENTO 3
    # ======================

    m3 = metrics["exp3"]["coincidencia"]
    good = metrics["exp3"]["skipsBuenos"]
    bad = metrics["exp3"]["skipsMalos"]
    responded = metrics["exp3"]["skipsRespondidos"]

    # coincidencia objetivo 60
    score -= abs(m3 - 60) * 10

    # ratio de skips
    score -= bad * 20
    score -= responded * 20


    return score

def objective(trial):

    tiempoExp = trial.suggest_int("tiempoExp", 100, 1000)
    umbralActividad = trial.suggest_float("umbralActividad", 0.2, 0.75)
    ruidoPM = trial.suggest_float("ruidoPM", 0.01, 4.0)
    ruidoTalamo = trial.suggest_float("ruidoTalamo", 0.0, 15.0)
    ruidoSNr = trial.suggest_float("ruidoSNr", 0.005, 15.0)
    ruidoGPe = trial.suggest_float("ruidoGPe", 0.001, 15.0)
    ruidoStrD2 = trial.suggest_float("ruidoStrD2", 0.05, 15.0)
    ruidoStrD1 = trial.suggest_float("ruidoStrD1", 0.05, 15.0)
    ruidoSTN = trial.suggest_float("ruidoSTN", 0.001, 15.0)
    ruidoTonos = trial.suggest_float("ruidoTonos", 0.0, 15.0)
    ruidoObjetivos = trial.suggest_float("ruidoObjetivos", 0.01, 15.0)
    tauObjStrD2 = trial.suggest_int("tauObjStrD2", 60, 10000)
    tauObjStrD1 = trial.suggest_int("tauObjStrD1", 600, 10000)
    tauTonesStrD2 = trial.suggest_int("tauTonesStrD2", 60, 10000)
    tauTonesStrD1 = trial.suggest_int("tauTonesStrD1", 600, 10000)
    tauTonesSTN = trial.suggest_int("tauTonesSTN", 1000, 10000)
    tauStrD2GPe = trial.suggest_int("tauStrD2GPe", 300, 10000)
    tauGPeSNr = trial.suggest_int("tauGPeSNr", 50, 10000)
    tauStrD1SNr = trial.suggest_int("tauStrD1SNr", 850, 10000)
    tauSTNSNr = trial.suggest_int("tauSTNSNr", 1000, 10000)
    tauSNrVA = trial.suggest_int("tauSNrVA", 50, 10000)
    tauVAPM = trial.suggest_int("tauVAPM", 50, 10000)

    metrics = simular(tiempoExp, umbralActividad, ruidoPM, ruidoTalamo, ruidoSNr, ruidoGPe, ruidoStrD2, ruidoStrD1, ruidoSTN, ruidoTonos, ruidoObjetivos, tauObjStrD2, tauObjStrD1, tauTonesStrD2, tauTonesStrD1, tauTonesSTN, tauStrD2GPe, tauGPeSNr, tauStrD1SNr, tauSTNSNr, tauSNrVA, tauVAPM)

    if metrics == -10000:
        return -10000

    puntaje = calculoScore(metrics)

    return puntaje

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///tesis.db",
        study_name="tesis",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=10)

    print(study.best_value)
    print(study.best_params)

    df = study.trials_dataframe()
    df.to_csv("resultados_optuna.csv", index=False)

    df_validos = df[df["value"] > -10000]
    df_validos.to_csv("resultados_validos.csv", index=False)